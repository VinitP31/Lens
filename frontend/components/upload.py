"""Adding a document, and watching it get indexed.

Two guards, both needed. `st.file_uploader` hands back the same file on every
rerun, so without the session guard one upload is sent four or five times and each
attempt pays for embeddings; the backend's hash check stops a duplicate being
indexed if one arrives anyway.

Indexing runs in the backend and this screen polls, because Streamlit serves a
session on one thread and doing the work here would freeze the page.
"""

import time

import streamlit as st

from config import settings
from frontend import api_client, state

# The stages the backend reports, in order, with wording a reader recognises.
STAGE_TEXT = {
    "queued": "Queued",
    "validating": "Checking the file",
    "extracting": "Reading the text and tables",
    "ocr": "Reading scanned pages",
    "chunking": "Splitting into passages",
    "embedding": "Preparing for search",
    "indexing": "Adding to the index",
    "ready": "Ready",
}

# Rejections the backend can return, in words that say what to do next.
REJECTION_TEXT = {
    "duplicate_document": "That file is already in your library.",
    "file_too_large": "That file is over the 25 MB limit.",
    "too_many_pages": "That file is over the 50 page limit.",
    "encrypted_pdf": "That PDF is password protected, so its text can't be read.",
    "corrupt_file": "That file couldn't be opened as a PDF.",
    "empty_document": "That PDF has no readable text in it.",
    "unreachable": "The backend isn't running. Start it and try again.",
}


def send(name: str, data: bytes) -> str | None:
    """Upload one file. Returns its document id, or None if it was refused.

    The refusal is shown here rather than raised, because a rejected file is an
    ordinary outcome the user can act on, not a failure of the app.
    """
    if state.already_sent(data):
        return None

    try:
        accepted = api_client.upload(name, data)
    except api_client.LensApiError as error:
        # Let it be retried: the guard exists to stop accidental repeats, not a
        # deliberate second attempt after a real failure.
        state.forget_upload(data)
        st.error(REJECTION_TEXT.get(error.code, error.message))
        return None

    return accepted["doc_id"]


def watch(doc_id: str) -> bool:
    """Show indexing progress until it finishes. Returns True when ready.

    Polling blocks this session's thread, which is why the interval is a couple
    of seconds rather than tight - and why the work itself is in the backend.
    """
    with st.status("Adding your document", expanded=True) as box:
        while True:
            try:
                progress = api_client.status(doc_id)
            except api_client.LensApiError as error:
                if error.code == "document_not_found":
                    # The backend rolls a failed document back and removes it, so
                    # a missing row here means indexing failed after it started.
                    box.update(label="That document couldn't be added", state="error")
                    st.caption(
                        "It was removed rather than half-added, so your library is unchanged."
                    )
                    return False
                box.update(label="Lost contact with the backend", state="error")
                st.caption(error.message)
                return False

            stage = progress.get("stage", "queued")
            st.write(STAGE_TEXT.get(stage, stage))

            if stage == "ready":
                box.update(label="Added", state="complete")
                return True

            if progress.get("finished"):
                box.update(label="That document couldn't be added", state="error")
                st.caption(progress.get("failure_reason") or "Indexing stopped before finishing.")
                return False

            time.sleep(settings.STATUS_POLL_SECONDS)
