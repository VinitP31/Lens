"""Tests for the screen.

The backend is stubbed, so nothing here starts a server or spends money. What is
real is Streamlit: the app script is executed by `AppTest`, so a rerun bug, a
duplicate widget key or an exception in a component shows up here rather than in
the browser.

The upload guard is the most important test in this file. Streamlit hands the same
file back on every rerun, and a rerun happens on every click, so without the guard
one upload is sent four or five times and each attempt pays for embeddings.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from frontend import api_client, state
from frontend.components import citations as citations_ui
from frontend.components import context_indicator

# Absolute: AppTest resolves a relative path against the file that calls it, not
# against the repository root.
APP = str(Path(__file__).resolve().parent.parent / "frontend" / "app.py")

DOCUMENTS = [
    {
        "doc_id": "d1",
        "display_name": "FleetLink TMS User Manual v4.2",
        "status": "ready",
        "page_count": 26,
        "chunk_count": 80,
        "table_count": 4,
        "image_count": 0,
        "size_bytes": 900_000,
        "ocr_applied": False,
        "uploaded_at": "2026-08-17T10:00:00+00:00",
    },
    {
        "doc_id": "d2",
        "display_name": "26-004 CRM Software RFP Package",
        "status": "ready",
        "page_count": 33,
        "chunk_count": 144,
        "table_count": 9,
        "image_count": 1,
        "size_bytes": 1_400_000,
        "ocr_applied": False,
        "uploaded_at": "2026-08-17T10:05:00+00:00",
    },
]

HEALTH = {
    "status": "ok",
    "documents": 2,
    "chunks": 224,
    "embedding_model": "text-embedding-3-small",
    "answer_model": "gpt-4o-mini",
    "embed_model_matches": True,
    "gate_threshold": 0.45,
}


def conversation(messages=None, scope_mode="library", scope_doc_ids=None) -> dict:
    return {
        "conv_id": "c1",
        "title": "Password rules",
        "title_is_auto": True,
        "scope_mode": scope_mode,
        "scope_doc_ids": scope_doc_ids,
        "created_at": "2026-08-17T10:00:00+00:00",
        "updated_at": "2026-08-17T10:10:00+00:00",
        "messages": messages or [],
    }


def message(role, content, citations=None, abstained=False) -> dict:
    return {
        "msg_id": "m1",
        "role": role,
        "content": content,
        "created_at": "2026-08-17T10:00:00+00:00",
        "citations": citations or [],
        "intent": "question",
        "abstained": abstained,
    }


def citation(n=1, doc_id="d1", page=5, kind="text") -> dict:
    return {
        "n": n,
        "chunk_id": f"{doc_id}:12",
        "doc_id": doc_id,
        "display_name": "FleetLink TMS User Manual v4.2",
        "page": page,
        "section_path": "FleetLink > 3. Getting Started > 3.2 Password rules",
        "element_type": kind,
        "snippet": "Passwords must be at least 12 characters…",
        "bboxes": [[72.0, 210.5, 523.0, 288.25]],
    }


@pytest.fixture
def backend(monkeypatch):
    """A stubbed backend. Tests override individual calls as needed."""
    calls = {"uploads": [], "created": 0}

    monkeypatch.setattr(api_client, "health", lambda: HEALTH)
    monkeypatch.setattr(api_client, "documents", lambda ready_only=False: list(DOCUMENTS))
    monkeypatch.setattr(api_client, "conversations", lambda: [])
    monkeypatch.setattr(api_client, "conversation", lambda conv_id: conversation())

    def create(scope_mode="library", scope_doc_ids=None):
        calls["created"] += 1
        return conversation(scope_mode=scope_mode, scope_doc_ids=scope_doc_ids)

    monkeypatch.setattr(api_client, "create_conversation", create)

    def upload(name, data):
        calls["uploads"].append((name, len(data)))
        return {"doc_id": "new", "job_id": "j1", "display_name": name, "page_count": 3}

    monkeypatch.setattr(api_client, "upload", upload)
    monkeypatch.setattr(api_client, "status", lambda doc_id: {"stage": "ready", "finished": True})
    return calls


# --- the upload guard ----------------------------------------------------


def test_the_same_file_is_only_sent_once():
    """Streamlit re-returns the file on every rerun, and a rerun happens on every
    click. Without the guard one upload becomes four or five, and each attempt
    pays for embeddings before the backend's own hash check rejects it."""
    import streamlit as st

    st.session_state[state.SENT_UPLOADS] = set()
    data = b"%PDF-1.7 pretend"

    seen = [state.already_sent(data) for _ in range(4)]

    assert seen == [False, True, True, True]


def test_a_different_file_is_not_blocked():
    import streamlit as st

    st.session_state[state.SENT_UPLOADS] = set()

    assert state.already_sent(b"%PDF one") is False
    assert state.already_sent(b"%PDF two") is False


def test_a_failed_send_can_be_retried():
    """The guard stops accidental repeats, not a deliberate retry after a real
    failure."""
    import streamlit as st

    st.session_state[state.SENT_UPLOADS] = set()
    data = b"%PDF-1.7 pretend"

    assert state.already_sent(data) is False
    state.forget_upload(data)

    assert state.already_sent(data) is False


def test_a_rejected_upload_is_reported_and_can_be_sent_again(monkeypatch):
    """A refusal is an ordinary outcome the user can act on, not a crash."""
    import streamlit as st

    from frontend.components import upload

    st.session_state[state.SENT_UPLOADS] = set()

    def refuse(name, data):
        raise api_client.LensApiError("duplicate_document", "already in the library")

    monkeypatch.setattr(api_client, "upload", refuse)
    shown = []
    monkeypatch.setattr(upload.st, "error", shown.append)

    assert upload.send("a.pdf", b"%PDF one") is None
    assert "already in your library" in shown[0]
    # The guard was cleared, so a second attempt actually reaches the backend.
    assert state.already_sent(b"%PDF one") is False


# --- what the screen says with nothing to search -------------------------


def test_an_empty_library_disables_the_input(monkeypatch, backend):
    """Every question would be refused, so the input is switched off rather than
    inviting one."""
    monkeypatch.setattr(api_client, "documents", lambda ready_only=False: [])

    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert at.chat_input[0].disabled
    assert any("Add a PDF" in m.value for m in at.markdown)


def test_an_unreachable_backend_says_what_to_do(monkeypatch):
    """Every control on the page would fail, so the page explains itself instead
    of rendering controls that do nothing."""

    def down():
        raise api_client.LensApiError("unreachable", "connection refused")

    monkeypatch.setattr(api_client, "health", down)

    at = AppTest.from_file(APP, default_timeout=30).run()

    assert not at.exception
    assert any("can't reach its backend" in m.value for m in at.markdown)
    assert not at.chat_input


# --- the thread ----------------------------------------------------------


def test_an_answer_shows_its_sources(monkeypatch, backend):
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[
                message("user", "How long must a password be?"),
                message("assistant", "Twelve characters [1].", citations=[citation()]),
            ]
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    assert not at.exception
    labels = [e.label for e in at.expander]
    assert any("page 5" in label for label in labels)
    assert any("FleetLink" in label for label in labels)


def test_a_table_citation_says_it_is_a_table(monkeypatch, backend):
    """A table is read differently from a sentence, so the label is honest about
    which one the answer leaned on."""
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[
                message("assistant", "Twelve [1].", citations=[citation(kind="table")]),
            ]
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    assert any("(table)" in e.label for e in at.expander)


def test_a_citation_to_a_removed_document_is_marked(monkeypatch, backend):
    """The answer still renders from what was stored, so an old answer does not
    break when a document leaves the library."""
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[
                message("assistant", "Twelve [1].", citations=[citation(doc_id="gone")]),
            ]
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    assert any("removed from library" in e.label for e in at.expander)


def test_a_refusal_is_calm_and_not_an_error(monkeypatch, backend):
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[
                message("user", "Who won the tender?"),
                message("assistant", "", abstained=True),
            ]
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    assert not at.error
    assert any("couldn't find this" in m.value for m in at.markdown)


def test_widening_is_only_suggested_when_the_selection_is_narrow(monkeypatch, backend):
    """Telling a user to widen when everything is already searched sends them
    looking for a setting that cannot help."""
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[message("assistant", "", abstained=True)], scope_mode="library"
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    captions = " ".join(c.value for c in at.caption)
    assert "widen" not in captions.lower()


def test_widening_is_suggested_when_only_some_documents_are_searched(monkeypatch, backend):
    monkeypatch.setattr(api_client, "conversations", lambda: [{"conv_id": "c1", "title": "T"}])
    monkeypatch.setattr(
        api_client,
        "conversation",
        lambda conv_id: conversation(
            messages=[message("assistant", "", abstained=True)],
            scope_mode="subset",
            scope_doc_ids=["d1"],
        ),
    )

    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    captions = " ".join(c.value for c in at.caption)
    assert "widen" in captions.lower()


# --- the sidebar ---------------------------------------------------------


def test_the_sidebar_holds_history_and_never_documents(monkeypatch, backend):
    """Two lists in one narrow column compete for the same space, so documents
    live in a drawer instead."""
    monkeypatch.setattr(
        api_client,
        "conversations",
        lambda: [
            {"conv_id": "c1", "title": "Password rules"},
            {"conv_id": "c2", "title": "Evaluation weighting"},
        ],
    )

    at = AppTest.from_file(APP, default_timeout=30).run()

    labels = [b.label for b in at.sidebar.button]
    assert "New chat" in labels
    assert "Password rules" in labels
    assert not any("FleetLink" in label for label in labels)


def test_a_long_chat_title_is_shortened_in_the_sidebar(monkeypatch, backend):
    long_title = "What is the cumulative target time from gate entry to putaway confirmation?"
    monkeypatch.setattr(
        api_client, "conversations", lambda: [{"conv_id": "c1", "title": long_title}]
    )

    at = AppTest.from_file(APP, default_timeout=30).run()

    shown = [b.label for b in at.sidebar.button if b.label != "New chat"][0]
    assert len(shown) <= 42


# --- the context indicator ----------------------------------------------


def test_the_whole_library_is_described_plainly():
    assert context_indicator.describe("library", None, {}) == "Entire knowledge base"


def test_one_document_is_named():
    names = {"d1": "FleetLink TMS User Manual v4.2"}

    assert context_indicator.describe("subset", ["d1"], names) == "FleetLink TMS User Manual v4.2"


def test_several_documents_name_the_first_and_count_the_rest():
    names = {"d1": "FleetLink", "d2": "CRM RFP", "d3": "Handbook"}

    assert context_indicator.describe("subset", ["d1", "d2", "d3"], names) == "FleetLink and 2 more"


def test_a_document_missing_from_the_library_still_describes():
    """An old chat can name a document that has since been removed."""
    assert "removed" in context_indicator.describe("subset", ["gone"], {})


# --- deleting a chat -----------------------------------------------------
# Deleting is permanent and the control sits beside the one that opens a chat, so
# a stray click is likely. It asks once.


@pytest.fixture
def two_chats(monkeypatch, backend):
    monkeypatch.setattr(
        api_client,
        "conversations",
        lambda: [
            {"conv_id": "c1", "title": "Password rules"},
            {"conv_id": "c2", "title": "Evaluation weighting"},
        ],
    )
    deleted = []
    monkeypatch.setattr(api_client, "delete_conversation", deleted.append)
    return deleted


def test_every_chat_offers_a_delete(two_chats):
    at = AppTest.from_file(APP, default_timeout=30).run()

    keys = [b.key for b in at.sidebar.button]
    assert "arm-c1" in keys
    assert "arm-c2" in keys


def test_one_click_does_not_delete(two_chats):
    """It asks first. A single click on a narrow row must not lose a conversation."""
    at = AppTest.from_file(APP, default_timeout=30).run()

    at.sidebar.button(key="arm-c1").click().run()

    assert two_chats == []
    assert any(b.key == "del-c1" for b in at.sidebar.button)


def test_the_confirmation_says_documents_are_kept(two_chats):
    """The main reason to hesitate is fear of losing the documents, so the wording
    answers it."""
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()

    captions = " ".join(c.value for c in at.sidebar.caption)
    assert "documents are kept" in captions.lower()


def test_confirming_deletes_it(two_chats):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="del-c1").click().run()

    assert two_chats == ["c1"]


def test_keeping_it_calls_the_delete_off(two_chats):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="keep-c1").click().run()

    assert two_chats == []
    # Back to an ordinary row, with its delete offered again.
    assert any(b.key == "arm-c1" for b in at.sidebar.button)


def test_only_one_chat_is_armed_at_a_time(two_chats):
    """Arming a second disarms the first, so there is never more than one row in
    a state where a click destroys something."""
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="arm-c2").click().run()

    keys = [b.key for b in at.sidebar.button]
    assert "del-c2" in keys
    assert "del-c1" not in keys


def test_opening_a_chat_calls_a_pending_delete_off(two_chats):
    """The user moved on. Leaving it armed means a later stray click deletes
    something they are no longer looking at."""
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="conv-c2").click().run()

    assert two_chats == []
    assert not any(b.key == "del-c1" for b in at.sidebar.button)


def test_deleting_the_open_chat_falls_back_to_a_new_one(two_chats):
    """Otherwise the screen points at a conversation that no longer exists."""
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state[state.CURRENT_CONV] = "c1"
    at.run()

    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="del-c1").click().run()

    assert two_chats == ["c1"]
    assert at.session_state[state.CURRENT_CONV] is None


def test_a_failed_delete_is_reported_and_leaves_the_chat_alone(monkeypatch, backend):
    monkeypatch.setattr(
        api_client, "conversations", lambda: [{"conv_id": "c1", "title": "Password rules"}]
    )

    def refuse(conv_id):
        raise api_client.LensApiError("unreachable", "connection refused")

    monkeypatch.setattr(api_client, "delete_conversation", refuse)

    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.button(key="arm-c1").click().run()
    at.sidebar.button(key="del-c1").click().run()

    assert not at.exception
    assert any("connection refused" in e.value for e in at.error)


# --- the app script actually loads --------------------------------------


def test_the_app_script_runs_when_launched_the_way_streamlit_launches_it(tmp_path):
    """Streamlit puts the script's own directory on the import path, not the one it
    was launched from. Without the path line at the top of `app.py`, every import of
    `frontend.…` fails and the page shows a traceback instead of a chat - while the
    server still answers 200, so a status check does not catch it.

    Run in a subprocess from an unrelated directory, with nothing added to the
    import path, because that is the only arrangement in which the bug appears.
    """
    import os
    import subprocess
    import sys

    probe = (
        "from streamlit.testing.v1 import AppTest;"
        f"app = AppTest.from_file({APP!r}, default_timeout=60);"
        "app.run();"
        "print('EXCEPTION' if app.exception else 'CLEAN');"
        "print('WIDGETS', len(app.chat_input))"
    )
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr[-2000:]
    assert "CLEAN" in result.stdout, result.stdout + result.stderr[-2000:]
    # A chat input proves the script reached the end, not merely that it imported.
    assert "WIDGETS 1" in result.stdout, result.stdout


# --- a table citation reads as a table -----------------------------------


TABLE_SNIPPET = (
    "| Rule | Setting |\n"
    "|------|---------|\n"
    "| Minimum length | 12 characters |\n"
    "| Reuse | The last 8 passwords cannot be reused |"
)


def test_a_table_snippet_becomes_rows():
    """Handed to markdown inside a quotation, a table arrives as one long line of
    pipes and dashes - unreadable in the one place a reader is checking the
    answer."""
    rows = citations_ui.as_rows(TABLE_SNIPPET)

    assert rows[0] == ["Rule", "Setting"]
    assert ["Minimum length", "12 characters"] in rows
    # The bar under the header carries no content and must not become a row.
    assert not any(set("".join(row)) <= set("-: ") for row in rows)


def test_a_row_cut_short_by_the_snippet_limit_keeps_what_survived():
    """The snippet is cut to a fixed length, so its last row is usually partial.
    Those cells are still what the answer was read from."""
    rows = citations_ui.as_rows(TABLE_SNIPPET + "\n| Maximum age | 90")

    assert rows[-1] == ["Maximum age", "90"]
    assert all(len(row) == len(rows[0]) for row in rows)


def test_prose_is_not_mistaken_for_a_table():
    assert citations_ui.as_rows("The vehicle is stopped at the outer barrier.") is None


def test_a_single_row_is_not_a_table():
    """One row is a fragment. Rendering it as a table would invent a header."""
    assert citations_ui.as_rows("| Minimum length | 12 characters |") is None


def test_a_table_stored_as_one_line_still_reads_as_a_table():
    """Answers given before the backend kept a table's line breaks are stored flat,
    and a stored citation is never resolved again by design - so an old answer has
    to keep rendering from exactly what it saved."""
    flat = (
        "| Rule | Setting | |-----|-----| | Minimum length | 12 characters | "
        "| Reuse | The last 8 passwords cannot be reused |"
    )

    rows = citations_ui.as_rows(flat)

    assert rows[0] == ["Rule", "Setting"]
    assert ["Minimum length", "12 characters"] in rows
    assert ["Reuse", "The last 8 passwords cannot be reused"] in rows


def test_the_truncation_mark_is_not_shown_as_a_row():
    """It is a message about the table, not a line of it."""
    from config import settings

    snippet = (
        "| Rule | Setting |\n|---|---|\n| Minimum length | 12 characters |\n"
        + settings.SNIPPET_TRUNCATED_MARK
    )

    rows = citations_ui.as_rows(snippet)

    assert rows == [["Rule", "Setting"], ["Minimum length", "12 characters"]]
