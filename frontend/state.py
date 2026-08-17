"""Session keys, and the upload guard.

Streamlit re-runs the whole script on every interaction, which decides what may
live here: only transient UI state. Chat history and the document library come
from the backend on every run, because `session_state` is wiped on refresh and a
screen rebuilt from memory would disagree with the database.

Nothing in this module may hold a connection, a client or a cached answer.
"""

import hashlib

import streamlit as st

# Which chat is open. None means "start a new one on the next question".
CURRENT_CONV = "current_conv"

# Hashes of files this session has already sent. The guard below.
SENT_UPLOADS = "sent_uploads"

# A document whose indexing the screen is still watching, so progress survives
# the reruns that polling causes.
WATCHING = "watching"

# Set when the user changes the document selection but has not saved it yet.
SCOPE_DRAFT = "scope_draft"

# A message the screen should show once, after a rerun caused by an action.
NOTICE = "notice"

# Which chat is waiting on a "are you sure?" before being deleted. Lives here
# rather than in the sidebar module because everything that state survives a
# rerun has to be a session key.
DELETE_ARMED = "delete_armed"

DEFAULTS = {
    CURRENT_CONV: None,
    SENT_UPLOADS: set(),
    WATCHING: None,
    SCOPE_DRAFT: None,
    NOTICE: None,
    DELETE_ARMED: None,
}


def init() -> None:
    """Put every key in place. Safe to call at the top of every rerun."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            # A fresh copy per session: a set shared between sessions would leak
            # one user's upload history into another's guard.
            st.session_state[key] = set(value) if isinstance(value, set) else value


def already_sent(data: bytes) -> bool:
    """Whether this exact file has already been sent from this session.

    `st.file_uploader` returns the same file again on every rerun, and a question
    or a button press causes a rerun. Without this, one upload is sent four or
    five times, and each attempt pays for embeddings before the backend's own
    hash check rejects it.

    Both guards are needed. This one stops the request being made; the backend's
    stops a duplicate being indexed if it arrives anyway.
    """
    digest = hashlib.sha256(data).hexdigest()
    if digest in st.session_state[SENT_UPLOADS]:
        return True
    st.session_state[SENT_UPLOADS].add(digest)
    return False


def forget_upload(data: bytes) -> None:
    """Let a file be sent again.

    Used when the send itself failed - the backend was unreachable, say. The
    guard exists to stop accidental repeats, not to stop a deliberate retry.
    """
    st.session_state[SENT_UPLOADS].discard(hashlib.sha256(data).hexdigest())


def notice(message: str, kind: str = "info") -> None:
    """Leave a message for the screen to show after the next rerun."""
    st.session_state[NOTICE] = (kind, message)


def take_notice() -> tuple[str, str] | None:
    """Read and clear the pending message, so it shows once and not again."""
    pending = st.session_state.get(NOTICE)
    st.session_state[NOTICE] = None
    return pending
