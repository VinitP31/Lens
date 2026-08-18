"""Session keys, and the upload guard.

Only transient UI state lives here. `session_state` is wiped on refresh, so chat
history and the document library come from the backend on every run, and nothing
here may hold a connection, a client or a cached answer.
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

# The page shown beside the thread is chosen by the answer, not by a click: the
# newest answer's first source appears there on its own. These two keys record the
# reader overruling that - opening a different source, or closing the panel - and
# both are cleared when the next answer arrives, because that answer cites
# something else.
PAGE_VIEW = "page_view"  # a source the reader opened instead
PAGE_CLOSED = "page_closed"  # the automatic page they dismissed
PAGE_ANCHOR = "page_anchor"  # which answer those two were about

DEFAULTS = {
    CURRENT_CONV: None,
    PAGE_VIEW: None,
    PAGE_CLOSED: None,
    PAGE_ANCHOR: None,
    SENT_UPLOADS: set(),
    WATCHING: None,
    SCOPE_DRAFT: None,
    NOTICE: None,
    DELETE_ARMED: None,
}


# Which chat is open, kept in the address bar. Session state is wiped by a
# refresh, and without this the app forgot which chat was on screen: the next
# question started a brand new one, so the sidebar filled with chats the user
# never asked for and a follow-up lost the history it needed to be understood.
CHAT_PARAM = "chat"


def init() -> None:
    """Put every key in place. Safe to call at the top of every rerun."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            # A fresh copy per session: a set shared between sessions would leak
            # one user's upload history into another's guard.
            st.session_state[key] = set(value) if isinstance(value, set) else value

    # A refresh arrives with empty session state and the address bar intact, so
    # this is what carries the open chat across one.
    if st.session_state[CURRENT_CONV] is None:
        from_url = st.query_params.get(CHAT_PARAM)
        # A repeated parameter arrives as a list. Taking the first is what a
        # browser does with `?chat=a&chat=b`, and passing a list on as an id would
        # fail deep inside the API client instead of here.
        if isinstance(from_url, list):
            from_url = from_url[0] if from_url else None
        if from_url:
            st.session_state[CURRENT_CONV] = from_url


def open_chat(conv_id: str | None) -> None:
    """Open a chat, or none, and record it where a refresh will still find it."""
    st.session_state[CURRENT_CONV] = conv_id
    if conv_id:
        st.query_params[CHAT_PARAM] = conv_id
    elif CHAT_PARAM in st.query_params:
        del st.query_params[CHAT_PARAM]


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
