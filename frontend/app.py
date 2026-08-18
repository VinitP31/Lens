"""The Lens screen.

    streamlit run frontend/app.py

Chat-first: this opens straight into a conversation. There is no dashboard and no
home page, and an empty library is this same screen with the input switched off,
so a user never has to learn a page they will only see once.

Everything on screen is read from the backend on each run. Streamlit re-runs this
whole file on every interaction, so nothing may be remembered between runs except
transient UI state - which chat is open, which uploads this session has already
sent. The database is the only thing that is always right.
"""

import sys
from pathlib import Path

# Streamlit puts the script's own directory on the import path, not the directory
# it was launched from, so `frontend.api_client` is not importable by default and
# the app fails on its first import with no clue as to why. Added here rather than
# asked of the reader as a PYTHONPATH: `streamlit run frontend/app.py` is the
# command everyone will type, and it has to work.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from frontend import api_client, state  # noqa: E402
from frontend.components import (
    chat,
    citations,
    context_indicator,
    documents_drawer,
    empty_state,
    sidebar,
)  # noqa: E402

st.set_page_config(page_title="Lens", layout="centered")


def main() -> None:
    state.init()

    # Asked once per run. A backend that is not there makes every control on the
    # page fail, so it is worth one call to say so plainly instead.
    try:
        api_client.health()
    except api_client.LensApiError as error:
        empty_state.render_backend_down(error.message)
        return

    try:
        documents = api_client.documents(ready_only=True)
    except api_client.LensApiError as error:
        empty_state.render_backend_down(error.message)
        return

    sidebar.render()

    pending = state.take_notice()
    if pending:
        kind, message = pending
        (st.success if kind == "info" else st.error)(message)

    conversation = _open_conversation()

    # Nothing to search yet. The input would only produce refusals.
    if not documents:
        empty_state.render()
        st.chat_input("Add a document first", disabled=True)
        return

    context_indicator.render(conversation, documents)

    left, right = st.columns([1, 1])
    with left:
        if st.button("Documents", width="stretch"):
            documents_drawer.open_drawer(documents)
    with right:
        if st.button("New chat", width="stretch"):
            st.session_state[state.CURRENT_CONV] = None
            st.rerun()

    chat.render_thread(conversation, documents)

    # Opened by a citation, after the thread is drawn so it sits on top.
    viewing = st.session_state.pop("page_view", None)
    if viewing:
        citations.page_dialog(viewing, documents)

    question = st.chat_input("Ask about your documents")
    if question:
        chat.ask(conversation, documents, question)


def _open_conversation() -> dict | None:
    """The chat currently open, or None when the next question starts a new one.

    A chat id held in session state can name a conversation that has since been
    deleted - in another tab, or before a reload. Rather than failing, that is
    treated as no chat being open.
    """
    conv_id = st.session_state.get(state.CURRENT_CONV)
    if not conv_id:
        return None

    try:
        return api_client.conversation(conv_id)
    except api_client.LensApiError as error:
        if error.code == "conversation_not_found":
            st.session_state[state.CURRENT_CONV] = None
            return None
        st.error(error.message)
        return None


main()
