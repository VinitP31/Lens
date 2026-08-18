"""The Lens screen.

    streamlit run frontend/app.py

Chat-first: this opens straight into a conversation, and an empty library is the
same screen with the input switched off.

Streamlit re-runs this whole file on every interaction, so nothing may be
remembered between runs except transient UI state. The database is the only thing
that is always right.
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

from config import settings  # noqa: E402
from frontend import api_client, state  # noqa: E402
from frontend.components import (
    chat,
    citations,
    context_indicator,
    documents_drawer,
    empty_state,
    sidebar,
)  # noqa: E402

# Wide, because a cited page is shown beside the answer rather than over it, and
# two columns in a centred page leave neither of them readable.
st.set_page_config(page_title="Lens", layout="wide")

# How the width is split when a page is open. The thread keeps the larger share:
# the answer is what is being read, and the page is what is being checked against
# it.
# The page panel is a fixed frame now, so the thread keeps most of the width and
# the panel takes what it needs and no more.
THREAD_SHARE, PAGE_SHARE = 1.6, 1.0

# Without a page open the thread would otherwise stretch to the full window, and
# a line of text that long is hard to read. Empty columns either side hold it to
# a comfortable measure.
MARGIN_SHARE = 0.5


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

    # Chosen by the newest answer, not by a click. Nothing cited - a greeting, a
    # question about the app, a refusal - means nothing to show.
    viewing = citations.cited_page(conversation)

    if viewing:
        thread_column, page_column = st.columns([THREAD_SHARE, PAGE_SHARE], gap="large")
    else:
        # One column of the same measure, so opening a page widens the window's
        # use rather than reflowing everything that was already on screen.
        _, thread_column, _ = st.columns([MARGIN_SHARE, THREAD_SHARE, MARGIN_SHARE])
        page_column = None

    with thread_column:
        context_indicator.render(conversation, documents)

        # Documents only. "New chat" lives in the sidebar beside the chat list,
        # where the rest of the chat controls are, and having it twice on one
        # screen only raises the question of whether the two differ.
        if st.button("Documents", width="stretch"):
            documents_drawer.open_drawer(documents)

        # A frame of its own, the same height as the page panel beside it. The
        # window itself does not scroll: left to grow, the thread moved the whole
        # page under the reader on every answer, so a question could be half in
        # view and its answer half out of it.
        thread_frame = st.container(height=settings.PANEL_HEIGHT, border=False)

    if viewing and page_column is not None:
        with page_column:
            citations.page_panel(viewing)

    # Outside the columns on purpose: Streamlit pins a top-level chat input to the
    # bottom of the window, and one nested in a column scrolls away with the
    # thread.
    question = st.chat_input("Ask about your documents")

    # Filled last so the streamed answer lands at the end of the thread, where the
    # conversation reads to, and the frame is then scrolled to it.
    with thread_frame:
        chat.render_thread(conversation, documents)
        if question:
            chat.ask(conversation, documents, question)

    chat.scroll_to_latest()


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
            # An id from the address bar can name a chat deleted since, so this is
            # reached on an ordinary refresh, not only by a stale tab.
            state.open_chat(None)
            return None
        st.error(error.message)
        return None


main()
