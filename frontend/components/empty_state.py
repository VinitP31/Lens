"""What the screen says before there is anything to search.

The user never leaves the chat screen. There is no home page and no setup step -
an empty library is the chat screen with the input switched off and one clear
thing to do.

The alternative, a separate welcome page, means the first thing anyone learns is
a screen they will never see again.
"""

import streamlit as st

from frontend import state
from frontend.components import upload


def render() -> None:
    """The first-run state: say what this is, and take a document."""
    st.markdown("#### Ask questions about your own documents")
    st.markdown(
        "Add a PDF and ask about what is in it. Every answer names the document "
        "and page it came from, and when the documents do not cover something, "
        "Lens says so rather than guessing."
    )

    chosen = st.file_uploader(
        "Add your first document",
        type="pdf",
        key="first-upload",
        help="Up to 25 MB and 50 pages.",
    )

    if chosen is None:
        return

    data = chosen.getvalue()
    doc_id = upload.send(chosen.name, data)
    if doc_id and upload.watch(doc_id):
        state.notice(f"Added {chosen.name}. Ask it something.")
        st.rerun()


def render_backend_down(message: str) -> None:
    """The backend is not answering. Say what to do, once.

    Shown instead of the chat rather than alongside it: every control on the page
    would fail, and a screen full of controls that do nothing is worse than a
    screen that explains itself.
    """
    st.markdown("#### Lens can't reach its backend")
    st.markdown(
        "The part that reads documents and answers questions isn't running. "
        "Start it, then reload this page:"
    )
    st.code("uvicorn backend.main:app --port 8000", language="bash")
    st.caption(message)
