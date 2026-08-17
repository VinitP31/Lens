"""The documents drawer: what is in the library, add, remove.

A drawer rather than a permanent panel. Documents are managed occasionally and
read from constantly, so they get space when asked for and none the rest of the
time - which is also why the sidebar holds only history.

Uploading from here changes the library and never the current chat's selection.
A user who opens the drawer is managing documents, not deciding what this
conversation searches.
"""

import streamlit as st

from frontend import api_client, state
from frontend.components import upload


@st.dialog("Documents", width="large")
def open_drawer(documents: list[dict]) -> None:
    """Draw the drawer. Any change reruns the page, which closes it."""
    if documents:
        passages = sum(document.get("chunk_count", 0) for document in documents)
        st.caption(f"{len(documents)} documents · {passages} passages")

    for document in documents:
        row, meta, remove = st.columns([5, 2, 1], vertical_alignment="center")
        with row:
            st.markdown(document["display_name"])
        with meta:
            pages = document.get("page_count") or "?"
            st.caption(f"{pages} pages · {document.get('chunk_count', 0)} passages")
        with remove:
            if st.button("Remove", key=f"rm-{document['doc_id']}", use_container_width=True):
                _remove(document)

    st.divider()

    chosen = st.file_uploader(
        "Add a document",
        type="pdf",
        key="drawer-upload",
        help="Up to 25 MB and 50 pages.",
    )

    if chosen is not None:
        data = chosen.getvalue()
        doc_id = upload.send(chosen.name, data)
        if doc_id and upload.watch(doc_id):
            state.notice(f"Added {chosen.name}.")
            st.rerun()


def _remove(document: dict) -> None:
    """Take a document out of the library.

    The file and its record are kept by the backend, so citations in answers
    already given still show what they showed. Only future searches change.
    """
    try:
        api_client.delete_document(document["doc_id"])
    except api_client.LensApiError as error:
        st.error(error.message)
        return
    state.notice(f"Removed {document['display_name']}.")
    st.rerun()
