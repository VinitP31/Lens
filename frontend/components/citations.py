"""The source list under an answer.

Every field shown here was resolved by the backend when the answer was given and
stored on the message. Nothing is looked up again, so a document deleted since
does not change or break an old answer - it is shown as it was, marked as no
longer in the library.

That is the point of the whole product: an answer you can check. So the sources
are always visible, never behind a toggle, and each one names its document and
page before you open it.
"""

import streamlit as st

from frontend import api_client

# What the backend calls a chunk's type, and what a reader should see. A table or
# a figure caption is read differently from a sentence, so the label is honest
# about which one the answer leaned on.
KIND_LABEL = {"table": "table", "figure_caption": "figure caption"}


def render(citations: list[dict], documents: list[dict], key_prefix: str) -> None:
    """Draw one expander per source.

    `key_prefix` keeps widget keys unique: Streamlit reruns the whole script, and
    two answers citing the same page would otherwise collide.
    """
    if not citations:
        return

    live = {document["doc_id"] for document in documents}
    st.caption("Sources")

    for citation in citations:
        page = citation.get("page")
        name = citation.get("display_name", "a document")
        removed = citation.get("doc_id") not in live

        heading = f"[{citation.get('n')}] {name} — page {page}"
        kind = KIND_LABEL.get(citation.get("element_type", ""))
        if kind:
            heading += f" ({kind})"
        if removed:
            heading += " — removed from library"

        with st.expander(heading):
            section = citation.get("section_path")
            if section:
                st.caption(section.replace(" > ", " › "))

            snippet = citation.get("snippet")
            if snippet:
                st.markdown(f"> {snippet}")

            if removed:
                st.caption(
                    "This document is no longer in your library. The page can "
                    "still be opened: the original file is kept so that answers "
                    "given before it was removed stay checkable."
                )

            st.button(
                f"View page {page}",
                key=f"{key_prefix}-view-{citation.get('n')}-{citation.get('chunk_id')}",
                on_click=_open_page,
                args=(citation,),
            )


def _open_page(citation: dict) -> None:
    """Open the page viewer.

    The rendered page with its highlight is stage 8 work. Until the backend can
    produce it, the honest thing is to say so rather than open an empty dialog -
    the passage and the page number are already on screen above.
    """
    st.session_state["page_view"] = citation


@st.dialog("Source page", width="large")
def page_dialog(citation: dict, documents: list[dict]) -> None:
    """Show the cited page.

    Kept in this module because it is part of checking a citation, not a separate
    feature. It draws whatever the backend can give it: once page rendering
    exists, the image with its highlight; until then, the passage and where it
    sits.
    """
    st.markdown(f"**{citation.get('display_name')}** — page {citation.get('page')}")
    section = citation.get("section_path")
    if section:
        st.caption(section.replace(" > ", " › "))

    try:
        image = api_client.page_image(
            citation["doc_id"], citation["page"], citation.get("chunk_id")
        )
    except api_client.LensApiError as error:
        # The page could not be drawn - usually the original file is gone. The
        # passage is still shown, because it is what the answer actually used and
        # it is better than an empty frame.
        st.warning(error.message)
        if citation.get("snippet"):
            st.markdown(f"> {citation['snippet']}")
        return

    st.image(image, width="stretch")
    st.caption("The highlighted region is the passage this answer used.")
