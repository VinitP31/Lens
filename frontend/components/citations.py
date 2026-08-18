"""The source list under an answer.

Every field shown here was resolved by the backend when the answer was given and
stored on the message. Nothing is looked up again, so a document deleted since
does not change or break an old answer - it is shown as it was, marked as no
longer in the library.

That is the point of the whole product: an answer you can check. So the sources
are always visible, never behind a toggle, and each one names its document and
page before you open it.
"""

import re

import streamlit as st

from config import settings
from frontend import api_client

# Written by the backend on its own line when a table had rows left over.
TRUNCATED_MARK = settings.SNIPPET_TRUNCATED_MARK

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
                _render_snippet(snippet, citation.get("element_type", ""))

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


def as_rows(snippet: str) -> list[list[str]] | None:
    """A markdown table snippet as rows, or None when it is not one.

    A table chunk is stored as markdown, and markdown handed to `st.markdown`
    inside a blockquote arrives as one long line of pipes and dashes - unreadable,
    and the exact case where a reader most wants to check the answer.

    The snippet is cut to a fixed length, so its last row is usually incomplete.
    A short row is padded rather than dropped: the cells that survived are still
    what the answer was read from, and dropping the row would hide them.
    """
    lines = [line.strip() for line in snippet.splitlines() if line.strip()]
    lines = [line for line in lines if line != TRUNCATED_MARK]

    # Answers given before the backend kept a table's line breaks are stored as
    # one long line, and they must still read as a table years later - a stored
    # citation is never resolved again, by design. `| |` is where one row's last
    # cell meets the next row's first.
    if len(lines) == 1:
        lines = [part.strip() for part in re.split(r"\|\s*\|", lines[0]) if part.strip()]
        lines = [line if line.startswith("|") else f"|{line}" for line in lines]

    cells = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in lines
        if line.startswith("|")
    ]
    # The bar under the header row carries no content in any renderer.
    cells = [row for row in cells if not all(set(cell) <= set("-: ") for cell in row)]
    cells = [row for row in cells if "".join(row).strip(TRUNCATED_MARK)]

    if len(cells) < 2:
        return None

    width = max(len(row) for row in cells)
    return [row + [""] * (width - len(row)) for row in cells]


def _render_snippet(snippet: str, element_type: str) -> None:
    """Show the passage the way it reads in the document.

    A table becomes a table, prose stays a quotation. Anything that does not parse
    falls back to plain text rather than to markdown, because markdown is what
    mangles it.
    """
    if element_type == "table":
        rows = as_rows(snippet)
        if rows:
            # Markdown rather than a dataframe: a dataframe cell clips its text at
            # the column width, so a long reason reads as another half sentence -
            # the very thing this is meant to stop. A markdown table wraps.
            header, body = rows[0], rows[1:]
            table = [_row(header), _row(["---"] * len(header)), *(_row(row) for row in body)]
            st.markdown("\n".join(table))
            if snippet.rstrip().endswith(TRUNCATED_MARK):
                st.caption("More rows on the page itself.")
            return
        st.text(snippet)
        return

    st.markdown(f"> {snippet}")


def _row(cells: list[str]) -> str:
    """One markdown row. A pipe inside a cell is escaped, or it splits the row."""
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


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
