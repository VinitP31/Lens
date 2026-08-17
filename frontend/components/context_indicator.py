"""What is being searched, shown above the thread and editable there.

This is always visible for one reason: the same question against a different
selection gives a different answer, and a user who cannot see the selection has
no way to understand why. A refusal in particular is only interpretable if you
know what was searched.

Three display states, per the spec: the whole library, one named document, or the
first name and how many others.
"""

import streamlit as st

from frontend import api_client, state

LIBRARY = "library"
SUBSET = "subset"


def describe(scope_mode: str, doc_ids: list[str] | None, names: dict[str, str]) -> str:
    """The label for the current selection.

    Falls back to a count when a name is missing, which happens when a document
    in an old chat's scope has since been removed from the library.
    """
    if scope_mode == LIBRARY or not doc_ids:
        return "Entire knowledge base"

    known = [names.get(doc_id, "a removed document") for doc_id in doc_ids]
    if len(known) == 1:
        return known[0]
    return f"{known[0]} and {len(known) - 1} more"


def render(conversation: dict | None, documents: list[dict]) -> None:
    """Draw the indicator and, when opened, the editor.

    `conversation` is None before the first question, when there is no chat yet.
    The selection still has to be shown and editable, because a user may want to
    narrow it before asking anything.
    """
    names = {document["doc_id"]: document["display_name"] for document in documents}

    scope_mode = LIBRARY
    doc_ids: list[str] | None = None
    if conversation:
        scope_mode = conversation.get("scope_mode", LIBRARY)
        doc_ids = conversation.get("scope_doc_ids")
    elif st.session_state.get(state.SCOPE_DRAFT):
        scope_mode, doc_ids = SUBSET, st.session_state[state.SCOPE_DRAFT]

    left, right = st.columns([5, 1], vertical_alignment="center")
    with left:
        st.caption(f"Searching: **{describe(scope_mode, doc_ids, names)}**")
    with right:
        editing = st.toggle("Change", key="scope-edit", disabled=not documents)

    if not editing:
        return

    chosen = st.multiselect(
        "Search only these documents",
        options=list(names),
        default=doc_ids or [],
        format_func=lambda doc_id: names.get(doc_id, doc_id),
        placeholder="Leave empty to search everything",
        label_visibility="collapsed",
    )

    save, widen = st.columns([1, 1])

    with save:
        if st.button("Save selection", use_container_width=True):
            if not chosen:
                # Refused rather than saved. An empty selection makes every
                # question abstain, which reads as the app being broken rather
                # than as the setting the user chose.
                st.warning(
                    "Choose at least one document, or use "
                    "**Search everything** to include the whole library."
                )
            else:
                _apply(conversation, SUBSET, chosen)

    with widen:
        if st.button("Search everything", use_container_width=True):
            _apply(conversation, LIBRARY, None)


def _apply(conversation: dict | None, scope_mode: str, doc_ids: list[str] | None) -> None:
    """Save the selection, on the chat if one exists or as a draft if not."""
    if conversation is None:
        # No chat yet. Held until the first question creates one, so the choice
        # is not lost by asking.
        st.session_state[state.SCOPE_DRAFT] = doc_ids
        st.rerun()

    try:
        api_client.update_conversation(
            conversation["conv_id"], scope_mode=scope_mode, scope_doc_ids=doc_ids
        )
    except api_client.LensApiError as error:
        st.error(error.message)
        return
    st.rerun()
