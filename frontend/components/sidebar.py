"""The sidebar: a new chat, the chats you have had, and removing one.

Documents are deliberately elsewhere - a drawer that opens on demand - so the
sidebar stays one list rather than two competing for a narrow column.

Deleting asks once. The rows are narrow and sit beside the row that opens a chat, so
a stray click is likely and the loss is permanent. One click arms it, a second
carries it out, clicking anything else puts it back.
"""

import streamlit as st

from frontend import api_client, state

# Only ever one chat is armed at a time: arming a second disarms the first, which
# is also what a user expects.
ARMED = state.DELETE_ARMED


# Longest chat title kept on the button. The sidebar is narrow and the delete
# control sits beside it, so anything longer wraps.
LABEL_MAX_CHARS = 26


def render() -> None:
    """Draw the sidebar. Selecting a chat reruns the script with it open."""
    with st.sidebar:
        st.markdown("### Lens")

        if st.button("New chat", width="stretch"):
            # No conversation is created here. An empty chat with no messages is
            # a row nobody asked for; it is created when the first question is
            # actually asked.
            state.open_chat(None)
            st.session_state[ARMED] = None
            st.rerun()

        try:
            chats = api_client.conversations()
        except api_client.LensApiError:
            # The main screen already reports an unreachable backend. Saying so
            # twice on one page is noise.
            return

        if not chats:
            return

        st.caption("Recent")
        current = st.session_state[state.CURRENT_CONV]
        armed = st.session_state.get(ARMED)

        for chat in chats:
            conv_id = chat["conv_id"]

            if conv_id == armed:
                _render_confirm(chat)
                continue

            label = chat.get("title") or "New chat"
            # Short enough to stay on one line in the sidebar. At 34 characters
            # every title wrapped onto two, which made a list of five chats look
            # like a wall and pushed the rest of the sidebar off the screen.
            if len(label) > LABEL_MAX_CHARS:
                label = label[: LABEL_MAX_CHARS - 1].rstrip() + "…"

            open_col, delete_col = st.columns([5, 1], vertical_alignment="center")
            with open_col:
                st.button(
                    label,
                    key=f"conv-{conv_id}",
                    width="stretch",
                    type="primary" if conv_id == current else "secondary",
                    on_click=_open,
                    args=(conv_id,),
                )
            with delete_col:
                st.button(
                    "×",
                    key=f"arm-{conv_id}",
                    help="Delete this chat",
                    width="stretch",
                    on_click=_arm,
                    args=(conv_id,),
                )


def _render_confirm(chat: dict) -> None:
    """The armed state: say what will happen, and let it be called off.

    The wording names what is lost and what is not. Documents are the expensive
    thing in this app and they are never touched by deleting a chat, so saying so
    removes the main reason to hesitate.
    """
    st.caption(f"Delete “{(chat.get('title') or 'this chat')[:30]}”? Your documents are kept.")

    confirm, cancel = st.columns(2)
    with confirm:
        st.button(
            "Delete",
            key=f"del-{chat['conv_id']}",
            width="stretch",
            type="primary",
            on_click=_delete,
            args=(chat["conv_id"],),
        )
    with cancel:
        st.button(
            "Keep",
            key=f"keep-{chat['conv_id']}",
            width="stretch",
            on_click=_disarm,
        )


def _open(conv_id: str) -> None:
    state.open_chat(conv_id)
    # Opening a chat cancels a pending delete on another one: the user has moved
    # on, and leaving it armed means a later stray click deletes something they
    # are no longer looking at.
    st.session_state[ARMED] = None


def _arm(conv_id: str) -> None:
    st.session_state[ARMED] = conv_id


def _disarm() -> None:
    st.session_state[ARMED] = None


def _delete(conv_id: str) -> None:
    """Remove the chat and its messages. Documents are untouched."""
    try:
        api_client.delete_conversation(conv_id)
    except api_client.LensApiError as error:
        state.notice(error.message, kind="error")
        st.session_state[ARMED] = None
        return

    st.session_state[ARMED] = None
    # Deleting the chat that is open would leave the screen pointing at something
    # that no longer exists, so it falls back to a new empty chat.
    if st.session_state.get(state.CURRENT_CONV) == conv_id:
        state.open_chat(None)
    state.notice("Chat deleted.")
