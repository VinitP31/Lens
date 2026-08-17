"""The sidebar: a new chat, the chats you have had, and removing one.

Documents are deliberately not here. They live in a drawer that opens on demand,
so the sidebar stays one thing - your history - rather than two lists competing
for the same narrow column.

Everything shown comes from the backend on every rerun. A sidebar rebuilt from
session memory would disagree with the database the moment a chat was renamed by
its first question.

Deleting asks once. The rows are narrow and sit right beside the row you click to
open a chat, so a stray click is likely and the loss is permanent - the messages
and the citations stored on them go with it. One click arms the delete, a second
carries it out, and clicking anything else puts it back.
"""

import streamlit as st

from frontend import api_client, state

# Only ever one chat is armed at a time: arming a second disarms the first, which
# is also what a user expects.
ARMED = state.DELETE_ARMED


def render() -> None:
    """Draw the sidebar. Selecting a chat reruns the script with it open."""
    with st.sidebar:
        st.markdown("### Lens")

        if st.button("New chat", use_container_width=True):
            # No conversation is created here. An empty chat with no messages is
            # a row nobody asked for; it is created when the first question is
            # actually asked.
            st.session_state[state.CURRENT_CONV] = None
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
            # A long question makes an unreadable button in a narrow column, and
            # the delete control needs room beside it.
            if len(label) > 34:
                label = label[:33].rstrip() + "…"

            open_col, delete_col = st.columns([5, 1], vertical_alignment="center")
            with open_col:
                st.button(
                    label,
                    key=f"conv-{conv_id}",
                    use_container_width=True,
                    type="primary" if conv_id == current else "secondary",
                    on_click=_open,
                    args=(conv_id,),
                )
            with delete_col:
                st.button(
                    "×",
                    key=f"arm-{conv_id}",
                    help="Delete this chat",
                    use_container_width=True,
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
            use_container_width=True,
            type="primary",
            on_click=_delete,
            args=(chat["conv_id"],),
        )
    with cancel:
        st.button(
            "Keep",
            key=f"keep-{chat['conv_id']}",
            use_container_width=True,
            on_click=_disarm,
        )


def _open(conv_id: str) -> None:
    st.session_state[state.CURRENT_CONV] = conv_id
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
        st.session_state[state.CURRENT_CONV] = None
    state.notice("Chat deleted.")
