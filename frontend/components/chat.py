"""The message thread, and asking a question.

Two things here are shaped by how Streamlit works rather than by preference.

The thread is redrawn from the backend on every rerun. Streamlit re-runs the
whole script whenever anything is clicked, so a thread held in memory would be
one interaction out of date; the database is the only thing that is always right.

An answer arrives as a stream, and `st.write_stream` consumes it. The final piece
of that stream is not text but the finished answer carrying its citations, which
cannot exist until the model has stopped citing. So the stream is wrapped: text
goes to the screen as it arrives, and the last object is kept back.
"""

import streamlit as st

from frontend import api_client, state
from frontend.components import citations as citations_ui

# What to say when there is nothing to answer from. Distinct from an error: the
# system worked correctly and is telling the truth about what it found.
ABSTAIN_TEXT = "I couldn't find this in your documents."


def render_thread(conversation: dict | None, documents: list[dict]) -> None:
    """Draw every turn of the open chat."""
    if not conversation:
        return

    for index, message in enumerate(conversation.get("messages", [])):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("abstained"):
                _render_abstention(conversation, documents)
                continue

            st.markdown(message["content"])
            citations_ui.render(message.get("citations", []), documents, key_prefix=f"m{index}")


def _render_abstention(conversation: dict | None, documents: list[dict]) -> None:
    """A calm refusal, not an error.

    Suggesting a wider selection is only honest when the selection is narrow. If
    everything is already being searched, telling the user to widen would send
    them looking for a setting that cannot help.
    """
    st.markdown(ABSTAIN_TEXT)

    narrowed = bool(conversation and conversation.get("scope_mode") == "subset")
    if narrowed and documents:
        st.caption(
            "Try rephrasing, or widen the search to the whole knowledge base "
            "using **Change** above."
        )
    else:
        st.caption("Try rephrasing it, or add a document that covers this.")


def ask(conversation: dict | None, documents: list[dict], question: str) -> None:
    """Send a question and show the answer as it arrives.

    Creates the chat if this is the first question. That is deliberate: an empty
    conversation with no messages is a row nobody asked for, and it would appear
    in the sidebar as "New chat" forever.
    """
    conv_id = conversation["conv_id"] if conversation else None

    if conv_id is None:
        draft = st.session_state.get(state.SCOPE_DRAFT)
        try:
            created = api_client.create_conversation(
                scope_mode="subset" if draft else "library",
                scope_doc_ids=draft,
            )
        except api_client.LensApiError as error:
            st.error(error.message)
            return
        conv_id = created["conv_id"]
        st.session_state[state.CURRENT_CONV] = conv_id
        st.session_state[state.SCOPE_DRAFT] = None

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        finished: dict = {}

        def stream():
            """Yield text; keep the final answer object out of the display."""
            for piece in api_client.ask(conv_id, question):
                if isinstance(piece, api_client.Answer):
                    finished["answer"] = piece
                else:
                    yield piece

        try:
            st.write_stream(stream)
        except api_client.LensApiError as error:
            # An error is not an abstention. Saying "not in your documents" when
            # the truth is that a call failed would be the one lie this system
            # exists to avoid.
            st.error(f"{error.message}")
            return

        answer = finished.get("answer")
        if answer is None:
            st.error("The answer did not finish. Ask again.")
            return

        if answer.abstained:
            _render_abstention(conversation, documents)
        else:
            citations_ui.render(answer.citations, documents, key_prefix="live")

    # Redrawn from the backend so the new turn, its title and its stored
    # citations are all read from one source rather than patched into memory.
    st.rerun()
