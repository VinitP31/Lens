"""The message thread, and asking a question.

Redrawn from the backend on every rerun, because Streamlit re-runs the whole script
whenever anything is clicked.

The last piece of a streamed answer is not text but the finished answer with its
citations, so the stream is wrapped and that object kept back.
"""

import streamlit as st

from frontend import api_client, state
from frontend.components import citations as citations_ui

# What to say when there is nothing to answer from. Distinct from an error: the
# system worked correctly and is telling the truth about what it found.
ABSTAIN_TEXT = "I couldn't find this in your documents."

# What the progress line says at each stage. Plain descriptions of what is
# actually happening, so the wait is explained rather than merely animated.
SEARCHING_TEXT = "Searching your documents"
WRITING_TEXT = "Writing the answer"
DONE_TEXT = "Answered"
NOTHING_FOUND_TEXT = "Nothing close enough found"
FAILED_TEXT = "That did not finish"


def turns(messages: list[dict]) -> list[list[dict]]:
    """Group a flat list of messages into turns, oldest first.

    A turn is a question and the answer to it. Grouping them is what lets the
    thread be drawn as pairs rather than as a flat run of messages.
    """
    grouped: list[list[dict]] = []
    for message in messages:
        if message.get("role") == "user" or not grouped:
            grouped.append([message])
        else:
            grouped[-1].append(message)
    return grouped


def render_thread(conversation: dict | None, documents: list[dict]) -> None:
    """Draw every turn of the open chat, oldest first.

    The order a conversation is read in. The thread sits in a frame of its own,
    which shows its top, so the frame is scrolled to the newest turn after it is
    drawn - see `scroll_to_latest`.
    """
    if not conversation:
        return

    for index, turn in enumerate(turns(conversation.get("messages", []))):
        for message in turn:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("abstained"):
                    _render_abstention(conversation, documents)
                    continue

                st.markdown(message["content"])
                citations_ui.render(message.get("citations", []), documents, key_prefix=f"t{index}")


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
    # The session's own record decides, not the fetched conversation: a fetch that
    # failed looks exactly like "no chat yet", and acting on that starts a second
    # chat holding one turn.
    conv_id = (conversation or {}).get("conv_id") or st.session_state.get(state.CURRENT_CONV)

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
        state.open_chat(conv_id)
        st.session_state[state.SCOPE_DRAFT] = None

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        finished: dict = {}
        # Streamlit serves a session on one thread, so the page is frozen until the
        # answer arrives and a stuck-looking page reads as a broken app. The stages
        # are real: the label changes when the first token arrives.
        progress = st.status(SEARCHING_TEXT, expanded=False)

        def stream():
            """Yield text; keep the final answer object out of the display."""
            writing = False
            for piece in api_client.ask(conv_id, question):
                if isinstance(piece, api_client.Answer):
                    finished["answer"] = piece
                else:
                    if not writing:
                        progress.update(label=WRITING_TEXT)
                        writing = True
                    yield piece

        try:
            st.write_stream(stream)
        except api_client.LensApiError as error:
            # An error is not an abstention.
            progress.update(label=FAILED_TEXT, state="error")
            st.error(f"{error.message}")
            return

        answer = finished.get("answer")
        if answer is None:
            progress.update(label=FAILED_TEXT, state="error")
            st.error("The answer did not finish. Ask again.")
            return

        if answer.abstained:
            # No text was ever streamed, so the label still says "searching".
            progress.update(label=NOTHING_FOUND_TEXT, state="complete")
            _render_abstention(conversation, documents)
        else:
            progress.update(label=DONE_TEXT, state="complete")
            citations_ui.render(answer.citations, documents, key_prefix="live")

    # Redrawn from the backend so the new turn, its title and its stored
    # citations are all read from one source rather than patched into memory.
    st.rerun()


def scroll_to_latest() -> None:
    """Put the newest turn in view inside the thread frame.

    Freshly drawn, the frame shows its oldest turn and the answer just asked for sits
    below the fold. Streamlit has no setting for this, so the last message is scrolled
    into view directly - the frame scrolls, not the window.

    One pixel tall because zero is rejected.
    """
    # `st.iframe` rather than `components.html`, which is deprecated past its
    # removal date. The markup is fixed and written here, never built from anything
    # a user or a document supplied.
    st.iframe(
        """
        <script>
            const doc = window.parent.document;
            const messages = doc.querySelectorAll('[data-testid="stChatMessage"]');
            if (messages.length) {
                messages[messages.length - 1].scrollIntoView({block: "end"});
            }
        </script>
        """,
        height=1,
    )
