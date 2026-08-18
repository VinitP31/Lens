"""Working out what a message is, and what it means on its own.

Two jobs in one call, because both need the same input - the chat so far plus what
was just typed.

What kind of message: a greeting, a question about the app, or a real question.
Only the last searches anything. Without this, "hi" is searched, matches nothing,
and comes back as "I could not find that in your documents".

What it means alone: "And for part-time employees?" becomes "what is the annual
leave entitlement for part-time employees?" and that is what gets embedded. The
user still sees what they typed.

Nothing important rests on the classification - neither mistake can produce a
wrong answer with a citation. If the call fails the message is searched as typed,
because degrading toward "do not search" would swallow real questions.
"""

import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from backend.errors import MissingApiKeyError
from backend.retrieval.condenser import keeps_specifics
from backend.storage.conversations import (
    INTENT_GREETING,
    INTENT_META,
    INTENT_QUESTION,
    ROLE_USER,
    Message,
)
from config import settings

ChatFunction = Callable[[list[dict[str, str]]], str]

API_KEY_VARIABLE = "OPENAI_API_KEY"

VALID_INTENTS = frozenset({INTENT_GREETING, INTENT_META, INTENT_QUESTION})

# Written to be answered with JSON and nothing else, because the reply is parsed
# rather than read. The three kinds are described by what they are for, not by
# example phrases - a list of examples is a list of things to fail outside of.
ANALYZE_PROMPT = f"""You prepare a user's message for a document search system.

Decide which of three kinds the message is:

"{INTENT_GREETING}" - a greeting, thanks, or small talk. Nothing is being asked \
about the documents.
"{INTENT_META}" - a question about the application itself: which documents are \
loaded, what it can do, how it works.
"{INTENT_QUESTION}" - anything asking about the content of the documents.

Then write the message as a question that stands on its own, using the \
conversation so far to fill in what it leaves out. "And for part-time staff?" \
after a question about annual leave becomes "What is the annual leave \
entitlement for part-time staff?".

Rules for the rewrite:
- Add only what the conversation already established. Never invent a subject, a \
document name, a number or a qualifier.
- Never drop a detail the message already contains. Keep every number, name, \
date, code and qualifier exactly as given. These are what find the right \
passage, and a rewrite without them retrieves the general topic instead of the \
specific fact.
- Do not summarise, shorten or tidy. Filling in what a message leaves out is the \
whole job; removing what it says is not.
- If the message already stands on its own, repeat it unchanged.
- If it is a greeting or about the application, repeat it unchanged.
- Keep the user's own words wherever they still make sense.

Reply with JSON only, in this exact shape:
{{"intent": "...", "standalone": "..."}}"""

# The model is asked for JSON and usually complies. Occasionally it wraps the
# object in a code fence or a sentence, so the object is located rather than the
# whole reply being parsed.
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

# Distinguishes "no scope recorded yet" from "scope was recorded as nothing".
_UNSET = object()


@dataclass(frozen=True)
class Analysis:
    """What the message is, and what it means on its own."""

    intent: str
    standalone: str
    original: str
    # True when the rewrite actually changed something. The UI shows the searched
    # form only when it differs, so the user is never told their own words were
    # rewritten when they were not.
    was_rewritten: bool
    # True when the analysis had to fall back rather than run. Recorded so a
    # provider having a bad day is visible in the trace log instead of looking
    # like every message suddenly being a question.
    degraded: bool = False

    @property
    def needs_search(self) -> bool:
        return self.intent == INTENT_QUESTION


def openai_analyzer() -> ChatFunction:
    """The real utility model, reading its key from the environment."""
    key = os.environ.get(API_KEY_VARIABLE)
    if not key:
        raise MissingApiKeyError(f"set {API_KEY_VARIABLE} in .env")

    from openai import OpenAI

    client = OpenAI(api_key=key)

    def chat(messages: list[dict[str, str]]) -> str:
        response = client.chat.completions.create(
            model=settings.MODEL_UTILITY,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.ANALYZE_MAX_OUTPUT_TOKENS,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()

    return chat


def _history(turns: Sequence[Message]) -> str:
    """The conversation so far, as plain text for the prompt.

    A turn that was searched against a different set of documents is marked.
    Without the marker, a chat that discussed one document, switched to another,
    and then got a bare follow-up would have that follow-up rewritten into a
    question about the old subject and searched in the new documents - a wrong
    answer built out of two correct halves.
    """
    lines: list[str] = []
    previous_scope: list[str] | None | object = _UNSET
    for turn in turns:
        if turn.scope_snapshot is not None:
            if previous_scope is not _UNSET and turn.scope_snapshot != previous_scope:
                lines.append("(the user changed which documents are being searched here)")
            previous_scope = turn.scope_snapshot
        speaker = "User" if turn.role == ROLE_USER else "Assistant"
        lines.append(f"{speaker}: {turn.content}")
    return "\n".join(lines)


def _parse(reply: str, message: str) -> tuple[str, str] | None:
    """Pull the intent and the rewrite out of the reply, or None if unusable."""
    match = JSON_OBJECT.search(reply)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except (ValueError, TypeError):
        return None

    intent = parsed.get("intent")
    standalone = parsed.get("standalone")
    if intent not in VALID_INTENTS:
        return None
    if not isinstance(standalone, str) or not standalone.strip():
        # A missing rewrite is recoverable: the message itself is used. A missing
        # intent is not, because there would be nothing to decide with.
        standalone = message
    return intent, standalone.strip()


def analyze(
    message: str,
    history: Sequence[Message] = (),
    chat: ChatFunction | None = None,
) -> Analysis:
    """Classify a message and rewrite it to stand on its own.

    Never raises. Anything that goes wrong - an unreachable provider, an
    unparseable reply, an intent that is not one of the three - degrades to
    treating the message as a real question, searched exactly as typed.

    That direction is deliberate. Searching something that did not need searching
    costs one wasted lookup and an honest refusal. Not searching something that
    did would silently swallow a real question.
    """
    chat = chat or openai_analyzer()

    conversation = _history(history)
    user_content = (
        f"Conversation so far:\n{conversation}\n\nMessage: {message}"
        if conversation
        else f"Message: {message}"
    )

    try:
        reply = chat(
            [
                {"role": "system", "content": ANALYZE_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
    except Exception:  # noqa: BLE001 - degrade to searching rather than fail
        return Analysis(
            intent=INTENT_QUESTION,
            standalone=message,
            original=message,
            was_rewritten=False,
            degraded=True,
        )

    parsed = _parse(reply, message)
    if parsed is None:
        return Analysis(
            intent=INTENT_QUESTION,
            standalone=message,
            original=message,
            was_rewritten=False,
            degraded=True,
        )

    intent, standalone = parsed

    # Two ways a rewrite is thrown away, both ending with the message searched
    # exactly as typed.
    #
    # A greeting or a question about the app has nothing to search, so a rewrite
    # could only mislead the user about what was done with their words.
    #
    # A rewrite of a real question that dropped a number is searching for the
    # topic rather than the fact. Filling in what a message leaves out is the
    # whole job; removing what it says is not.
    if intent != INTENT_QUESTION or not keeps_specifics(message, standalone):
        standalone = message

    return Analysis(
        intent=intent,
        standalone=standalone,
        original=message,
        was_rewritten=standalone.strip() != message.strip(),
    )
