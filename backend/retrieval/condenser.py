"""Shortening a very long message before it is searched with.

An embedding is one point for the whole text, so a thousand-word message sits some
distance from every chunk rather than close to the right one. That is not a wrong
match but a weak match against everything, which drags the top score down and can
push an answerable question below the gate.

Above a threshold the message is reduced to the question inside it, and only for
searching: the original is what the user sees and what the answer model is asked.
The UI says it happened, because silently searching for something other than what
was typed is exactly the substitution this project avoids.
"""

import os
import re
from collections.abc import Callable
from dataclasses import dataclass

from backend.errors import MissingApiKeyError
from config import settings

# Anything containing a digit: "100", "12", "FR-01", "31-13-9", "4.2". These are
# what find a specific passage rather than the general topic, so a shortened or
# rewritten question that has lost one is searching for something weaker than
# what was asked.
SPECIFIC = re.compile(r"[\w./-]*\d[\w./-]*")

# Takes the messages, returns the reply as one string. Injected so the suite runs
# offline and free.
ChatFunction = Callable[[list[dict[str, str]]], str]

API_KEY_VARIABLE = "OPENAI_API_KEY"

CONDENSE_PROMPT = """You reduce a long message to the single question it is asking.

Keep every detail that identifies what is being asked about: names, numbers, \
dates, document references, and any qualifier such as part-time, overseas or \
first-year.

Drop greetings, background, apologies and repetition.

Write it as one short, direct question, as somebody would type it into a search \
box. Do not restate the same request twice in different words, and do not append \
a second clause asking for the same thing again. Every extra word makes the \
search worse, even when it is accurate.

Reply with the question alone. Do not answer it, do not explain what you removed, \
and do not add anything the message does not say."""


@dataclass(frozen=True)
class Condensed:
    """The text to search with, and whether it differs from what was typed."""

    text: str
    original: str
    was_condensed: bool


def keeps_specifics(original: str, rewritten: str) -> bool:
    """Whether a rewrite kept every number and code the original had.

    Shared with the analyzer, since both restate a question and both fail the same
    way. Measured: a message mentioning a total of 100 points was rewritten without
    the number, and the passage holding the figures dropped out of the results
    entirely. Instructions alone did not fix it reliably, so the decision is made
    here - a rewrite that lost a specific is discarded.

    Only losses matter; a rewrite that adds a number is the prompt's problem.
    """
    lost = set(SPECIFIC.findall(original)) - set(SPECIFIC.findall(rewritten))
    return not lost


def needed(message: str) -> bool:
    """Whether this message is long enough to be worth reducing."""
    return len(message) > settings.CONDENSE_CHAR_THRESHOLD


def openai_condenser() -> ChatFunction:
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
            max_tokens=settings.CONDENSE_MAX_OUTPUT_TOKENS,
            messages=messages,
        )
        return (response.choices[0].message.content or "").strip()

    return chat


def condense(message: str, chat: ChatFunction | None = None) -> Condensed:
    """Reduce a long message to the question inside it.

    A short message is returned untouched without any call being made.

    A failure here is not fatal. If the model is unreachable or returns nothing
    usable, the original message is searched with instead: a slightly weaker
    search is a far better outcome than refusing to answer at all, and the user
    still gets the same answer path.
    """
    if not needed(message):
        return Condensed(text=message, original=message, was_condensed=False)

    chat = chat or openai_condenser()
    try:
        reduced = chat(
            [
                {"role": "system", "content": CONDENSE_PROMPT},
                {"role": "user", "content": message},
            ]
        ).strip()
    except Exception:  # noqa: BLE001 - degrade to the original rather than fail
        return Condensed(text=message, original=message, was_condensed=False)

    # An empty reply, or one longer than what it was meant to shorten, is not a
    # reduction. Using it would make the search worse than doing nothing.
    if not reduced or len(reduced) >= len(message):
        return Condensed(text=message, original=message, was_condensed=False)

    # A reduction that dropped a number is searching for something weaker than
    # what was asked. Better a long message than a short wrong one.
    if not keeps_specifics(message, reduced):
        return Condensed(text=message, original=message, was_condensed=False)

    return Condensed(text=reduced, original=message, was_condensed=True)
