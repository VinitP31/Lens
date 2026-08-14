"""Producing one grounded answer.

The gate has already run by the time anything here is called. That ordering is
the point of the whole design, so this module never re-decides whether to answer -
it is reached only when the numbers said yes.

What it does decide is whether what came back is usable. There are three
outcomes, and conflating any two of them would misinform the user:

An answer, with at least one citation code could resolve.

An abstention, because the model reported the passages do not hold the answer, or
because every citation it offered was invented. Both mean the same thing to a
reader - nothing here can be checked - and both are shown as "I don't know".

A failure, because the provider could not be reached. This is not an abstention.
Telling somebody their documents do not cover a question when the truth is that a
network call failed would be a lie in the one place this system exists not to
tell one.

The model call is injected, so the whole suite runs offline for nothing. Tests
here assert on structure and on the three outcomes, never on the wording of an
answer, which varies between runs and would make the suite flaky while proving
nothing.
"""

import os
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from backend.errors import GenerationFailedError, MissingApiKeyError
from backend.retrieval import citations, prompt
from backend.storage.vector_store import Hit
from config import settings

# Takes the assembled messages, yields the answer in pieces as they arrive.
# A stand-in in tests yields from a fixed string.
ChatFunction = Callable[[list[dict[str, str]]], Iterator[str]]

API_KEY_VARIABLE = "OPENAI_API_KEY"

# Why an answer turned into an abstention. Stable strings: the UI maps them to
# wording and the trace log counts them, so neither may depend on a sentence.
REASON_NOT_IN_DOCUMENTS = "not_in_documents"
REASON_NO_VALID_CITATIONS = "no_valid_citations"
REASON_EMPTY_ANSWER = "empty_answer"


@dataclass(frozen=True)
class Answer:
    """One answer, and everything needed to show, check and explain it."""

    text: str
    citations: list[citations.Citation]
    abstained: bool
    # Set only when `abstained` is true.
    reason: str | None = None
    # Numbers the model cited that were never supplied. Reported every turn, not
    # only when something went wrong, because a rate that starts climbing is the
    # first sign the prompt's citation contract has stopped holding.
    fabricated: list[int] = field(default_factory=list)
    prompt_passages: int = 0


def openai_chat() -> ChatFunction:
    """The real answer model, reading its key from the environment.

    Built lazily, so nothing imports a client at module load and the offline
    tools run with no key present.
    """
    key = os.environ.get(API_KEY_VARIABLE)
    if not key:
        raise MissingApiKeyError(f"set {API_KEY_VARIABLE} in .env")

    from openai import OpenAI

    client = OpenAI(api_key=key)

    def chat(messages: list[dict[str, str]]) -> Iterator[str]:
        stream = client.chat.completions.create(
            model=settings.MODEL_ANSWER,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.ANSWER_MAX_OUTPUT_TOKENS,
            messages=messages,
            stream=True,
        )
        for piece in stream:
            if not piece.choices:
                continue
            token = piece.choices[0].delta.content
            if token:
                yield token

    return chat


def _retryable(error: Exception) -> bool:
    """Whether trying again could plausibly succeed.

    Matched on the provider's exception classes, never on message text. A bad key
    or a malformed request fails identically every time, so retrying only delays
    the same failure.
    """
    import openai

    transient = tuple(
        getattr(openai, name)
        for name in (
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
        )
        if hasattr(openai, name)
    )
    return isinstance(error, transient)


def _collect(messages: list[dict[str, str]], chat: ChatFunction) -> str:
    """The whole answer, with retries on transient failures only.

    A stream that fails halfway is retried from the start rather than resumed.
    Half an answer stitched to the start of another is worse than a slower one,
    and there is no way to know the two halves belong together.
    """
    last: Exception | None = None

    for attempt in range(1, settings.ANSWER_MAX_RETRIES + 1):
        try:
            return "".join(chat(messages))
        except Exception as error:  # noqa: BLE001 - re-raised as a typed error below
            last = error
            if attempt == settings.ANSWER_MAX_RETRIES or not _retryable(error):
                break
            time.sleep(settings.ANSWER_RETRY_BACKOFF_SECONDS * attempt)

    raise GenerationFailedError(
        f"generation failed after {settings.ANSWER_MAX_RETRIES} attempts: {last}"
    ) from last


def _abstention(reason: str, fabricated: list[int], supplied: int) -> Answer:
    """An abstention carries no text. The UI owns that wording.

    Deliberately not a sentence written here. The message differs by what the
    user selected - whether to suggest widening the scope - and a backend that
    returned prose would have the UI matching on it to decide.
    """
    return Answer(
        text="",
        citations=[],
        abstained=True,
        reason=reason,
        fabricated=fabricated,
        prompt_passages=supplied,
    )


def generate(
    question: str,
    hits: Sequence[Hit],
    names: dict[str, str],
    chat: ChatFunction | None = None,
) -> Answer:
    """Answer one question from these passages, or abstain.

    Raises:
        GenerationFailedError: the model could not be reached. Never returned as
            an abstention, which would misreport an outage as an honest refusal.
    """
    chat = chat or openai_chat()
    reply = _collect(prompt.assemble(question, hits, names), chat).strip()

    if not reply:
        return _abstention(REASON_EMPTY_ANSWER, [], len(hits))

    # Checked on the stripped reply rather than by equality. A model that adds a
    # trailing full stop to the marker still means it, and reading that as an
    # answer would show the marker itself to the user as though it were one.
    if reply.startswith(settings.ABSTENTION_MARKER):
        return _abstention(REASON_NOT_IN_DOCUMENTS, [], len(hits))

    validated = citations.validate(reply, hits, names)

    # An answer whose every citation was invented cannot be checked against
    # anything, which is exactly the state this system exists to avoid. It is
    # reported as an abstention rather than shown with no sources.
    if not validated.grounded:
        return _abstention(REASON_NO_VALID_CITATIONS, validated.fabricated, len(hits))

    return Answer(
        text=reply,
        citations=validated.citations,
        abstained=False,
        fabricated=validated.fabricated,
        prompt_passages=len(hits),
    )


def stream(
    question: str,
    hits: Sequence[Hit],
    names: dict[str, str],
    chat: ChatFunction | None = None,
) -> Iterator[str | Answer]:
    """The same answer, yielding text as it arrives and the `Answer` last.

    Streaming exists so text appears immediately, but citations cannot be
    resolved until the reply is complete, and an abstention must not be streamed
    at all - a user would watch `NOT_IN_DOCUMENTS` type itself out.

    So tokens are held back until enough has arrived to rule out an abstention,
    then released. The marker is the first thing in the reply when it is used, so
    the delay is the length of one word rather than the whole answer.
    """
    chat = chat or openai_chat()
    messages = prompt.assemble(question, hits, names)

    pieces: list[str] = []
    releasing = False
    try:
        for piece in chat(messages):
            pieces.append(piece)
            if releasing:
                yield piece
                continue
            sofar = "".join(pieces).lstrip()
            # Still short enough to be the start of the marker: keep holding.
            if settings.ABSTENTION_MARKER.startswith(sofar[: len(settings.ABSTENTION_MARKER)]):
                if sofar.startswith(settings.ABSTENTION_MARKER):
                    break
                continue
            releasing = True
            yield sofar
    except Exception as error:  # noqa: BLE001 - re-raised as a typed error
        if isinstance(error, GenerationFailedError):
            raise
        raise GenerationFailedError(f"generation failed mid-stream: {error}") from error

    reply = "".join(pieces).strip()
    if not reply:
        yield _abstention(REASON_EMPTY_ANSWER, [], len(hits))
        return
    if reply.startswith(settings.ABSTENTION_MARKER):
        yield _abstention(REASON_NOT_IN_DOCUMENTS, [], len(hits))
        return

    validated = citations.validate(reply, hits, names)
    if not validated.grounded:
        yield _abstention(REASON_NO_VALID_CITATIONS, validated.fabricated, len(hits))
        return

    yield Answer(
        text=reply,
        citations=validated.citations,
        abstained=False,
        fabricated=validated.fabricated,
        prompt_passages=len(hits),
    )
