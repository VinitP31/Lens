"""Producing one grounded answer.

The gate has already run, so this never re-decides whether to answer. What it
decides is whether what came back is usable, and there are three outcomes that
must not be conflated:

An answer, with at least one citation code could resolve. An abstention, because
the model said the passages do not hold the answer or because every citation it
offered was invented - both mean nothing here can be checked. And a failure,
because the provider could not be reached, which is not an abstention: telling
somebody their documents do not cover a question when a network call failed would
be a lie in the one place this system exists not to tell one.

The model call is injected, so the suite runs offline. Tests assert on structure
and on the three outcomes, never on wording.
"""

import os
import re
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


def strip_marker(reply: str) -> tuple[str, bool]:
    """Remove the abstention marker from a reply, wherever it appears.

    A two-part question can be half answerable, and the model then answers the half
    it can and appends the marker for the rest. Checking only the start of the reply
    let that through, and the literal marker was shown at the end of an otherwise
    good answer.

    Returns the cleaned text and whether the marker was there, so a partly
    answerable question shows up in diagnostics.
    """
    if settings.ABSTENTION_MARKER not in reply:
        return reply.strip(), False
    cleaned = reply.replace(settings.ABSTENTION_MARKER, " ")
    # Removing it mid-text leaves the gap it sat in, so blank lines and runs of
    # spaces are closed up rather than shown.
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
    return cleaned.strip(), True


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
    # The model answered part of the question and reported the rest as absent.
    # Not a failure - a two-part question can be half answerable - but worth
    # recording, because a rate that climbs means questions are routinely
    # arriving with more parts than the retrieved passages cover.
    partly_absent: bool = False


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

    # The marker is removed wherever it appears, not only from the start. What
    # remains decides the outcome: nothing left means the model refused, and
    # anything left is an answer it could give.
    answer_text, marker_seen = strip_marker(reply)
    if not answer_text:
        return _abstention(REASON_NOT_IN_DOCUMENTS, [], len(hits))

    validated = citations.validate(answer_text, hits, names)

    # An answer whose every citation was invented cannot be checked against
    # anything, which is exactly the state this system exists to avoid. It is
    # reported as an abstention rather than shown with no sources.
    if not validated.grounded:
        return _abstention(REASON_NO_VALID_CITATIONS, validated.fabricated, len(hits))

    return Answer(
        text=answer_text,
        citations=validated.citations,
        abstained=False,
        fabricated=validated.fabricated,
        prompt_passages=len(hits),
        partly_absent=marker_seen,
    )


def stream(
    question: str,
    hits: Sequence[Hit],
    names: dict[str, str],
    chat: ChatFunction | None = None,
) -> Iterator[str | Answer]:
    """The same answer, yielding text as it arrives and the `Answer` last.

    Text is released with a short hold-back - everything but the last few
    characters, which might be the beginning of the abstention marker. The delay is
    one word, not the whole answer, and the marker never reaches the screen.

    The whole stream is filtered, not just its opening: a half-answerable question
    gets the marker appended rather than sent alone.
    """
    chat = chat or openai_chat()
    messages = prompt.assemble(question, hits, names)
    marker = settings.ABSTENTION_MARKER

    collected: list[str] = []
    pending = ""
    emitted = False
    marker_seen = False

    def split(text: str) -> tuple[str, str, bool]:
        """Divide text into what is safe to show now and what must wait.

        A complete marker is removed. A few trailing characters are held back
        when they could still turn into one, so the marker is never shown and
        then regretted.
        """
        seen = marker in text
        if seen:
            text = text.replace(marker, " ")
        for keep in range(min(len(marker) - 1, len(text)), 0, -1):
            if marker.startswith(text[-keep:]):
                return text[: len(text) - keep], text[len(text) - keep :], seen
        return text, "", seen

    try:
        for piece in chat(messages):
            collected.append(piece)
            safe, pending, seen = split(pending + piece)
            marker_seen = marker_seen or seen
            # Whitespace left behind by a removed marker is not worth showing
            # before any real text has been shown.
            if safe and (safe.strip() or emitted):
                emitted = True
                yield safe
    except Exception as error:  # noqa: BLE001 - re-raised as a typed error
        if isinstance(error, GenerationFailedError):
            raise
        raise GenerationFailedError(f"generation failed mid-stream: {error}") from error

    # Nothing further is coming, so what is still held cannot become a marker.
    if pending and (pending.strip() or emitted):
        yield pending

    answer_text, seen_in_full = strip_marker("".join(collected))
    marker_seen = marker_seen or seen_in_full

    if not answer_text:
        yield _abstention(
            REASON_NOT_IN_DOCUMENTS if marker_seen else REASON_EMPTY_ANSWER, [], len(hits)
        )
        return

    validated = citations.validate(answer_text, hits, names)
    if not validated.grounded:
        yield _abstention(REASON_NO_VALID_CITATIONS, validated.fabricated, len(hits))
        return

    yield Answer(
        text=answer_text,
        citations=validated.citations,
        abstained=False,
        fabricated=validated.fabricated,
        prompt_passages=len(hits),
        partly_absent=marker_seen,
    )
