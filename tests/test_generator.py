"""Tests for grounded generation.

The model call is injected and every test supplies the exact reply it depends on,
which is the only way to test the outcomes without asserting on wording.

The three outcomes are the point - an answer, an abstention, and a failure.
Reporting an outage as "your documents do not cover this" would be a lie.
"""

import pytest

from backend.errors import GenerationFailedError
from backend.retrieval import generator
from backend.storage.vector_store import Hit
from config import settings


def hit(number: int = 0, doc_id: str = "doc1") -> Hit:
    return Hit(
        chunk_id=f"{doc_id}:{number}",
        doc_id=doc_id,
        page=17,
        section_path="4.2 Leave",
        element_type="text",
        text="Employees accrue leave monthly after twelve months of service.",
        bboxes=[(1.0, 2.0, 3.0, 4.0)],
        similarity=0.7,
        raw_distance=0.7,
    )


NAMES = {"doc1": "Employee Handbook", "doc2": "Onboarding Guide"}


def replying(text: str):
    """A stand-in model that streams a fixed reply in small pieces.

    Chunked rather than returned whole, so the streaming path is exercised the
    way a real provider drives it.
    """

    def chat(_messages):
        for start in range(0, len(text), 7):
            yield text[start : start + 7]

    return chat


def failing(error: Exception):
    def chat(_messages):
        raise error
        yield ""  # pragma: no cover - never reached, makes this a generator

    return chat


# --- an answer -----------------------------------------------------------


def test_a_cited_answer_comes_back_as_an_answer():
    result = generator.generate(
        "How much leave accrues?", [hit()], NAMES, chat=replying("Leave accrues monthly [1].")
    )

    assert not result.abstained
    assert result.text == "Leave accrues monthly [1]."
    assert [c.number for c in result.citations] == [1]


def test_the_answer_reports_how_many_passages_it_was_given():
    """Every answer reports diagnostics, and this is one of them."""
    result = generator.generate("A question", [hit(0), hit(1)], NAMES, chat=replying("Yes [1]."))

    assert result.prompt_passages == 2


def test_citations_are_resolved_to_real_sources():
    result = generator.generate("A question", [hit()], NAMES, chat=replying("Yes [1]."))

    assert result.citations[0].document_name == "Employee Handbook"
    assert result.citations[0].page == 17


# --- an abstention -------------------------------------------------------


def test_the_marker_becomes_an_abstention():
    result = generator.generate(
        "Who won the tender?", [hit()], NAMES, chat=replying(settings.ABSTENTION_MARKER)
    )

    assert result.abstained
    assert result.reason == generator.REASON_NOT_IN_DOCUMENTS


def test_the_marker_with_trailing_punctuation_is_still_an_abstention():
    """A model that adds a full stop still means it. Read as an answer, the marker
    itself would be shown to the user as though it were one."""
    result = generator.generate(
        "A question", [hit()], NAMES, chat=replying(f"{settings.ABSTENTION_MARKER}.")
    )

    assert result.abstained


def test_an_abstention_carries_no_text():
    """The wording belongs to the UI, which alone knows whether suggesting a wider
    scope would be honest."""
    result = generator.generate(
        "A question", [hit()], NAMES, chat=replying(settings.ABSTENTION_MARKER)
    )

    assert result.text == ""
    assert result.citations == []


def test_an_answer_citing_only_invented_sources_becomes_an_abstention():
    """An answer whose every citation was invented becomes a refusal, because it
    exists to avoid. Shown as "I don't know", not as an answer with no sources."""
    result = generator.generate("A question", [hit()], NAMES, chat=replying("Certainly true [4]."))

    assert result.abstained
    assert result.reason == generator.REASON_NO_VALID_CITATIONS
    assert result.fabricated == [4]


def test_an_answer_with_no_citations_at_all_becomes_an_abstention():
    result = generator.generate(
        "A question", [hit()], NAMES, chat=replying("Leave accrues monthly.")
    )

    assert result.abstained
    assert result.reason == generator.REASON_NO_VALID_CITATIONS


def test_an_empty_reply_becomes_an_abstention_with_its_own_reason():
    result = generator.generate("A question", [hit()], NAMES, chat=replying("   "))

    assert result.abstained
    assert result.reason == generator.REASON_EMPTY_ANSWER


def test_a_partly_invented_answer_is_kept_and_the_invention_reported():
    """One real citation is enough to check the answer. The invented number still
    has to surface, because a rate that climbs means the contract has stopped
    holding."""
    result = generator.generate("A question", [hit()], NAMES, chat=replying("Both [1] and [9]."))

    assert not result.abstained
    assert [c.number for c in result.citations] == [1]
    assert result.fabricated == [9]


# --- a failure -----------------------------------------------------------


def test_an_unreachable_model_raises_rather_than_abstaining():
    """Reporting an outage as "your documents do not cover this" would be a lie."""
    with pytest.raises(GenerationFailedError):
        generator.generate("A question", [hit()], NAMES, chat=failing(ValueError("boom")))


def test_a_permanent_failure_is_not_retried_repeatedly(monkeypatch):
    """A bad key fails identically every time, so retrying only delays it."""
    calls = []

    def chat(_messages):
        calls.append(1)
        raise ValueError("bad request")
        yield ""  # pragma: no cover

    monkeypatch.setattr(settings, "ANSWER_RETRY_BACKOFF_SECONDS", 0)
    with pytest.raises(GenerationFailedError):
        generator.generate("A question", [hit()], NAMES, chat=chat)

    assert len(calls) == 1


# --- streaming -----------------------------------------------------------


def _drain(pieces):
    """Split a stream into its text and its final Answer."""
    text = []
    final = None
    for piece in pieces:
        if isinstance(piece, generator.Answer):
            final = piece
        else:
            text.append(piece)
    return "".join(text), final


def test_streaming_yields_the_text_then_the_answer():
    stream = generator.stream(
        "A question", [hit()], NAMES, chat=replying("Leave accrues monthly [1].")
    )
    text, final = _drain(stream)

    assert text == "Leave accrues monthly [1]."
    assert final is not None
    assert not final.abstained


def test_an_abstention_is_never_streamed_as_text():
    """A user would watch the marker type itself out. Tokens are held until enough
    has arrived to rule the marker out."""
    stream = generator.stream(
        "A question", [hit()], NAMES, chat=replying(settings.ABSTENTION_MARKER)
    )
    text, final = _drain(stream)

    assert text == ""
    assert final.abstained
    assert final.reason == generator.REASON_NOT_IN_DOCUMENTS


def test_an_answer_beginning_with_a_capital_n_still_streams():
    """The hold-back must not swallow a real answer that happens to start with
    letters the marker also starts with."""
    stream = generator.stream(
        "A question", [hit()], NAMES, chat=replying("Nothing accrues before twelve months [1].")
    )
    text, final = _drain(stream)

    assert text == "Nothing accrues before twelve months [1]."
    assert not final.abstained


def test_an_ungrounded_stream_ends_as_an_abstention():
    """The text has already been shown by then. The caller replaces it, which is
    why the final Answer carries the decision rather than the stream."""
    stream = generator.stream("A question", [hit()], NAMES, chat=replying("It is so [8]."))
    _text, final = _drain(stream)

    assert final.abstained
    assert final.reason == generator.REASON_NO_VALID_CITATIONS


def test_a_stream_that_dies_midway_raises_a_typed_error():
    def chat(_messages):
        yield "Leave accrues"
        raise ValueError("connection reset")

    with pytest.raises(GenerationFailedError):
        _drain(generator.stream("A question", [hit()], NAMES, chat=chat))


@pytest.mark.parametrize(
    "reply",
    [
        "Nothing accrues before twelve months [1].",
        "N [1].",
        "No leave accrues [1].",
        "NOTE: leave accrues monthly [1].",
    ],
)
def test_no_text_is_lost_when_tokens_arrive_one_character_at_a_time(reply):
    """The hold-back compares a growing prefix against the marker, so a reply that
    shares its opening letters is the case that would silently truncate. Real
    providers stream a character or two at a time, which is exactly this."""

    def one_character(_messages):
        yield from reply

    text, final = _drain(generator.stream("A question", [hit()], NAMES, chat=one_character))

    assert text == reply
    assert not final.abstained


# --- a half-answerable question ------------------------------------------
# A question with two parts can be answerable in one part and not the other.
# The model then answers what it can and marks the rest, putting the marker in
# the middle of a reply rather than at the start. Measured on the real corpus,
# not imagined - and the first version of this code showed the raw marker to the
# user at the end of an otherwise good answer.


HALF = "Fees are 30 points [1].\n\nNOT_IN_DOCUMENTS"


def test_a_marker_after_an_answer_never_reaches_the_user():
    result = generator.generate("two questions", [hit()], NAMES, chat=replying(HALF))

    assert settings.ABSTENTION_MARKER not in result.text
    assert result.text.startswith("Fees are 30 points [1].")


def test_a_half_answerable_question_still_counts_as_an_answer():
    result = generator.generate("two questions", [hit()], NAMES, chat=replying(HALF))

    assert not result.abstained
    assert [c.number for c in result.citations] == [1]


def test_a_half_answerable_question_is_recorded_as_partly_absent():
    """Not a failure, but a rate that climbs means questions are arriving with
    more parts than the retrieved passages cover."""
    result = generator.generate("two questions", [hit()], NAMES, chat=replying(HALF))

    assert result.partly_absent


def test_an_ordinary_answer_is_not_marked_partly_absent():
    result = generator.generate("a question", [hit()], NAMES, chat=replying("Yes [1]."))

    assert not result.partly_absent


def test_a_reply_that_is_only_a_marker_with_whitespace_is_an_abstention():
    result = generator.generate(
        "a question", [hit()], NAMES, chat=replying(f"\n\n{settings.ABSTENTION_MARKER}\n")
    )

    assert result.abstained
    assert result.reason == generator.REASON_NOT_IN_DOCUMENTS


def test_the_marker_is_never_streamed_even_in_the_middle_of_an_answer():
    text, final = _drain(generator.stream("two questions", [hit()], NAMES, chat=replying(HALF)))

    assert settings.ABSTENTION_MARKER not in text
    assert "Fees are 30 points [1]." in text
    assert not final.abstained
    assert final.partly_absent


@pytest.mark.parametrize(
    "reply",
    [
        HALF,
        "Fees are 30 points [1]. NOT_IN_DOCUMENTS for the rest.",
        "NOT_IN_DOCUMENTS",
        "Nothing was found [1]. NOT",
    ],
)
def test_streaming_one_character_at_a_time_never_leaks_the_marker(reply):
    """Real providers stream a character or two at a time, which is exactly the
    case a naive prefix check gets wrong."""

    def one_character(_messages):
        yield from reply

    text, _final = _drain(generator.stream("q", [hit()], NAMES, chat=one_character))

    assert settings.ABSTENTION_MARKER not in text
