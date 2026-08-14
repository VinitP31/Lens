"""Tests for citation validation.

This is the module that makes the product's central claim true, so the tests are
about what code refuses to resolve rather than about what the model wrote.

Nothing here asserts on answer wording. Every case constructs a reply as a string
and checks what validation does with it.
"""

from backend.retrieval import citations
from backend.storage.vector_store import Hit
from config import settings


def hit(number: int, doc_id: str = "doc1", text: str = "Annual leave accrues monthly.") -> Hit:
    return Hit(
        chunk_id=f"{doc_id}:{number}",
        doc_id=doc_id,
        page=number + 10,
        section_path="4.2 Leave",
        element_type="text",
        text=text,
        bboxes=[(1.0, 2.0, 3.0, 4.0)],
        similarity=0.7,
        raw_distance=0.7,
    )


NAMES = {"doc1": "Employee Handbook", "doc2": "Onboarding Guide"}


# --- parsing -------------------------------------------------------------


def test_citation_numbers_are_found_in_the_order_they_appear():
    assert citations.parse("First [2] then [1].") == [2, 1]


def test_a_passage_cited_twice_is_one_source():
    """Listing it twice beneath the answer suggests two pieces of evidence."""
    assert citations.parse("Both [1] and again [1].") == [1]


def test_prose_that_is_not_a_bracketed_number_is_not_a_citation():
    """The contract in the prompt is explicit. Resolving a near-miss would let a
    malformed reference point at a real source."""
    assert citations.parse("See passage 2 and [not a number] and (3).") == []


def test_an_answer_with_no_citations_parses_to_nothing():
    assert citations.parse("Leave accrues monthly.") == []


# --- what code refuses to resolve ----------------------------------------


def test_a_number_that_was_never_supplied_is_thrown_away():
    """The model was given two passages. There is no third to resolve."""
    validated = citations.validate("Stated in [1] and [3].", [hit(0), hit(1)], NAMES)

    assert [c.number for c in validated.citations] == [1]
    assert validated.fabricated == [3]


def test_zero_is_not_a_valid_citation():
    """Passages are numbered from 1, so a zero is invented whatever it looks like."""
    validated = citations.validate("As shown in [0].", [hit(0)], NAMES)

    assert validated.citations == []
    assert validated.fabricated == [0]


def test_an_answer_citing_only_invented_sources_is_not_grounded():
    """Nothing in it can be checked, which is the state the system exists to
    avoid. The caller shows an abstention rather than an answer with no sources."""
    validated = citations.validate("Clearly stated in [7].", [hit(0)], NAMES)

    assert not validated.grounded


def test_an_answer_with_one_real_citation_is_grounded():
    validated = citations.validate("Stated in [1].", [hit(0)], NAMES)

    assert validated.grounded


# --- resolution ----------------------------------------------------------


def test_the_source_is_resolved_from_the_passage_not_the_reply():
    """The model wrote only the number 1. Everything else is looked up."""
    validated = citations.validate("Stated in [1].", [hit(0, doc_id="doc2")], NAMES)
    citation = validated.citations[0]

    assert citation.document_name == "Onboarding Guide"
    assert citation.page == 10
    assert citation.section_path == "4.2 Leave"
    assert citation.chunk_id == "doc2:0"


def test_the_number_maps_to_the_passage_at_that_position():
    """Passages are numbered from 1 and lists index from 0. An off-by-one here
    attributes every answer to the wrong source while looking entirely correct."""
    passages = [hit(0, doc_id="doc1"), hit(1, doc_id="doc2")]
    validated = citations.validate("Stated in [2].", passages, NAMES)

    assert validated.citations[0].doc_id == "doc2"
    assert validated.citations[0].chunk_id == "doc2:1"


def test_coordinates_travel_with_the_citation():
    """Without them the citation cannot be highlighted on the rendered page."""
    validated = citations.validate("Stated in [1].", [hit(0)], NAMES)

    assert validated.citations[0].bboxes == [(1.0, 2.0, 3.0, 4.0)]


def test_an_unknown_document_falls_back_to_its_id():
    """An otherwise good answer must not be discarded because one row could not
    be read."""
    validated = citations.validate("Stated in [1].", [hit(0, doc_id="missing")], NAMES)

    assert validated.citations[0].document_name == "missing"


# --- snippets ------------------------------------------------------------


def test_a_long_passage_is_cut_at_a_word_boundary():
    """A snippet ending mid-word reads as though the passage itself were broken."""
    long_text = "leave " * 200
    validated = citations.validate("Stated in [1].", [hit(0, text=long_text)], NAMES)
    snippet = validated.citations[0].snippet

    assert len(snippet) <= settings.CITATION_SNIPPET_CHARS + 1
    assert snippet.endswith("…")
    assert "leav…" not in snippet


def test_a_short_passage_is_kept_whole_with_no_ellipsis():
    validated = citations.validate("Stated in [1].", [hit(0, text="Leave accrues.")], NAMES)

    assert validated.citations[0].snippet == "Leave accrues."


def test_whitespace_in_a_snippet_is_collapsed():
    """Table markup and line breaks arrive in chunk text and would render as gaps."""
    validated = citations.validate("Stated in [1].", [hit(0, text="Leave\n\n  accrues.")], NAMES)

    assert validated.citations[0].snippet == "Leave accrues."


# --- plain text ----------------------------------------------------------


def test_markers_can_be_stripped_for_a_plain_copy():
    assert citations.strip_markers("Leave accrues [1] monthly [2].") == "Leave accrues monthly."
