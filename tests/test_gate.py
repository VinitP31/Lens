"""Tests for the confidence gate.

Pure logic over a retrieval result, so these need no store and no network. Hits
are constructed directly with the exact scores each test depends on.

The threshold is passed in rather than read from settings wherever a test depends
on a specific number, so recalibrating the real value cannot silently turn these
tests into assertions about nothing.
"""

from backend.retrieval import gate
from backend.retrieval.retriever import Retrieved
from backend.storage.vector_store import Hit
from config import settings


def hit(similarity: float) -> Hit:
    return Hit(
        chunk_id="doc1:0",
        doc_id="doc1",
        page=1,
        section_path="1. Leave",
        element_type="text",
        text="Annual leave accrues monthly.",
        bboxes=[(1.0, 2.0, 3.0, 4.0)],
        similarity=similarity,
        raw_distance=similarity,
    )


def retrieved(*similarities: float, scope: list[str] | None = None) -> Retrieved:
    return Retrieved(
        hits=[hit(value) for value in similarities],
        candidates_fetched=len(similarities),
        duplicates_removed=0,
        scope=scope,
    )


# --- The comparison ------------------------------------------------------


def test_a_score_above_the_threshold_passes():
    decision = gate.evaluate(retrieved(0.60), threshold=0.45)

    assert decision.passed
    assert decision.reason is None


def test_a_score_below_the_threshold_is_refused():
    decision = gate.evaluate(retrieved(0.30), threshold=0.45)

    assert not decision.passed
    assert decision.reason == gate.REASON_BELOW_THRESHOLD


def test_a_score_exactly_at_the_threshold_passes():
    """The comparison is `<`, so the threshold is the lowest acceptable score
    rather than the highest rejected one. Stated because an off-by-one here
    changes behaviour at the only value anyone will ever test by hand."""
    decision = gate.evaluate(retrieved(0.45), threshold=0.45)

    assert decision.passed


def test_only_the_best_chunk_decides():
    """The rest are context. A single strong match is enough to answer from, and
    requiring several would refuse anything stated once."""
    decision = gate.evaluate(retrieved(0.70, 0.20, 0.10), threshold=0.45)

    assert decision.passed
    assert decision.top_similarity == 0.70


# --- Nothing to answer from ----------------------------------------------


def test_no_matches_at_all_is_refused_with_its_own_reason():
    """Distinct from a poor match: vector search returning nothing means an empty
    index, and the honest message is different."""
    decision = gate.evaluate(retrieved(), threshold=0.45)

    assert not decision.passed
    assert decision.reason == gate.REASON_NO_MATCHES
    assert decision.top_similarity is None


def test_an_empty_scope_is_refused_as_having_no_documents():
    """The user selected nothing, or selected only deleted documents. Telling them
    the documents do not cover it would be wrong - none were searched."""
    decision = gate.evaluate(retrieved(scope=[]), threshold=0.45)

    assert not decision.passed
    assert decision.reason == gate.REASON_NO_DOCUMENTS


def test_a_whole_library_search_is_not_mistaken_for_an_empty_scope():
    """Whole-library scope is expressed as None, which is the opposite of empty."""
    decision = gate.evaluate(retrieved(0.60, scope=None), threshold=0.45)

    assert decision.passed


# --- Diagnostics ---------------------------------------------------------


def test_the_decision_carries_the_numbers_behind_it():
    """Every answer reports its margin, and a surprising refusal has to be
    explainable without re-running the query."""
    decision = gate.evaluate(retrieved(0.52), threshold=0.45)

    assert decision.top_similarity == 0.52
    assert decision.threshold == 0.45
    assert decision.margin == 0.07


def test_the_margin_is_negative_on_a_refusal():
    decision = gate.evaluate(retrieved(0.40), threshold=0.45)

    assert decision.margin < 0


def test_there_is_no_margin_when_nothing_was_retrieved():
    assert gate.evaluate(retrieved(), threshold=0.45).margin is None


# --- The configured value ------------------------------------------------


def test_the_configured_threshold_is_used_by_default():
    decision = gate.evaluate(retrieved(1.0))

    assert decision.threshold == settings.GATE_THRESHOLD


def test_the_configured_threshold_admits_the_lowest_measured_real_answer():
    """The calibration decided the gate must lose no real answer. The lowest
    scoring answerable question in the golden set was +0.496; if a future change
    pushes the threshold above that, this fails rather than silently starting to
    refuse questions the corpus does answer."""
    lowest_real_answer = 0.496

    assert gate.evaluate(retrieved(lowest_real_answer)).passed


def test_the_configured_threshold_still_refuses_something():
    """A threshold low enough to admit everything would be a gate in name only.
    The lowest scoring out-of-scope question measured +0.392."""
    clearly_irrelevant = 0.392

    assert not gate.evaluate(retrieved(clearly_irrelevant)).passed
