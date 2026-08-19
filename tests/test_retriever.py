"""Tests for retrieval.

A real registry and vector store with hand-written vectors, so the suite stays
offline and every expected ranking is obvious by eye.

Under test: scoping, over-fetch, deduplication. Whether the chunks are good enough
is the gate's job, and whether the model uses them well is generation's.
"""

import pytest

from backend.ingestion.chunk import Chunk
from backend.retrieval import retriever
from backend.storage import registry, vector_store
from config import settings


@pytest.fixture
def stores(tmp_path):
    db = registry.connect(tmp_path / "lens.db")
    store = vector_store.connect(tmp_path / "chunks.db")
    yield db, store
    db.close()
    store.close()


def vector(*leading: float) -> list[float]:
    values = list(leading)
    return values + [0.0] * (settings.EMBEDDING_DIMENSIONS - len(values))


def embedder_returning(*leading: float):
    """A stand-in embedder that always produces the same query vector."""

    def embed(texts):
        return [vector(*leading) for _ in texts]

    return embed


def chunk(index: int, text: str, page: int = 1) -> Chunk:
    return Chunk(
        index=index,
        text=text,
        page=page,
        section_path="1. Leave",
        element_type="text",
        token_count=len(text.split()),
        bboxes=[(1.0, 2.0, 3.0, 4.0)],
        context_header=f"[Doc > p.{page}]",
    )


def add_document(db, store, name, chunks, vectors, *, ready=True):
    document = registry.register(
        db,
        original_filename=name,
        content_hash=f"hash-{name}",
        size_bytes=1024,
        file_path=name,
    )
    vector_store.upsert(store, document.doc_id, chunks, vectors)
    if ready:
        registry.mark_ready(db, document.doc_id, page_count=1, chunk_count=len(chunks))
    return document


# --- The basics ----------------------------------------------------------


def test_the_closest_chunk_comes_first(stores):
    db, store = stores
    add_document(
        db,
        store,
        "handbook.pdf",
        [chunk(0, "far away"), chunk(1, "exact match")],
        [vector(0.0, 1.0), vector(1.0, 0.0)],
    )

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0, 0.0))

    assert result.hits[0].text == "exact match"
    assert result.top_similarity == pytest.approx(1.0, abs=0.01)


def test_only_the_context_budget_is_passed_on(stores):
    """Twelve are fetched so the five that survive deduplication are five real
    passages rather than repeats of one."""
    db, store = stores
    # Genuinely different passages, so none of them collapse into each other.
    subjects = [
        "password rules require twelve characters and one symbol",
        "the session idle timeout ends work after twenty minutes",
        "export is limited to fifty thousand rows per download",
        "lists refresh automatically once every sixty seconds",
        "saved filters stay private unless marked as shared",
        "notifications warn when a pallet is unconfirmed for hours",
        "gate entry records the driver licence and mobile number",
        "the seal number is compared against the lorry receipt",
        "quality disposition defaults to quarantine when omitted",
        "pallet labels print automatically once a receipt posts",
        "bin addresses encode zone, aisle, level and position",
        "the audit trail cannot be amended by any role at all",
    ]
    chunks = [chunk(index, text) for index, text in enumerate(subjects)]
    add_document(
        db, store, "handbook.pdf", chunks, [vector(1.0, index * 0.1) for index in range(12)]
    )

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0, 0.0))

    assert len(result.hits) == settings.CONTEXT_CHUNKS
    assert result.candidates_fetched == settings.RETRIEVE_CANDIDATES


def test_an_empty_library_returns_nothing_rather_than_failing(stores):
    """Vector search has no notion of "no match", so the honest result of an
    empty library is no hits, which the gate turns into an abstention."""
    db, store = stores

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0))

    assert result.hits == []
    assert result.top_similarity is None


# --- Scope ---------------------------------------------------------------


def test_search_can_be_restricted_to_chosen_documents(stores):
    db, store = stores
    first = add_document(db, store, "one.pdf", [chunk(0, "from one")], [vector(1.0)])
    add_document(db, store, "two.pdf", [chunk(0, "from two")], [vector(1.0)])

    result = retriever.retrieve(
        db, store, "question", doc_ids=[first.doc_id], embed=embedder_returning(1.0)
    )

    assert [hit.text for hit in result.hits] == ["from one"]


def test_no_scope_means_the_whole_library(stores):
    db, store = stores
    add_document(db, store, "one.pdf", [chunk(0, "from one")], [vector(1.0)])
    add_document(db, store, "two.pdf", [chunk(0, "from two")], [vector(1.0)])

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0))

    assert len(result.hits) == 2


def test_selecting_nothing_searches_nothing(stores):
    """An empty selection is the user having chosen no documents. Widening that to
    the whole library would answer from sources they excluded."""
    db, store = stores
    add_document(db, store, "one.pdf", [chunk(0, "from one")], [vector(1.0)])

    result = retriever.retrieve(db, store, "question", doc_ids=[], embed=embedder_returning(1.0))

    assert result.hits == []


def test_a_deleted_document_stops_answering_questions(stores):
    """The vector store has no idea a document was deleted - that lives in
    SQLite. Without asking, a removed document keeps being cited."""
    db, store = stores
    kept = add_document(db, store, "kept.pdf", [chunk(0, "still here")], [vector(1.0)])
    gone = add_document(db, store, "gone.pdf", [chunk(0, "removed")], [vector(1.0)])
    registry.soft_delete(db, gone.doc_id)

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0))

    assert [hit.text for hit in result.hits] == ["still here"]
    assert result.scope == [kept.doc_id]


def test_explicitly_selecting_a_deleted_document_returns_nothing(stores):
    db, store = stores
    gone = add_document(db, store, "gone.pdf", [chunk(0, "removed")], [vector(1.0)])
    registry.soft_delete(db, gone.doc_id)

    result = retriever.retrieve(
        db, store, "question", doc_ids=[gone.doc_id], embed=embedder_returning(1.0)
    )

    assert result.hits == []


def test_a_document_still_ingesting_is_not_searched(stores):
    """Its chunks are only partly written, so it would answer some questions and
    silently miss the rest of its own content."""
    db, store = stores
    add_document(db, store, "ready.pdf", [chunk(0, "finished")], [vector(1.0)])
    add_document(db, store, "busy.pdf", [chunk(0, "half indexed")], [vector(1.0)], ready=False)

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0))

    assert [hit.text for hit in result.hits] == ["finished"]


# --- Deduplication -------------------------------------------------------


def test_two_chunks_sharing_an_overlap_window_collapse_to_one(stores):
    """Overlap makes neighbours near-identical by design. Both retrieved is one
    answer twice, and the second has displaced a different passage."""
    db, store = stores
    # A realistic overlap window: the chunker copies whole sentences, roughly 55
    # words, from the end of one chunk into the start of the next.
    shared = (
        "Requests for planned absence require approval fourteen days in advance. "
        "Approval is recorded by the reporting manager in the leave register, and "
        "an absence taken without a recorded approval is treated as unplanned."
    )
    add_document(
        db,
        store,
        "handbook.pdf",
        [
            chunk(0, f"{shared} An absence without approval is unplanned."),
            chunk(1, f"{shared} Urgent medical absence is exempt."),
        ],
        [vector(1.0, 0.0), vector(1.0, 0.05)],
    )

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0, 0.0))

    assert len(result.hits) == 1
    assert result.duplicates_removed == 1


def test_the_higher_scoring_copy_is_the_one_kept(stores):
    db, store = stores
    # A realistic overlap window: the chunker copies whole sentences, roughly 55
    # words, from the end of one chunk into the start of the next.
    shared = (
        "Requests for planned absence require approval fourteen days in advance. "
        "Approval is recorded by the reporting manager in the leave register, and "
        "an absence taken without a recorded approval is treated as unplanned."
    )
    add_document(
        db,
        store,
        "handbook.pdf",
        [
            chunk(0, f"{shared} Keep this one, it is closer."),
            chunk(1, f"{shared} Drop this one."),
        ],
        [vector(1.0, 0.0), vector(1.0, 0.6)],
    )

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0, 0.0))

    assert "Keep this one" in result.hits[0].text


def test_two_documents_stating_the_same_rule_are_both_kept(stores):
    """Genuinely two sources. Both deserve a citation, and collapsing them would
    hide that the rule is corroborated."""
    db, store = stores
    same = (
        "Blind confirmation, in which the bin is keyed rather than scanned, is not "
        "permitted under any circumstance, because keyed confirmations accounted "
        "for the majority of stock located in a bin other than the one recorded."
    )
    add_document(db, store, "manual.pdf", [chunk(0, same)], [vector(1.0)])
    add_document(db, store, "procedure.pdf", [chunk(0, same)], [vector(1.0)])

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0))

    assert len(result.hits) == 2
    assert result.duplicates_removed == 0


def test_two_pages_of_one_document_are_both_kept(stores):
    """A passage continued on the next page is a different citation, and each
    must be able to open its own page."""
    db, store = stores
    same = (
        "Detention charges accrue to the transporter and are recorded against the "
        "trip in FleetLink so that they are captured in the transporter settlement "
        "run described in the vehicle placement procedure."
    )
    add_document(
        db,
        store,
        "manual.pdf",
        [chunk(0, same, page=9), chunk(1, same, page=10)],
        [vector(1.0), vector(1.0)],
    )

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0))

    assert {hit.page for hit in result.hits} == {9, 10}


def test_different_passages_on_one_page_are_not_collapsed(stores):
    """Deduplication must not eat distinct content that happens to share common
    words like "the" and "is"."""
    db, store = stores
    add_document(
        db,
        store,
        "handbook.pdf",
        [
            chunk(0, "The minimum password length is twelve characters."),
            chunk(1, "The session idle timeout is twenty minutes."),
        ],
        [vector(1.0, 0.0), vector(1.0, 0.05)],
    )

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0, 0.0))

    assert len(result.hits) == 2


def test_sharing_ordinary_vocabulary_is_not_duplication():
    """Why duplication is measured as a run of consecutive words rather than as a
    proportion of shared vocabulary. These two sentences have most of their words
    in common and state entirely different facts. A proportion-based rule
    collapsed them, losing one of the two answers."""
    first = "The minimum password length is twelve characters."
    second = "The session idle timeout is twenty minutes."

    assert retriever._longest_shared_run(first, second) < settings.DEDUPE_MIN_SHARED_WORDS


def test_a_copied_passage_is_duplication():
    tail = (
        "Approval is recorded by the reporting manager in the leave register, and an "
        "absence taken without a recorded approval is treated as unplanned."
    )

    assert (
        retriever._longest_shared_run(f"Earlier text. {tail}", f"{tail} Later text.")
        >= settings.DEDUPE_MIN_SHARED_WORDS
    )


# --- Diagnostics ---------------------------------------------------------


def test_the_working_is_reported_for_every_search(stores):
    """Every answer reports diagnostics, and the gate threshold is calibrated
    from these scores, so they have to be observable rather than reconstructed."""
    db, store = stores
    add_document(db, store, "handbook.pdf", [chunk(0, "text")], [vector(1.0)])

    result = retriever.retrieve(db, store, "question", embed=embedder_returning(1.0))

    assert result.candidates_fetched == 1
    assert result.duplicates_removed == 0
    assert result.top_similarity is not None
    assert result.scope is None
