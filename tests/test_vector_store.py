"""Tests for the vector store.

Milvus Lite is a local file, so these run offline against a real store in a
temporary directory. Nothing is mocked: the two things most worth proving here
are that a second ingest does not duplicate rows and that data outlives the
process, and neither can be proved against a fake.

Vectors are hand-written, not embedded. The embedding model is not under test,
and hand-written vectors make the expected ranking obvious by eye.
"""

import pytest

from backend.errors import EmbedModelMismatchError
from backend.ingestion.chunk import Chunk
from backend.storage import vector_store
from config import settings


@pytest.fixture
def store(tmp_path):
    client = vector_store.connect(tmp_path / "chunks.db")
    yield client
    client.close()


def vector(*leading: float) -> list[float]:
    """A full-width vector whose first few components are given.

    The rest are zero, so cosine similarity depends only on what the test set.
    """
    values = list(leading)
    return values + [0.0] * (settings.EMBEDDING_DIMENSIONS - len(values))


def chunk(index: int, text="Annual leave accrues monthly.", page=1, element_type="text") -> Chunk:
    return Chunk(
        index=index,
        text=text,
        page=page,
        section_path="1. Leave > 1.2 Accrual",
        element_type=element_type,
        token_count=len(text.split()),
        bboxes=[(10.0, 20.0, 300.0, 40.0)],
        context_header=f"[Handbook > 1. Leave > p.{page}]",
    )


# --- Chunk ids -----------------------------------------------------------


def test_the_chunk_id_is_the_document_id_and_position():
    assert vector_store.chunk_id("a3f2", 41) == "a3f2:41"


def test_chunk_ids_are_stored_in_position_order(store):
    chunks = [chunk(index) for index in range(3)]
    vector_store.upsert(store, "doc1", chunks, [vector(float(i + 1)) for i in range(3)])

    assert vector_store.chunk_ids(store, "doc1") == ["doc1:0", "doc1:1", "doc1:2"]


# --- The stage gate: reingest must not duplicate -------------------------


def test_ingesting_the_same_document_twice_does_not_double_the_chunks(store):
    """Half the stage gate. The ids are derived from document and position, so a
    second ingest lands on the same primary keys and replaces them."""
    chunks = [chunk(index) for index in range(5)]
    vectors = [vector(float(index + 1)) for index in range(5)]

    vector_store.upsert(store, "doc1", chunks, vectors)
    assert vector_store.count(store, "doc1") == 5

    vector_store.upsert(store, "doc1", chunks, vectors)
    assert vector_store.count(store, "doc1") == 5


def test_reingesting_replaces_the_text_rather_than_keeping_both(store):
    """A corrected extraction must win, not sit beside the old one."""
    vector_store.upsert(store, "doc1", [chunk(0, text="four hours")], [vector(1.0)])
    vector_store.upsert(store, "doc1", [chunk(0, text="at least four hours")], [vector(1.0)])

    hits = vector_store.search(store, vector(1.0), doc_ids=["doc1"])

    assert len(hits) == 1
    assert hits[0].text == "at least four hours"


def test_a_shorter_reingest_leaves_no_orphan_chunks(store):
    """Re-extraction can produce fewer chunks. The leftovers of the longer run
    would otherwise stay searchable forever, citing text no longer extracted."""
    vector_store.upsert(store, "doc1", [chunk(i) for i in range(5)], [vector(1.0)] * 5)
    vector_store.delete_document(store, "doc1")
    vector_store.upsert(store, "doc1", [chunk(i) for i in range(2)], [vector(1.0)] * 2)

    assert vector_store.count(store, "doc1") == 2


# --- The stage gate: survives a restart ----------------------------------


def test_chunks_survive_reopening_the_store(tmp_path):
    """The other half of the stage gate. Written and read through two separate
    clients against the same file."""
    path = tmp_path / "chunks.db"

    first = vector_store.connect(path)
    vector_store.upsert(first, "doc1", [chunk(0, text="carried over")], [vector(1.0)])
    first.close()

    second = vector_store.connect(path)
    hits = vector_store.search(second, vector(1.0))
    total = vector_store.count(second)
    second.close()

    assert total == 1
    assert hits[0].text == "carried over"


def test_a_released_collection_is_loaded_again_on_connect(tmp_path):
    """A real restart, not just a second handle in the same process.

    Milvus hands back an existing collection in a released state, and a released
    collection refuses every search. The plain reopen test above passes while this
    is broken, because a collection created in this process is still loaded - so
    the fault only appears on the second run of the application, which is every
    run after the first.
    """
    path = tmp_path / "chunks.db"
    first = vector_store.connect(path)
    vector_store.upsert(first, "doc1", [chunk(0, text="survives a restart")], [vector(1.0)])
    # What a new process finds waiting for it.
    first.release_collection(settings.MILVUS_COLLECTION)
    first.close()

    second = vector_store.connect(path)
    hits = vector_store.search(second, vector(1.0))
    second.close()

    assert [hit.text for hit in hits] == ["survives a restart"]


def test_connecting_to_an_existing_store_does_not_wipe_it(tmp_path):
    path = tmp_path / "chunks.db"

    first = vector_store.connect(path)
    vector_store.upsert(first, "doc1", [chunk(0)], [vector(1.0)])
    first.close()

    second = vector_store.connect(path)
    total = vector_store.count(second)
    second.close()

    assert total == 1


# --- Scores are the right way up -----------------------------------------


def test_a_closer_chunk_scores_higher(store):
    """The single most invertible number in the system. With cosine, Milvus
    reports similarity in a field named `distance`: identical is +1.0, unrelated
    0.0, opposite -1.0. Measured against a live collection."""
    vector_store.upsert(
        store,
        "doc1",
        [chunk(0, text="identical"), chunk(1, text="unrelated"), chunk(2, text="opposite")],
        [vector(1.0, 0.0), vector(0.0, 1.0), vector(-1.0, 0.0)],
    )

    hits = vector_store.search(store, vector(1.0, 0.0), doc_ids=["doc1"])

    assert [hit.text for hit in hits] == ["identical", "unrelated", "opposite"]
    assert hits[0].similarity == pytest.approx(1.0, abs=0.01)
    assert hits[1].similarity == pytest.approx(0.0, abs=0.01)
    assert hits[2].similarity == pytest.approx(-1.0, abs=0.01)


def test_similarity_decreases_down_the_result_list(store):
    """What the gate depends on: threshold the top hit and the rest are worse."""
    vector_store.upsert(
        store,
        "doc1",
        [chunk(index) for index in range(3)],
        [vector(1.0, 0.0), vector(1.0, 0.6), vector(1.0, 3.0)],
    )

    similarities = [hit.similarity for hit in vector_store.search(store, vector(1.0, 0.0))]

    assert similarities == sorted(similarities, reverse=True)


def test_the_raw_milvus_score_is_kept_alongside_the_similarity(store):
    """Both are logged side by side from the first line of code, because an
    inverted gate looks like a prompt bug and costs days."""
    vector_store.upsert(store, "doc1", [chunk(0)], [vector(1.0)])

    hit = vector_store.search(store, vector(1.0))[0]

    assert hit.raw_distance == pytest.approx(hit.similarity, abs=0.001)


# --- Scoped search -------------------------------------------------------


def test_search_can_be_restricted_to_chosen_documents(store):
    vector_store.upsert(store, "doc1", [chunk(0, text="from one")], [vector(1.0)])
    vector_store.upsert(store, "doc2", [chunk(0, text="from two")], [vector(1.0)])

    hits = vector_store.search(store, vector(1.0), doc_ids=["doc2"])

    assert [hit.text for hit in hits] == ["from two"]


def test_search_without_a_scope_covers_the_whole_library(store):
    vector_store.upsert(store, "doc1", [chunk(0)], [vector(1.0)])
    vector_store.upsert(store, "doc2", [chunk(0)], [vector(1.0)])

    assert len(vector_store.search(store, vector(1.0))) == 2


def test_search_honours_the_candidate_limit(store):
    vector_store.upsert(store, "doc1", [chunk(index) for index in range(20)], [vector(1.0)] * 20)

    assert len(vector_store.search(store, vector(1.0), limit=3)) == 3


def test_searching_an_empty_store_returns_nothing_rather_than_failing(store):
    """Vector search has no idea of "no match", so an empty store must simply
    yield no hits and let the gate abstain."""
    assert vector_store.search(store, vector(1.0)) == []


# --- Deletion ------------------------------------------------------------


def test_deleting_a_document_removes_only_its_chunks(store):
    vector_store.upsert(store, "doc1", [chunk(0), chunk(1)], [vector(1.0)] * 2)
    vector_store.upsert(store, "doc2", [chunk(0)], [vector(1.0)])

    vector_store.delete_document(store, "doc1")

    assert vector_store.count(store, "doc1") == 0
    assert vector_store.count(store, "doc2") == 1


def test_deleting_a_document_with_no_chunks_is_not_an_error(store):
    """Rollback of a failed ingest may run before anything was written."""
    vector_store.delete_document(store, "never-indexed")

    assert vector_store.count(store) == 0


# --- Metadata carried through --------------------------------------------


def test_page_section_and_type_come_back_for_the_citation(store):
    vector_store.upsert(store, "doc1", [chunk(0, page=17, element_type="table")], [vector(1.0)])

    hit = vector_store.search(store, vector(1.0))[0]

    assert hit.page == 17
    assert hit.element_type == "table"
    assert hit.section_path == "1. Leave > 1.2 Accrual"
    assert hit.doc_id == "doc1"


def test_bounding_boxes_survive_the_round_trip(store):
    """Without these a citation cannot highlight anything on the page."""
    stored = chunk(0)
    vector_store.upsert(store, "doc1", [stored], [vector(1.0)])

    hit = vector_store.search(store, vector(1.0))[0]

    assert hit.bboxes == [(10.0, 20.0, 300.0, 40.0)]


def test_a_chunk_with_no_boxes_round_trips_as_an_empty_list(store):
    bare = Chunk(
        index=0,
        text="text with no boxes",
        page=1,
        section_path="",
        element_type="text",
        token_count=4,
        bboxes=[],
    )
    vector_store.upsert(store, "doc1", [bare], [vector(1.0)])

    assert vector_store.search(store, vector(1.0))[0].bboxes == []


# --- Writing -------------------------------------------------------------


def test_upserting_nothing_writes_nothing(store):
    assert vector_store.upsert(store, "doc1", [], []) == 0


def test_a_vector_count_mismatch_is_refused(store):
    """One vector per chunk. A silent misalignment would attach every chunk's
    text to another chunk's meaning."""
    with pytest.raises(ValueError):
        vector_store.upsert(store, "doc1", [chunk(0), chunk(1)], [vector(1.0)])


# --- Embedding model guard -----------------------------------------------


def test_the_collection_is_created_at_the_configured_width(store):
    assert vector_store.dimension(store) == settings.EMBEDDING_DIMENSIONS


def test_a_matching_configuration_is_accepted(store):
    vector_store.assert_usable(store)


def test_a_dimension_mismatch_refuses_to_start(store, monkeypatch):
    """Vectors from two models occupy different spaces. Mixing them wrecks
    retrieval while every part of the system still reports success."""
    monkeypatch.setattr(settings, "EMBEDDING_DIMENSIONS", 768)

    with pytest.raises(EmbedModelMismatchError) as raised:
        vector_store.assert_usable(store)

    assert raised.value.code == "embed_model_mismatch"
    assert "768" in raised.value.detail
