"""End-to-end tests across the whole ingest path.

Every other test file covers one module. These cover the seams between them,
which is where the failures that matter actually live: a change to `Element`'s
shape, or to what `Chunk` carries, can break ingestion while every unit test
stays green.

The path is validate (by hash) -> extract -> chunk -> store -> mark ready. The
embedding step is stood in for by a deterministic fake, because the embedding
model is not under test here and the suite must run offline.
"""

import hashlib
import random

import pytest

from backend.errors import DuplicateDocumentError
from backend.ingestion import chunker, extractor
from backend.storage import registry, vector_store
from config import settings
from tests.conftest import PAGE_MARKERS


def fake_vectors(chunks) -> list[list[float]]:
    """Stands in for the embedder, which is the next piece to be built.

    Seeded from the chunk's own text, so the same text always produces the same
    vector. That makes reingest genuinely comparable rather than randomly
    different every run.
    """
    vectors = []
    for chunk in chunks:
        rng = random.Random(chunk.text)
        vectors.append([rng.uniform(-1.0, 1.0) for _ in range(settings.EMBEDDING_DIMENSIONS)])
    return vectors


@pytest.fixture
def stores(tmp_path):
    """A registry and a vector store, both isolated to this test."""
    db = registry.connect(tmp_path / "lens.db")
    store = vector_store.connect(tmp_path / "chunks.db")
    yield db, store
    db.close()
    store.close()


def ingest(db, store, pdf):
    """The ingest path, in the order backend rules require.

    Deliberately written out rather than hidden behind a helper in the codebase:
    the pipeline module does not exist yet, and this is the shape it has to have.
    """
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()

    existing = registry.find_by_hash(db, digest)
    document = existing or registry.register(
        db,
        original_filename=pdf.name,
        content_hash=digest,
        size_bytes=pdf.stat().st_size,
        file_path=str(pdf),
    )

    registry.set_status(db, document.doc_id, registry.STATUS_EXTRACTING)
    extracted = extractor.extract(pdf)

    registry.set_status(db, document.doc_id, registry.STATUS_CHUNKING)
    chunks = chunker.chunk(extracted, title=document.display_name)

    registry.set_status(db, document.doc_id, registry.STATUS_INDEXING)
    written = vector_store.upsert(store, document.doc_id, chunks, fake_vectors(chunks))

    ready = registry.mark_ready(
        db,
        document.doc_id,
        page_count=extracted.page_count,
        chunk_count=written,
        table_count=extracted.table_count,
        image_count=extracted.picture_count,
        chars_per_page=extracted.chars_per_page,
        ocr_applied=extracted.needs_ocr,
    )
    return ready, chunks


# --- The path runs -------------------------------------------------------


def test_a_pdf_goes_from_file_to_searchable(stores, simple_pdf):
    db, store = stores

    document, chunks = ingest(db, store, simple_pdf)

    assert document.status == registry.STATUS_READY
    assert document.page_count == 3
    assert document.chunk_count == len(chunks) > 0
    assert vector_store.count(store, document.doc_id) == document.chunk_count


def test_what_the_registry_reports_matches_what_is_stored(stores, simple_pdf):
    """A count that drifts from reality is how a half-indexed document hides."""
    db, store = stores

    document, _ = ingest(db, store, simple_pdf)

    assert document.chunk_count == vector_store.count(store, document.doc_id)


def test_stored_chunk_ids_are_the_document_id_and_position(stores, simple_pdf):
    db, store = stores

    document, chunks = ingest(db, store, simple_pdf)

    assert vector_store.chunk_ids(store, document.doc_id) == [
        f"{document.doc_id}:{index}" for index in range(len(chunks))
    ]


# --- The stage gate ------------------------------------------------------


def test_ingesting_the_same_pdf_twice_does_not_double_the_chunks(stores, simple_pdf):
    """The stage gate, across the real path rather than against the store alone."""
    db, store = stores

    first, _ = ingest(db, store, simple_pdf)
    after_first = vector_store.count(store)

    second, _ = ingest(db, store, simple_pdf)

    assert second.doc_id == first.doc_id
    assert vector_store.count(store) == after_first
    assert len(registry.list_documents(db)) == 1


def test_registering_the_same_bytes_again_is_refused(stores, simple_pdf):
    db, store = stores
    ingest(db, store, simple_pdf)
    digest = hashlib.sha256(simple_pdf.read_bytes()).hexdigest()

    with pytest.raises(DuplicateDocumentError):
        registry.register(
            db,
            original_filename="copy.pdf",
            content_hash=digest,
            size_bytes=1,
            file_path="copy.pdf",
        )


def test_everything_survives_reopening_both_stores(tmp_path, simple_pdf):
    """The other half of the gate: nothing depends on the process staying alive."""
    db = registry.connect(tmp_path / "lens.db")
    store = vector_store.connect(tmp_path / "chunks.db")
    document, chunks = ingest(db, store, simple_pdf)
    db.close()
    store.close()

    db = registry.connect(tmp_path / "lens.db")
    store = vector_store.connect(tmp_path / "chunks.db")
    restored = registry.get(db, document.doc_id)
    stored = vector_store.count(store, document.doc_id)
    db.close()
    store.close()

    assert restored.status == registry.STATUS_READY
    assert restored.chunk_count == len(chunks)
    assert stored == len(chunks)


# --- Provenance survives the whole path ----------------------------------


def test_a_searched_chunk_still_knows_its_page_and_boxes(stores, simple_pdf):
    """Everything a citation needs has to survive extraction, chunking and
    storage. Losing the boxes here is invisible until the page viewer is built."""
    db, store = stores
    document, chunks = ingest(db, store, simple_pdf)

    hits = vector_store.search(store, fake_vectors(chunks)[0], doc_ids=[document.doc_id], limit=1)

    assert hits
    assert 1 <= hits[0].page <= 3
    assert hits[0].bboxes
    assert hits[0].doc_id == document.doc_id


def test_page_attribution_survives_to_the_stored_chunk(stores, simple_pdf):
    """The most important check in the project: a marker word appears on exactly
    one page of the fixture, and must still be attributed to that page after
    chunking and storage. An off-by-one here makes every citation wrong."""
    db, store = stores
    document, chunks = ingest(db, store, simple_pdf)

    for expected_page, marker in enumerate(PAGE_MARKERS, start=1):
        holding = [chunk for chunk in chunks if marker in chunk.text]
        assert holding, f"{marker!r} vanished from the pipeline"
        assert all(chunk.page == expected_page for chunk in holding), (
            f"{marker!r} should be on page {expected_page}, got {[chunk.page for chunk in holding]}"
        )


def test_the_context_header_uses_the_registry_display_name(stores, simple_pdf):
    """Citations show the display name, so that is what must be embedded, not
    the raw filename the user happened to upload."""
    db, store = stores

    document, chunks = ingest(db, store, simple_pdf)

    assert chunks[0].context_header.startswith(f"[{document.display_name}")


# --- Rollback ------------------------------------------------------------


def test_rolling_back_a_failed_ingest_leaves_nothing_behind(stores, simple_pdf):
    """A half-indexed document answers some questions and silently skips the
    rest of its own content, which is worse than not being there at all."""
    db, store = stores
    document, _ = ingest(db, store, simple_pdf)

    vector_store.delete_document(store, document.doc_id)
    registry.discard(db, document.doc_id)

    assert vector_store.count(store, document.doc_id) == 0
    assert registry.list_documents(db) == []
    # And the file can be uploaded again rather than being locked out forever.
    assert registry.find_by_hash(db, hashlib.sha256(simple_pdf.read_bytes()).hexdigest()) is None


def test_two_different_pdfs_stay_separately_searchable(stores, simple_pdf, table_pdf):
    """Scoped search is what the document picker relies on."""
    db, store = stores

    first, _ = ingest(db, store, simple_pdf)
    second, _ = ingest(db, store, table_pdf)

    assert first.doc_id != second.doc_id
    assert vector_store.count(store) == first.chunk_count + second.chunk_count

    scoped = vector_store.search(
        store, [0.0] * settings.EMBEDDING_DIMENSIONS, doc_ids=[second.doc_id]
    )
    assert {hit.doc_id for hit in scoped} == {second.doc_id}
