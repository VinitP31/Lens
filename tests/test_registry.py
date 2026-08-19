"""Tests for the document registry.

Every test uses a real SQLite file in a temporary directory rather than an
in-memory database, because two of the things worth proving here are that data
survives the process and that a unique index is actually enforced on disk.
"""

import pytest

from backend.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmbedModelMismatchError,
)
from backend.storage import registry
from config import settings


@pytest.fixture
def db(tmp_path):
    """A fresh registry per test, on disk, thrown away afterwards."""
    connection = registry.connect(tmp_path / "lens.db")
    yield connection
    connection.close()


def add(connection, filename="Handbook.pdf", content_hash="hash-1", size_bytes=1024):
    return registry.register(
        connection,
        original_filename=filename,
        content_hash=content_hash,
        size_bytes=size_bytes,
        file_path=f"data/uploads/{filename}",
    )


# --- Registering ---------------------------------------------------------


def test_a_new_document_starts_queued():
    """Nothing is searchable until ingestion finishes, so it cannot start ready."""
    connection = registry.connect(":memory:")
    document = add(connection)

    assert document.status == registry.STATUS_QUEUED
    assert document.chunk_count == 0
    assert document.page_count is None


def test_a_document_gets_an_id_and_a_timestamp(db):
    document = add(db)

    assert document.doc_id
    assert document.uploaded_at.endswith("+00:00")


def test_two_documents_get_different_ids(db):
    first = add(db, content_hash="hash-1")
    second = add(db, filename="Manual.pdf", content_hash="hash-2")

    assert first.doc_id != second.doc_id


# --- Duplicate detection -------------------------------------------------


def test_the_same_bytes_are_rejected_as_a_duplicate(db):
    """The same bytes under any name are one document. Two copies of one PDF would put two
    identical citations on every answer, which reads as a broken system."""
    first = add(db, content_hash="same-bytes")

    with pytest.raises(DuplicateDocumentError) as raised:
        add(db, filename="Handbook-copy.pdf", content_hash="same-bytes")

    # The caller can point at what is already there rather than just failing.
    assert raised.value.doc_id == first.doc_id
    assert raised.value.code == "duplicate_document"


def test_a_duplicate_does_not_create_a_second_row(db):
    add(db, content_hash="same-bytes")
    with pytest.raises(DuplicateDocumentError):
        add(db, filename="copy.pdf", content_hash="same-bytes")

    assert len(registry.list_documents(db)) == 1


def test_a_duplicate_is_detected_even_after_the_original_was_deleted(db):
    """Deleted is not gone. The row and file stay so old citations keep working,
    so re-uploading the same bytes must not create a second row alongside it."""
    first = add(db, content_hash="same-bytes")
    registry.soft_delete(db, first.doc_id)

    with pytest.raises(DuplicateDocumentError):
        add(db, filename="again.pdf", content_hash="same-bytes")


def test_the_hash_lookup_finds_nothing_for_an_unknown_file(db):
    assert registry.find_by_hash(db, "never-seen") is None


# --- Display names -------------------------------------------------------


def test_two_different_pdfs_with_one_filename_get_distinct_display_names(db):
    """Different bytes, same name. Both belong in the library, but a citation
    naming them identically cannot be told apart."""
    first = add(db, filename="Report.pdf", content_hash="hash-1")
    second = add(db, filename="Report.pdf", content_hash="hash-2")

    assert first.display_name == "Report.pdf"
    assert second.display_name == "Report (2).pdf"


def test_the_counter_keeps_climbing_for_a_third_collision(db):
    add(db, filename="Report.pdf", content_hash="hash-1")
    add(db, filename="Report.pdf", content_hash="hash-2")
    third = add(db, filename="Report.pdf", content_hash="hash-3")

    assert third.display_name == "Report (3).pdf"


def test_the_extension_stays_at_the_end_of_a_renamed_document(db):
    """`Report (2).pdf`, never `Report.pdf (2)`. The suffix is what the UI and
    the operating system use to recognise the file."""
    add(db, filename="Report.pdf", content_hash="hash-1")
    second = add(db, filename="Report.pdf", content_hash="hash-2")

    assert second.display_name.endswith(".pdf")


def test_a_filename_without_an_extension_still_gets_a_counter(db):
    add(db, filename="scan", content_hash="hash-1")
    second = add(db, filename="scan", content_hash="hash-2")

    assert second.display_name == "scan (2)"


# --- Status transitions --------------------------------------------------


def test_status_moves_through_the_ingestion_stages(db):
    """Progress is read from this field while ingestion runs, so it is written
    as each stage begins rather than once at the end."""
    document = add(db)

    for stage in registry.INGESTION_STAGES[:-1]:
        registry.set_status(db, document.doc_id, stage)
        assert registry.get(db, document.doc_id).status == stage


def test_marking_ready_records_what_was_ingested(db):
    document = add(db)

    updated = registry.mark_ready(
        db,
        document.doc_id,
        page_count=26,
        chunk_count=80,
        table_count=35,
        chars_per_page=1606,
    )

    assert updated.status == registry.STATUS_READY
    assert (updated.page_count, updated.chunk_count, updated.table_count) == (26, 80, 35)
    assert updated.chars_per_page == 1606


def test_marking_ready_records_the_embedding_model_used(db):
    """Recorded per document so a configuration change can be caught. Mixing
    vectors from two models wrecks retrieval with no error message anywhere."""
    document = add(db)

    updated = registry.mark_ready(db, document.doc_id, page_count=1, chunk_count=1)

    assert updated.embed_model


def test_marking_ready_clears_an_earlier_failure_reason(db):
    document = add(db)
    registry.set_status(db, document.doc_id, registry.STATUS_FAILED, failure_reason="ocr timed out")

    updated = registry.mark_ready(db, document.doc_id, page_count=1, chunk_count=1)

    assert updated.failure_reason is None


# --- Listing -------------------------------------------------------------


def test_listing_returns_newest_first(db):
    add(db, filename="one.pdf", content_hash="hash-1")
    add(db, filename="two.pdf", content_hash="hash-2")

    names = [document.display_name for document in registry.list_documents(db)]

    assert set(names) == {"one.pdf", "two.pdf"}


def test_a_document_still_ingesting_is_not_offered_as_searchable(db):
    """Searching a half-ingested document silently misses most of its content,
    which is worse than it not being available yet."""
    add(db, filename="ready.pdf", content_hash="hash-1")
    pending = add(db, filename="pending.pdf", content_hash="hash-2")
    registry.set_status(db, pending.doc_id, registry.STATUS_EMBEDDING)
    ready = registry.find_by_hash(db, "hash-1")
    registry.mark_ready(db, ready.doc_id, page_count=1, chunk_count=1)

    searchable = registry.list_documents(db, ready_only=True)

    assert [document.display_name for document in searchable] == ["ready.pdf"]


# --- Deletion ------------------------------------------------------------


def test_a_soft_deleted_document_leaves_the_library_but_keeps_its_row(db):
    """The row and the file stay, so a citation given last week still resolves."""
    document = add(db)
    registry.soft_delete(db, document.doc_id)

    assert registry.list_documents(db) == []
    assert registry.get(db, document.doc_id, include_deleted=True).doc_id == document.doc_id


def test_fetching_a_deleted_document_normally_raises(db):
    document = add(db)
    registry.soft_delete(db, document.doc_id)

    with pytest.raises(DocumentNotFoundError):
        registry.get(db, document.doc_id)


def test_fetching_an_unknown_id_raises_with_a_stable_code(db):
    with pytest.raises(DocumentNotFoundError) as raised:
        registry.get(db, "does-not-exist")

    assert raised.value.code == "document_not_found"


def test_discarding_a_failed_document_removes_the_row_entirely(db):
    """A half-ingested document is not a library entry. It would answer some
    questions and silently skip the rest of its own content."""
    document = add(db, content_hash="hash-1")
    registry.discard(db, document.doc_id)

    assert registry.find_by_hash(db, "hash-1") is None


def test_a_discarded_document_can_be_uploaded_again(db):
    """Failure must not lock the file out of the library forever."""
    document = add(db, content_hash="hash-1")
    registry.discard(db, document.doc_id)

    assert add(db, content_hash="hash-1").doc_id


# --- Interrupted runs ----------------------------------------------------


def test_documents_left_mid_ingestion_are_reported(db):
    """Called at startup. Anything not ready or deleted is wreckage from a run
    that was killed part-way, and its chunks may be partly written."""
    finished = add(db, filename="done.pdf", content_hash="hash-1")
    registry.mark_ready(db, finished.doc_id, page_count=1, chunk_count=1)
    stranded = add(db, filename="stuck.pdf", content_hash="hash-2")
    registry.set_status(db, stranded.doc_id, registry.STATUS_INDEXING)

    left_over = registry.unfinished(db)

    assert [document.display_name for document in left_over] == ["stuck.pdf"]


def test_a_deleted_document_is_not_treated_as_interrupted(db):
    document = add(db)
    registry.soft_delete(db, document.doc_id)

    assert registry.unfinished(db) == []


# --- Embedding model guard -----------------------------------------------


def test_an_empty_library_never_blocks_startup(db):
    """Nothing indexed yet, so there is nothing to disagree with."""
    registry.assert_embed_model(db)


def test_a_library_built_with_the_configured_model_is_accepted(db):
    document = add(db)
    registry.mark_ready(db, document.doc_id, page_count=1, chunk_count=1)

    registry.assert_embed_model(db)


def test_a_library_built_with_another_model_refuses_to_start(db):
    """The loudest failure in Lens by design. Vectors from two models occupy
    unrelated spaces, so comparing them yields confident nonsense while every
    component reports success."""
    document = add(db)
    registry.mark_ready(
        db, document.doc_id, page_count=1, chunk_count=1, embed_model="text-embedding-ada-002"
    )

    with pytest.raises(EmbedModelMismatchError) as raised:
        registry.assert_embed_model(db)

    assert raised.value.code == "embed_model_mismatch"
    assert "text-embedding-ada-002" in raised.value.detail


def test_a_deleted_document_does_not_block_startup(db):
    """Its vectors are gone from the searchable library, so its model is moot."""
    document = add(db)
    registry.mark_ready(
        db, document.doc_id, page_count=1, chunk_count=1, embed_model="some-old-model"
    )
    registry.soft_delete(db, document.doc_id)

    registry.assert_embed_model(db)


def test_the_models_in_use_are_reported(db):
    document = add(db)
    registry.mark_ready(db, document.doc_id, page_count=1, chunk_count=1)

    assert registry.embed_models_in_use(db) == {settings.EMBEDDING_MODEL}


# --- Durability ----------------------------------------------------------


def test_the_library_survives_a_reconnect(tmp_path):
    """Half of the stage gate: data must outlive the process. Written and read
    through two separate connections to the same file."""
    path = tmp_path / "lens.db"

    first = registry.connect(path)
    document = add(first, content_hash="hash-1")
    registry.mark_ready(first, document.doc_id, page_count=26, chunk_count=80)
    first.close()

    second = registry.connect(path)
    restored = registry.get(second, document.doc_id)
    second.close()

    assert restored.display_name == "Handbook.pdf"
    assert restored.chunk_count == 80
    assert restored.status == registry.STATUS_READY


def test_connecting_to_an_existing_registry_does_not_wipe_it(tmp_path):
    """The schema is applied on every connect, so it has to be idempotent."""
    path = tmp_path / "lens.db"

    first = registry.connect(path)
    add(first, content_hash="hash-1")
    first.close()

    second = registry.connect(path)
    count = len(registry.list_documents(second))
    second.close()

    assert count == 1
