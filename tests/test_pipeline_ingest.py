"""Tests for ingestion orchestration.

The claim being tested is the one the library's honesty depends on: a document is
either wholly indexed or wholly absent. Never partly there, answering some
questions and silently skipping the rest of its own content.

Extraction is injected, so nothing here spawns a worker or loads Docling, and the
embedding function is a stand-in, so nothing costs money. What is real is the
SQLite registry and the Milvus store - the rollback has to be shown removing rows
from the actual stores, not from mocks that would agree with anything.
"""

import pymupdf
import pytest

from backend.errors import EmptyDocumentError, FileTooLargeError
from backend.ingestion import pipeline
from backend.ingestion.chunk import Chunk
from backend.ingestion.prepare import Prepared
from backend.storage import files, registry, vector_store
from config import settings


def pdf_bytes(pages: int = 2) -> bytes:
    document = pymupdf.open()
    for number in range(pages):
        document.new_page().insert_text((72, 100), f"Page {number + 1}.")
    data = document.tobytes()
    document.close()
    return data


def chunk(index: int, page: int = 1) -> Chunk:
    return Chunk(
        index=index,
        text=f"Body of chunk {index}.",
        page=page,
        section_path="1. Leave",
        element_type="text",
        token_count=8,
        bboxes=[(1.0, 2.0, 3.0, 4.0)],
        context_header=f"[Doc > 1. Leave > p.{page}]",
    )


def prepared(count: int = 3) -> Prepared:
    return Prepared(
        chunks=[chunk(i, page=i + 1) for i in range(count)],
        page_count=count,
        table_count=1,
        picture_count=0,
        chars_per_page=1200,
        needs_ocr=False,
        seconds=1.5,
    )


def embedder_returning(dimensions: int = settings.EMBEDDING_DIMENSIONS):
    def embed(texts):
        return [[0.01] * dimensions for _ in texts]

    return embed


@pytest.fixture
def stores(tmp_path, monkeypatch):
    """A real registry and a real vector store, isolated per test."""
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    db = registry.connect(tmp_path / "lens.db")
    store = vector_store.connect(tmp_path / "chunks.db")
    yield db, store
    db.close()
    store.close()


# --- accepting an upload -------------------------------------------------


def test_accepting_an_upload_registers_it_and_opens_a_job(stores):
    db, _store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")

    document = registry.get(db, accepted.doc_id)
    assert document.status == registry.STATUS_QUEUED
    assert registry.latest_job(db, accepted.doc_id).job_id == accepted.job_id


def test_the_original_file_is_written_to_disk(stores):
    db, _store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")

    document = registry.get(db, accepted.doc_id)
    from pathlib import Path

    assert Path(document.file_path).exists()


def test_a_rejected_upload_leaves_nothing_behind(stores, monkeypatch):
    """Validation runs before anything is written, so a rejection cannot leave a
    file with no row or a row with no file."""
    db, _store = stores
    monkeypatch.setattr(settings, "MAX_FILE_BYTES", 10)

    with pytest.raises(FileTooLargeError):
        pipeline.accept(db, pdf_bytes(), "big.pdf")

    assert registry.list_documents(db) == []
    assert not list((settings.UPLOAD_DIR).glob("*")) if settings.UPLOAD_DIR.exists() else True


def test_two_files_sharing_a_name_both_index_under_distinct_names(stores):
    db, _store = stores
    first = pipeline.accept(db, pdf_bytes(1), "report.pdf")
    second = pipeline.accept(db, pdf_bytes(2), "report.pdf")

    assert first.display_name != second.display_name


# --- a successful ingest -------------------------------------------------


def test_a_successful_ingest_marks_the_document_ready(stores):
    db, store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")

    written = pipeline.ingest(
        db,
        store,
        accepted.doc_id,
        accepted.job_id,
        prepare_document=lambda _path, _title: prepared(3),
        embed=embedder_returning(),
    )

    document = registry.get(db, accepted.doc_id)
    assert written == 3
    assert document.status == registry.STATUS_READY
    assert document.chunk_count == 3
    assert vector_store.count(store) == 3


def test_the_job_finishes_and_reports_full_progress(stores):
    db, store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")
    pipeline.ingest(
        db,
        store,
        accepted.doc_id,
        accepted.job_id,
        prepare_document=lambda _path, _title: prepared(),
        embed=embedder_returning(),
    )

    job = registry.latest_job(db, accepted.doc_id)
    assert job.finished
    assert job.stage == registry.STATUS_READY
    assert job.progress == 1.0


def test_the_stages_are_recorded_as_they_happen(stores):
    """The status endpoint reads these while the ingest is still running, so each
    stage has to be written before its work rather than after it."""
    db, store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")
    seen = []

    def watching(_path, _title):
        seen.append(registry.get(db, accepted.doc_id).status)
        return prepared()

    def watching_embed(texts):
        seen.append(registry.get(db, accepted.doc_id).status)
        return embedder_returning()(texts)

    pipeline.ingest(
        db,
        store,
        accepted.doc_id,
        accepted.job_id,
        prepare_document=watching,
        embed=watching_embed,
    )

    assert seen == [registry.STATUS_EXTRACTING, registry.STATUS_EMBEDDING]


# --- rollback ------------------------------------------------------------


def test_a_failure_during_extraction_removes_the_document(stores):
    db, store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")

    def exploding(_path, _title):
        raise RuntimeError("the worker died")

    with pytest.raises(RuntimeError):
        pipeline.ingest(db, store, accepted.doc_id, accepted.job_id, prepare_document=exploding)

    assert registry.list_documents(db) == []


def test_a_failure_during_embedding_leaves_no_chunks_behind(stores):
    """The worst case. Chunks written before the failure would still be retrieved
    and cited, from a document the library no longer lists."""
    db, store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")

    def exploding(_texts):
        raise RuntimeError("the provider is down")

    with pytest.raises(Exception):  # noqa: B017 - the embedder wraps it in its own type
        pipeline.ingest(
            db,
            store,
            accepted.doc_id,
            accepted.job_id,
            prepare_document=lambda _p, _t: prepared(),
            embed=exploding,
        )

    assert vector_store.count(store) == 0
    assert registry.list_documents(db) == []


def test_a_rollback_removes_the_stored_file(stores):
    db, store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")
    document = registry.get(db, accepted.doc_id)

    with pytest.raises(RuntimeError):
        pipeline.ingest(
            db,
            store,
            accepted.doc_id,
            accepted.job_id,
            prepare_document=lambda _p, _t: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    assert not files.path_for(document.content_hash).exists()


def test_a_document_yielding_no_chunks_is_rolled_back(stores):
    """It opened and has pages, but nothing indexable came out. Kept, it would sit
    in the library answering nothing."""
    db, store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "blank.pdf")

    with pytest.raises(EmptyDocumentError):
        pipeline.ingest(
            db,
            store,
            accepted.doc_id,
            accepted.job_id,
            prepare_document=lambda _p, _t: prepared(0),
        )

    assert registry.list_documents(db) == []


def test_a_rollback_leaves_other_documents_untouched(stores):
    """One document failing must never affect another. This is the whole reason
    ingestion is per-document rather than per-batch."""
    db, store = stores
    good = pipeline.accept(db, pdf_bytes(1), "good.pdf")
    pipeline.ingest(
        db,
        store,
        good.doc_id,
        good.job_id,
        prepare_document=lambda _p, _t: prepared(2),
        embed=embedder_returning(),
    )

    bad = pipeline.accept(db, pdf_bytes(2), "bad.pdf")
    with pytest.raises(RuntimeError):
        pipeline.ingest(
            db,
            store,
            bad.doc_id,
            bad.job_id,
            prepare_document=lambda _p, _t: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    assert [d.doc_id for d in registry.list_documents(db)] == [good.doc_id]
    assert vector_store.count(store) == 2


def test_a_failed_upload_can_be_retried(stores):
    """The rollback removed the hash along with the row, so the same file is not
    then rejected as a duplicate of the attempt that failed."""
    db, store = stores
    data = pdf_bytes()
    first = pipeline.accept(db, data, "handbook.pdf")

    with pytest.raises(RuntimeError):
        pipeline.ingest(
            db,
            store,
            first.doc_id,
            first.job_id,
            prepare_document=lambda _p, _t: (_ for _ in ()).throw(RuntimeError("boom")),
        )

    second = pipeline.accept(db, data, "handbook.pdf")
    written = pipeline.ingest(
        db,
        store,
        second.doc_id,
        second.job_id,
        prepare_document=lambda _p, _t: prepared(2),
        embed=embedder_returning(),
    )

    assert written == 2


# --- recovery after an interrupted run -----------------------------------


def test_a_document_left_mid_ingest_is_cleaned_up_at_startup(stores):
    """A killed process leaves a row in a transient stage and possibly some
    chunks. There is no way to know how many, so it is not resumed."""
    db, store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")
    registry.set_status(db, accepted.doc_id, registry.STATUS_EMBEDDING)

    removed = pipeline.recover(db, store)

    assert removed == [accepted.doc_id]
    assert registry.list_documents(db) == []


def test_recovery_leaves_ready_documents_alone(stores):
    db, store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")
    pipeline.ingest(
        db,
        store,
        accepted.doc_id,
        accepted.job_id,
        prepare_document=lambda _p, _t: prepared(2),
        embed=embedder_returning(),
    )

    assert pipeline.recover(db, store) == []
    assert len(registry.list_documents(db)) == 1


def test_recovery_on_a_clean_library_does_nothing(stores):
    db, store = stores

    assert pipeline.recover(db, store) == []


def test_chunks_already_written_are_removed_when_a_later_step_fails(stores, monkeypatch):
    """The only case that exercises chunk removal, and the one that matters most.

    Everything else fails before anything is written. Here the chunks are in the
    index and the failure comes after, so a rollback that forgot them would leave
    passages that are still retrieved and cited from a document the library no
    longer lists.
    """
    db, store = stores
    accepted = pipeline.accept(db, pdf_bytes(), "handbook.pdf")

    def failing_mark_ready(*_args, **_kwargs):
        raise RuntimeError("the registry write failed after indexing")

    monkeypatch.setattr(registry, "mark_ready", failing_mark_ready)

    with pytest.raises(RuntimeError):
        pipeline.ingest(
            db,
            store,
            accepted.doc_id,
            accepted.job_id,
            prepare_document=lambda _p, _t: prepared(3),
            embed=embedder_returning(),
        )

    assert vector_store.count(store) == 0
    assert registry.list_documents(db) == []
