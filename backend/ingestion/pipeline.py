"""Running one document through ingestion, and cleaning up if it fails.

Any failure after registration removes the chunks, the file and the row. A
half-indexed document is worse than a rejected one: it answers from the pages that
made it in and silently omits the rest.

`accept` is synchronous and does only what is cheap and certain - validate, save,
register - so a too-large or encrypted file is refused while the user is still
looking at the dialog. `ingest` is the slow half and runs in the background.
`recover` cleans up documents left mid-ingest by a killed process.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.errors import EmptyDocumentError
from backend.ingestion import embedder, prepare, validator
from backend.logging import trace
from backend.storage import files, registry, vector_store

log = logging.getLogger(__name__)

# Injected in tests so the suite never spawns Docling or calls a provider.
PrepareFunction = Callable[[Path, str], prepare.Prepared]


@dataclass(frozen=True)
class Accepted:
    """The result of the synchronous half: enough for the API to reply 202."""

    doc_id: str
    job_id: str
    display_name: str
    page_count: int


def accept(
    db,
    data: bytes,
    filename: str,
) -> Accepted:
    """Validate, store the bytes, and register the document.

    Everything here is fast and cannot half-succeed in a way that leaves the
    library wrong. Raises the validator's typed errors, which the API turns into
    a rejection the user sees immediately.
    """

    def seen(digest: str) -> str | None:
        existing = registry.find_by_hash(db, digest)
        return existing.display_name if existing else None

    inspection = validator.validate(data, seen=seen)

    # The file is written before the row exists. A file with no row is invisible
    # and harmless; a row pointing at a file that was never written would break
    # every citation into that document.
    path = files.save(data, inspection.content_hash)

    document = registry.register(
        db,
        original_filename=filename,
        content_hash=inspection.content_hash,
        size_bytes=inspection.size_bytes,
        file_path=str(path),
    )
    job = registry.start_job(db, document.doc_id)

    return Accepted(
        doc_id=document.doc_id,
        job_id=job.job_id,
        display_name=document.display_name,
        page_count=inspection.page_count,
    )


def _rollback(db, store, doc_id: str, content_hash: str) -> None:
    """Remove every trace of a failed ingest.

    Order matters. Chunks go first, because a chunk left in the vector store
    would still be retrieved and cited after the row naming it is gone - an
    answer citing a document the library does not have. The row goes last, since
    it is what tells a later recovery run that this document needs cleaning up at
    all.
    """
    try:
        vector_store.delete_document(store, doc_id)
    except Exception:  # noqa: BLE001 - cleanup continues regardless
        log.exception("could not remove chunks for %s", doc_id)

    files.delete(content_hash)
    registry.discard(db, doc_id)


def ingest(
    db,
    store,
    doc_id: str,
    job_id: str,
    prepare_document: PrepareFunction | None = None,
    embed: embedder.EmbedFunction | None = None,
) -> int:
    """Extract, chunk, embed and index one registered document.

    Returns the number of chunks written.

    The stage is recorded before each step, not after, so a document killed during
    embedding is found in the embedding stage - the only clue available when the
    process died outright. Any failure rolls the document back and re-raises.
    """
    prepare_document = prepare_document or prepare.prepare
    document = registry.get(db, doc_id)
    started = time.perf_counter()
    record = trace.DocumentTrace(doc_id=doc_id, display_name=document.display_name)

    def elapsed() -> int:
        return int((time.perf_counter() - started) * 1000)

    def stage(name: str, message: str | None = None) -> None:
        registry.set_status(db, doc_id, name)
        registry.advance_job(db, job_id, name, message)

    try:
        # Extraction and chunking happen together in a worker process. They are
        # reported as one stage because that is what they are from here: one
        # call, one failure, one duration.
        stage(registry.STATUS_EXTRACTING, "reading the document")
        prepared = prepare_document(Path(document.file_path), document.display_name)

        if not prepared.chunks:
            # Opens, has pages, yields nothing indexable. Rolled back rather than
            # stored as a document that is present and answers nothing.
            raise EmptyDocumentError("no indexable text found in the document")

        record.stage_ms["extract"] = elapsed()
        record.page_count = prepared.page_count
        record.chars_per_page = prepared.chars_per_page
        record.ocr_applied = prepared.needs_ocr
        record.table_count = prepared.table_count
        record.image_count = prepared.picture_count
        record.chunk_count = len(prepared.chunks)

        stage(registry.STATUS_EMBEDDING, f"embedding {len(prepared.chunks)} chunks")
        vectors = embedder.embed_chunks(prepared.chunks, embed=embed)
        record.stage_ms["embed"] = elapsed() - sum(record.stage_ms.values())

        stage(registry.STATUS_INDEXING, "writing to the index")
        written = vector_store.upsert(store, doc_id, prepared.chunks, vectors)
        record.stage_ms["index"] = elapsed() - sum(record.stage_ms.values())

        registry.mark_ready(
            db,
            doc_id,
            page_count=prepared.page_count,
            chunk_count=written,
            table_count=prepared.table_count,
            image_count=prepared.picture_count,
            chars_per_page=prepared.chars_per_page,
            ocr_applied=prepared.needs_ocr,
        )
        registry.finish_job(db, job_id, registry.STATUS_READY)
        record.chunk_count = written
        record.total_ms = elapsed()
        trace.write_document(record)
        return written

    except Exception as error:
        # The stage is read and logged before the rollback, because the rollback
        # removes the job row along with the document. The log is the only place a
        # failed ingest survives, which is what the data model intends: a failed
        # document is not a library entry, so there is nothing to keep a row for.
        failed_at = _stage_of(db, job_id)
        log.warning("ingest failed for %s at stage %s: %s", doc_id, failed_at, error)
        # Written before the rollback removes the row. Once the document is gone
        # this line is the only record that it was ever attempted.
        record.failed_at = failed_at
        record.error = f"{type(error).__name__}: {error}"
        record.total_ms = elapsed()
        trace.write_document(record)
        _rollback(db, store, doc_id, document.content_hash)
        raise


def _stage_of(db, job_id: str) -> str:
    """The stage a job is currently on, read back from its row."""
    row = db.execute("SELECT stage FROM ingestion_jobs WHERE job_id = ?", (job_id,)).fetchone()
    return row["stage"] if row else registry.STATUS_FAILED


def recover(db, store) -> list[str]:
    """Clean up documents left mid-ingestion by an interrupted run.

    Called at startup. Each one may have chunks partly written, and there is no
    way to know how many, so none are resumed. Returns the ids removed, for the
    log - a silent cleanup at startup is the kind of thing that hides a crash
    loop.
    """
    removed: list[str] = []
    for document in registry.unfinished(db):
        _rollback(db, store, document.doc_id, document.content_hash)
        removed.append(document.doc_id)
        log.warning(
            "discarded %s, left in %s by an interrupted run", document.doc_id, document.status
        )
    return removed
