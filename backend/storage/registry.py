"""The document registry: what is in the library, and what state it is in.

SQLite, one local file, standard library only. A restart loses nothing and there
is no server to run.

This module owns the `documents` table. It answers three questions the rest of
the system keeps asking: have we seen this file before, what is its ingestion
state, and what should a citation call it.
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from backend.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmbedModelMismatchError,
)
from config import settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# The ingestion stages, in the order they happen. A document is only ever left
# in READY or DELETED; the rest are transient, and one found in a transient
# state at startup is the wreckage of an interrupted run.
STATUS_QUEUED = "queued"
STATUS_VALIDATING = "validating"
STATUS_EXTRACTING = "extracting"
STATUS_OCR = "ocr"
STATUS_CHUNKING = "chunking"
STATUS_EMBEDDING = "embedding"
STATUS_INDEXING = "indexing"
STATUS_READY = "ready"
STATUS_FAILED = "failed"
STATUS_DELETED = "deleted"

TERMINAL_STATUSES = frozenset({STATUS_READY, STATUS_DELETED})

INGESTION_STAGES = (
    STATUS_QUEUED,
    STATUS_VALIDATING,
    STATUS_EXTRACTING,
    STATUS_OCR,
    STATUS_CHUNKING,
    STATUS_EMBEDDING,
    STATUS_INDEXING,
    STATUS_READY,
)


@dataclass(frozen=True)
class Document:
    """A row of the registry, as the rest of the system sees it."""

    doc_id: str
    display_name: str
    original_filename: str
    content_hash: str
    size_bytes: int
    status: str
    file_path: str
    uploaded_at: str
    page_count: int | None = None
    chunk_count: int = 0
    table_count: int = 0
    image_count: int = 0
    chars_per_page: int | None = None
    ocr_applied: bool = False
    embed_model: str | None = None
    visibility: str = "all"
    failure_reason: str | None = None
    deleted_at: str | None = None


def _now() -> str:
    """UTC, ISO 8601, to the second. Sorts correctly as a string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open the registry, creating the schema if this is a first run.

    `path` is injectable so tests can use a temporary file, and so nothing in
    the suite can touch a real library.

    Foreign keys are off by default in SQLite and have to be asked for per
    connection. Left off, a message could reference a conversation that does not
    exist and nothing would complain.
    """
    target = Path(path) if path is not None else settings.DB_PATH
    if target != Path(":memory:"):
        target.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(target)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    # Survive an interrupted write rather than corrupting the file, and let a
    # reader continue while a background ingest writes.
    connection.execute("PRAGMA journal_mode = WAL")
    connection.executescript(SCHEMA_PATH.read_text())
    return connection


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        doc_id=row["doc_id"],
        display_name=row["display_name"],
        original_filename=row["original_filename"],
        content_hash=row["content_hash"],
        size_bytes=row["size_bytes"],
        status=row["status"],
        file_path=row["file_path"],
        uploaded_at=row["uploaded_at"],
        page_count=row["page_count"],
        chunk_count=row["chunk_count"],
        table_count=row["table_count"],
        image_count=row["image_count"],
        chars_per_page=row["chars_per_page"],
        ocr_applied=bool(row["ocr_applied"]),
        embed_model=row["embed_model"],
        visibility=row["visibility"],
        failure_reason=row["failure_reason"],
        deleted_at=row["deleted_at"],
    )


def find_by_hash(connection: sqlite3.Connection, content_hash: str) -> Document | None:
    """The document with these exact bytes, deleted ones included.

    Deleted ones count. A file that was uploaded and removed is still the same
    file, and the caller decides whether that means restore or reject.
    """
    row = connection.execute(
        "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
    ).fetchone()
    return _row_to_document(row) if row else None


def _unique_display_name(connection: sqlite3.Connection, name: str) -> str:
    """`Handbook.pdf`, then `Handbook (2).pdf`, and so on.

    Two genuinely different PDFs can share a filename. Their hashes differ so
    both belong in the library, but a citation naming both of them identically
    is unreadable, so the second gets a counter.
    """
    taken = {
        row["display_name"]
        for row in connection.execute("SELECT display_name FROM documents").fetchall()
    }
    if name not in taken:
        return name

    stem, dot, suffix = name.rpartition(".")
    base, extension = (stem, f"{dot}{suffix}") if dot else (name, "")
    counter = 2
    while f"{base} ({counter}){extension}" in taken:
        counter += 1
    return f"{base} ({counter}){extension}"


def register(
    connection: sqlite3.Connection,
    *,
    original_filename: str,
    content_hash: str,
    size_bytes: int,
    file_path: str,
) -> Document:
    """Record a new document as queued, and return it.

    Raises `DuplicateDocumentError` if these bytes are already known. That check
    is a hash lookup on a unique index, so re-uploading the same file costs
    nothing - no extraction, no embedding, no second copy answering every
    question twice.
    """
    existing = find_by_hash(connection, content_hash)
    if existing is not None:
        raise DuplicateDocumentError(
            f"already in the library as {existing.display_name!r}",
            doc_id=existing.doc_id,
        )

    document = Document(
        doc_id=uuid.uuid4().hex[:12],
        display_name=_unique_display_name(connection, original_filename),
        original_filename=original_filename,
        content_hash=content_hash,
        size_bytes=size_bytes,
        status=STATUS_QUEUED,
        file_path=file_path,
        uploaded_at=_now(),
    )
    connection.execute(
        """
        INSERT INTO documents (
            doc_id, display_name, original_filename, content_hash,
            size_bytes, status, file_path, uploaded_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document.doc_id,
            document.display_name,
            document.original_filename,
            document.content_hash,
            document.size_bytes,
            document.status,
            document.file_path,
            document.uploaded_at,
        ),
    )
    connection.commit()
    return document


def get(connection: sqlite3.Connection, doc_id: str, *, include_deleted: bool = False) -> Document:
    """One document by id. Raises rather than returning None.

    A missing document is a real error at every call site in the system, so it
    is raised once here instead of being re-checked everywhere.
    """
    row = connection.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
    if row is None:
        raise DocumentNotFoundError(f"no document {doc_id!r}")

    document = _row_to_document(row)
    if document.deleted_at is not None and not include_deleted:
        raise DocumentNotFoundError(f"document {doc_id!r} was deleted")
    return document


def list_documents(connection: sqlite3.Connection, *, ready_only: bool = False) -> list[Document]:
    """The library, newest first. Deleted documents are never included.

    `ready_only` is what the retrieval scope and the UI picker want: a document
    still being ingested must not be offered as searchable, because searching it
    would silently miss most of its content.
    """
    query = "SELECT * FROM documents WHERE deleted_at IS NULL"
    parameters: tuple[str, ...] = ()
    if ready_only:
        query += " AND status = ?"
        parameters = (STATUS_READY,)
    query += " ORDER BY uploaded_at DESC, doc_id"

    return [_row_to_document(row) for row in connection.execute(query, parameters).fetchall()]


def set_status(
    connection: sqlite3.Connection,
    doc_id: str,
    status: str,
    *,
    failure_reason: str | None = None,
) -> None:
    """Move a document to the next ingestion stage.

    The status is what the upload progress display reads, so it is written as
    each stage begins rather than batched at the end.
    """
    connection.execute(
        "UPDATE documents SET status = ?, failure_reason = ? WHERE doc_id = ?",
        (status, failure_reason, doc_id),
    )
    connection.commit()


def mark_ready(
    connection: sqlite3.Connection,
    doc_id: str,
    *,
    page_count: int,
    chunk_count: int,
    table_count: int = 0,
    image_count: int = 0,
    chars_per_page: int | None = None,
    ocr_applied: bool = False,
    embed_model: str | None = None,
) -> Document:
    """Record the results of a successful ingest and make the document usable.

    The embedding model is stored per document on purpose. If the configured
    model ever changes, mixing its vectors with the old ones wrecks retrieval
    with no error message anywhere, so what was actually used has to be on
    record to be checked against.
    """
    connection.execute(
        """
        UPDATE documents SET
            status = ?, page_count = ?, chunk_count = ?, table_count = ?,
            image_count = ?, chars_per_page = ?, ocr_applied = ?,
            embed_model = ?, failure_reason = NULL
        WHERE doc_id = ?
        """,
        (
            STATUS_READY,
            page_count,
            chunk_count,
            table_count,
            image_count,
            chars_per_page,
            int(ocr_applied),
            embed_model or settings.EMBEDDING_MODEL,
            doc_id,
        ),
    )
    connection.commit()
    return get(connection, doc_id)


def soft_delete(connection: sqlite3.Connection, doc_id: str) -> None:
    """Remove a document from the library without destroying its history.

    The row and the file stay. An answer given last week cites this document by
    page and coordinates, and that citation must keep working after the document
    is removed from the library.
    """
    document = get(connection, doc_id)
    connection.execute(
        "UPDATE documents SET status = ?, deleted_at = ? WHERE doc_id = ?",
        (STATUS_DELETED, _now(), document.doc_id),
    )
    connection.commit()


def discard(connection: sqlite3.Connection, doc_id: str) -> None:
    """Delete the row outright. For a failed ingest, not for a user deletion.

    A part-indexed document would answer some questions and silently skip the rest
    of its own content, so the row goes and the failure survives in the trace log.

    Its jobs go in the same transaction: `ingestion_jobs.doc_id` is a foreign key,
    so leaving them turns every rollback into a second failure.
    """
    with connection:
        connection.execute("DELETE FROM ingestion_jobs WHERE doc_id = ?", (doc_id,))
        connection.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))


def embed_models_in_use(connection: sqlite3.Connection) -> set[str]:
    """Every embedding model the live library's vectors were produced with.

    Normally one. More than one means the configuration changed while documents
    were already indexed.
    """
    rows = connection.execute(
        "SELECT DISTINCT embed_model FROM documents "
        "WHERE embed_model IS NOT NULL AND deleted_at IS NULL"
    ).fetchall()
    return {row["embed_model"] for row in rows}


def assert_embed_model(connection: sqlite3.Connection) -> None:
    """Refuse to start if the library was built with a different embedding model.

    This is the loudest failure in Lens by design. Vectors from two models
    occupy unrelated spaces, so comparing them produces confident nonsense while
    every component reports success - no exception, no warning, nothing in a log
    to notice. An empty library is always fine: there is nothing to disagree
    with yet.
    """
    in_use = embed_models_in_use(connection)
    foreign = in_use - {settings.EMBEDDING_MODEL}
    if foreign:
        raise EmbedModelMismatchError(
            f"library was indexed with {sorted(foreign)}, configured model is "
            f"{settings.EMBEDDING_MODEL!r}. Re-index before starting."
        )


def unfinished(connection: sqlite3.Connection) -> list[Document]:
    """Documents left mid-ingestion, which means a previous run was interrupted.

    Called at startup. Each one is wreckage: its chunks may be partly written,
    so it is cleaned up rather than resumed.
    """
    placeholders = ", ".join("?" for _ in TERMINAL_STATUSES)
    rows = connection.execute(
        f"SELECT * FROM documents WHERE status NOT IN ({placeholders}) AND deleted_at IS NULL",
        tuple(TERMINAL_STATUSES),
    ).fetchall()
    return [_row_to_document(row) for row in rows]


# --- Ingestion jobs ------------------------------------------------------
# A job is what the UI polls while a document is being indexed. The document row
# already holds the stage; the job adds a fraction and a human sentence, so an
# upload that sits on one stage for a minute still shows something honest.


@dataclass(frozen=True)
class Job:
    """One ingestion run, as the status endpoint reports it."""

    job_id: str
    doc_id: str
    stage: str
    progress: float
    message: str | None
    started_at: str
    finished_at: str | None

    @property
    def finished(self) -> bool:
        return self.finished_at is not None


def stage_progress(stage: str) -> float:
    """How far through ingestion a stage is, as a fraction.

    Derived from the position of the stage in `INGESTION_STAGES` rather than
    written down per stage. A stage inserted into that tuple then reports a
    sensible fraction without anyone maintaining a second list that could
    disagree with the first.
    """
    if stage not in INGESTION_STAGES:
        return 0.0
    return INGESTION_STAGES.index(stage) / (len(INGESTION_STAGES) - 1)


def start_job(connection: sqlite3.Connection, doc_id: str) -> Job:
    """Open a job for a document about to be ingested."""
    job = Job(
        job_id=uuid.uuid4().hex[:12],
        doc_id=doc_id,
        stage=STATUS_QUEUED,
        progress=stage_progress(STATUS_QUEUED),
        message=None,
        started_at=_now(),
        finished_at=None,
    )
    with connection:
        connection.execute(
            "INSERT INTO ingestion_jobs (job_id, doc_id, stage, progress, message, started_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (job.job_id, job.doc_id, job.stage, job.progress, job.message, job.started_at),
        )
    return job


def advance_job(
    connection: sqlite3.Connection, job_id: str, stage: str, message: str | None = None
) -> None:
    """Move a job to a stage, updating its fraction to match.

    The fraction is never passed in. Two callers disagreeing about how far
    "embedding" is would make the progress bar jump backwards.
    """
    with connection:
        connection.execute(
            "UPDATE ingestion_jobs SET stage = ?, progress = ?, message = ? WHERE job_id = ?",
            (stage, stage_progress(stage), message, job_id),
        )


def finish_job(connection: sqlite3.Connection, job_id: str, stage: str) -> None:
    """Close a job, successfully or not.

    The final stage is recorded rather than assumed, so a failed job still says
    where it stopped. That is the only record of what went wrong once the
    document row itself has been discarded.
    """
    with connection:
        connection.execute(
            "UPDATE ingestion_jobs SET stage = ?, progress = ?, finished_at = ? WHERE job_id = ?",
            (stage, stage_progress(stage), _now(), job_id),
        )


def latest_job(connection: sqlite3.Connection, doc_id: str) -> Job | None:
    """The most recent job for a document, or None if it has never been ingested."""
    row = connection.execute(
        "SELECT * FROM ingestion_jobs WHERE doc_id = ?"
        " ORDER BY started_at DESC, rowid DESC LIMIT 1",
        (doc_id,),
    ).fetchone()
    if row is None:
        return None
    return Job(
        job_id=row["job_id"],
        doc_id=row["doc_id"],
        stage=row["stage"],
        progress=row["progress"],
        message=row["message"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
    )
