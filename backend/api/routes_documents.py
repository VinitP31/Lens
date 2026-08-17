"""Library endpoints: upload, list, status, delete.

The split that matters is in upload. Validation runs before the response is sent,
so a file that is too big, password protected, corrupt or already present is
rejected while the user is still looking at the dialog. Indexing is slow, so it
runs afterwards in the background and the caller polls.

Getting that the other way round - accepting everything and reporting failures
through the job - would make every rejection arrive minutes late, for a reason
that was knowable immediately.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, File, Request, UploadFile

from backend.api.schemas import DocumentSummary, IngestStatus, UploadAccepted
from backend.ingestion import pipeline
from backend.storage import registry

log = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

# Declared once at module level rather than in the signature, so the dependency
# is not rebuilt on every call.
UPLOADED_FILE = File(...)


def _summary(document) -> DocumentSummary:
    return DocumentSummary(
        doc_id=document.doc_id,
        display_name=document.display_name,
        status=document.status,
        page_count=document.page_count,
        chunk_count=document.chunk_count,
        table_count=document.table_count,
        image_count=document.image_count,
        size_bytes=document.size_bytes,
        ocr_applied=document.ocr_applied,
        uploaded_at=document.uploaded_at,
    )


def _index(store, doc_id: str, job_id: str) -> None:
    """The background half of an upload.

    Opens its own SQLite connection rather than borrowing the one the app holds.
    A connection belongs to the thread that created it, and this runs on a worker
    thread - sharing it raises "SQLite objects created in a thread can only be
    used in that same thread" and the document silently never indexes. Two
    connections to one file are safe here because the database is in WAL mode.

    The vector store is shared, because it is a client to one local store rather
    than a per-thread handle.

    Failures are swallowed on purpose. `pipeline.ingest` has already rolled the
    document back by the time this sees the exception, so there is nothing left
    to clean up and nobody to raise to - the response was sent long ago. It is
    logged, and the caller learns of it because the document is gone from the
    library.
    """
    db = registry.connect()
    try:
        pipeline.ingest(db, store, doc_id, job_id)
    except Exception:  # noqa: BLE001 - already rolled back; nowhere to raise
        log.exception("background ingest failed for %s", doc_id)
    finally:
        db.close()


@router.post("", response_model=UploadAccepted, status_code=202)
async def upload(
    request: Request,
    background: BackgroundTasks,
    file: UploadFile = UPLOADED_FILE,
) -> UploadAccepted:
    """Accept a PDF, or reject it immediately with a reason.

    Returns 202 rather than 201: the document exists but is not yet usable, and
    saying "created" would invite the caller to search it.
    """
    data = await file.read()
    accepted = pipeline.accept(request.app.state.db, data, file.filename or "document.pdf")

    background.add_task(_index, request.app.state.store, accepted.doc_id, accepted.job_id)
    return UploadAccepted(**vars(accepted))


@router.get("", response_model=list[DocumentSummary])
async def list_documents(request: Request, ready_only: bool = False) -> list[DocumentSummary]:
    """The library. Soft-deleted documents are never listed."""
    return [
        _summary(document)
        for document in registry.list_documents(request.app.state.db, ready_only=ready_only)
    ]


@router.get("/{doc_id}/status", response_model=IngestStatus)
async def status(request: Request, doc_id: str) -> IngestStatus:
    """How far this document has got.

    Reads the job rather than only the document row, because the job carries the
    fraction and the sentence the UI shows while a stage is taking a while.
    """
    db = request.app.state.db
    document = registry.get(db, doc_id)
    job = registry.latest_job(db, doc_id)

    return IngestStatus(
        doc_id=doc_id,
        job_id=job.job_id if job else None,
        stage=document.status,
        progress=job.progress if job else registry.stage_progress(document.status),
        message=job.message if job else None,
        finished=document.status in registry.TERMINAL_STATUSES,
        failure_reason=document.failure_reason,
    )


@router.delete("/{doc_id}", status_code=204)
async def delete(request: Request, doc_id: str) -> None:
    """Remove a document from the library.

    A soft delete. The row and the file stay, because answers already given cite
    its pages and must keep rendering. Search excludes it from the next query
    onward.
    """
    registry.soft_delete(request.app.state.db, doc_id)
