"""The application: startup checks, shared connections, and error translation.

Startup fails rather than degrades. The worst of the three checks is the embedding
model: vectors from two models occupy different spaces, so retrieval rots while
everything reports success.

A document left part-written by a killed process is discarded before the first
request, and every typed error becomes one response shape.
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.api import routes_chat, routes_conversations, routes_documents
from backend.errors import LensError, MissingApiKeyError, StoreMismatchError
from backend.ingestion import pipeline
from backend.storage import registry, vector_store
from config import settings

log = logging.getLogger(__name__)

API_KEY_VARIABLE = "OPENAI_API_KEY"

# The README tells a reader to put their key in `.env`, so the app reads it. An
# exported variable wins: it is a deliberate choice for this one run.
load_dotenv(settings.PROJECT_ROOT / ".env", override=False)

# Which failures are the caller's fault and which are ours. Anything not listed
# is a 500, because an unmapped error is a surprise and a surprise is not a
# client error.
STATUS_BY_CODE = {
    "duplicate_document": 409,
    "file_too_large": 413,
    "too_many_pages": 413,
    "encrypted_pdf": 415,
    "corrupt_file": 415,
    "empty_document": 422,
    "unreadable_document": 422,
    "empty_scope": 422,
    "document_not_found": 404,
    "conversation_not_found": 404,
    "missing_api_key": 503,
    "embedding_failed": 502,
    "generation_failed": 502,
    "extraction_failed": 422,
    "vector_store_error": 503,
    "store_mismatch": 503,
    "page_not_found": 404,
    "render_failed": 422,
}


def check_stores_agree(db, store) -> None:
    """Refuse to start if the registry lists documents whose chunks are missing.

    Only the empty-store case: chunk totals drift legitimately, since a soft-deleted
    document keeps its chunks and a reingest upserts.
    """
    expected = sum(
        document.chunk_count for document in registry.list_documents(db, ready_only=True)
    )
    if expected and vector_store.count(store) == 0:
        raise StoreMismatchError(
            f"the library lists {expected} chunks but the vector store is empty. "
            "One of the two stores has been removed or replaced; re-index to rebuild it."
        )


def check_startup(db, store) -> None:
    """Refuse to start on anything that would make answers quietly wrong.

    Raises the same typed errors the rest of the system uses, so the failure at
    startup names the same cause it would at request time.
    """
    if not os.environ.get(API_KEY_VARIABLE):
        raise MissingApiKeyError(f"set {API_KEY_VARIABLE} in .env")

    vector_store.assert_usable(store)
    # The check worth having: mixing two embedding models degrades retrieval with
    # nothing in the logs.
    registry.assert_embed_model(db)
    check_stores_agree(db, store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the stores once, check them, clean up, and hold them for the process.

    Request handlers share them, since they all run on the same loop. Background
    work opens its own SQLite connection - a connection belongs to the thread that
    made it - which WAL mode makes safe.
    """
    settings.ensure_dirs()
    db = registry.connect()
    store = vector_store.connect()
    check_startup(db, store)

    discarded = pipeline.recover(db, store)
    if discarded:
        log.warning("discarded %d document(s) left by an interrupted run", len(discarded))

    app.state.db = db
    app.state.store = store
    try:
        yield
    finally:
        db.close()
        store.close()


def create_app() -> FastAPI:
    app = FastAPI(title="Lens", version="1.0", lifespan=lifespan)

    @app.exception_handler(LensError)
    async def _lens_error(_request: Request, error: LensError) -> JSONResponse:
        """One shape for every typed failure.

        The UI switches on `code`, never on `message`, so rewording a message can
        never change which case it thinks it is in.
        """
        return JSONResponse(
            status_code=STATUS_BY_CODE.get(error.code, 500),
            content={"code": error.code, "message": str(error)},
        )

    app.include_router(routes_documents.router)
    app.include_router(routes_conversations.router)
    app.include_router(routes_chat.router)
    return app


app = create_app()
