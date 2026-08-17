"""The application: startup checks, shared connections, and error translation.

Three decisions live here.

**Startup fails rather than degrades.** An unreachable store, a missing key, or an
embedding model that does not match the one the index was built with all stop the
app from starting. A backend that starts and then gives subtly wrong answers is
worse than one that refuses and says why, and the embedding mismatch is the worst
of the three: vectors from two different models occupy different spaces, so
retrieval quietly rots while every part of the system reports success.

**Wreckage is cleaned up before the first request.** A process killed mid-ingest
leaves a document part-written. It is discarded at startup, so the library never
contains something half-there.

**Every typed error becomes the same response shape.** One handler maps an
exception's `code` to an HTTP status, so no route repeats that mapping and the UI
has one thing to read. A code is stable; a message is not.
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

# The README tells a reader to put their key in `.env`, so the app has to read it.
# Without this line it worked only because a dependency happened to load the file
# on import, which is luck rather than behaviour, and it would break silently the
# day that dependency stopped.
#
# Nothing already in the environment is overwritten: an exported variable is a
# deliberate choice for this one run, and a file on disk must not beat it.
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

    The two stores are separate files and nothing keeps them in step. Delete one,
    restore one from a backup, or fill the disk mid-write, and the library still
    lists its documents while the text behind them is gone. Every question then
    answers "not found in your documents" for a library that plainly contains
    documents, with nothing anywhere to say why.

    Only the empty-store case is checked, not the exact count. Chunk totals drift
    legitimately - a soft-deleted document keeps its chunks, and a reingest
    upserts - so an exact comparison would refuse to start on a healthy library.
    Nothing at all against a registry that expects something is unambiguous.
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

    # Reachable, and holding vectors of the width this build expects.
    vector_store.assert_usable(store)
    # Built with the embedding model now configured. This is the check worth
    # having: mixing two models degrades retrieval with nothing in the logs.
    registry.assert_embed_model(db)
    # And the two stores still describe the same library.
    check_stores_agree(db, store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the stores once, check them, clean up, and hold them for the process.

    Both are opened once and shared by the request handlers, which all run on
    the same loop. Background work does not share the SQLite connection: a
    connection belongs to the thread that made it, and a background task runs on
    another, so it opens its own. WAL mode is what makes two connections to one
    file safe.
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
