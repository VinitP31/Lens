"""Typed exceptions with stable error codes.

Every failure in Lens raises one of these. The `code` is a contract: the UI maps
codes to user-facing messages, so wording can change freely without breaking
anything. Nothing anywhere may match on exception text.

This file grows one build stage at a time, gaining an exception when the code
that raises it is written.
"""


class LensError(Exception):
    """Base for every Lens failure.

    Subclasses set `code` to a short, stable identifier. `detail` carries the
    specifics of this particular occurrence, such as an actual page count.
    """

    code = "lens_error"

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        super().__init__(detail or self.code)


# --- Extraction ----------------------------------------------------------


class ExtractionFailedError(LensError):
    """Docling could not process the PDF at all."""

    code = "extraction_failed"


class EmptyDocumentError(LensError):
    """The PDF has no pages, or no extractable content on any page."""

    code = "empty_document"


# --- Embedding -----------------------------------------------------------


class EmbeddingFailedError(LensError):
    """The embedding provider could not be reached, or returned something unusable.

    Raised only after retries are exhausted, or immediately for a failure that
    retrying cannot fix, such as a missing or rejected key.
    """

    code = "embedding_failed"


class MissingApiKeyError(LensError):
    """No API key in the environment.

    Its own error because it is the one failure a user can fix themselves, and
    the message needs to say which variable to set rather than surfacing a
    provider's authentication error.
    """

    code = "missing_api_key"


# --- Storage -------------------------------------------------------------


class DuplicateDocumentError(LensError):
    """A document with these exact bytes is already in the library.

    Carries the existing document's id in `doc_id`, so the caller can point at
    what is already there instead of reporting a bare failure.
    """

    code = "duplicate_document"

    def __init__(self, detail: str = "", doc_id: str = "") -> None:
        self.doc_id = doc_id
        super().__init__(detail)


class DocumentNotFoundError(LensError):
    """No document with this id, or it has been deleted."""

    code = "document_not_found"


class EmbedModelMismatchError(LensError):
    """The configured embedding model is not the one the collection was built with.

    Refuse to start rather than carry on. Vectors from two different models
    occupy different spaces, so mixing them wrecks retrieval while every part of
    the system reports success. Nothing else in Lens fails this quietly.
    """

    code = "embed_model_mismatch"


class VectorStoreError(LensError):
    """The vector store could not be opened or written."""

    code = "vector_store_error"
