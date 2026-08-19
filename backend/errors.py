"""Typed exceptions with stable error codes.

Every failure in Lens raises one of these. The `code` is a contract: the UI maps
codes to messages, so wording can change freely and nothing matches on text.
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


class CorruptFileError(LensError):
    """The upload could not be opened as a PDF at all."""

    code = "corrupt_file"


class EncryptedPDFError(LensError):
    """Password protected, so its text cannot be read.

    Separate from a corrupt file: the user can fix this one.
    """

    code = "encrypted_pdf"


class FileTooLargeError(LensError):
    """Over the upload size limit."""

    code = "file_too_large"


class TooManyPagesError(LensError):
    """Over the page limit."""

    code = "too_many_pages"


# --- Embedding -----------------------------------------------------------


class EmbeddingFailedError(LensError):
    """The embedding provider failed, after retries.

    Raised immediately for a failure retrying cannot fix, such as a rejected key.
    """

    code = "embedding_failed"


class MissingApiKeyError(LensError):
    """No API key in the environment.

    Its own error so the message can name the variable to set, rather than surfacing
    a provider's authentication error.
    """

    code = "missing_api_key"


# --- Storage -------------------------------------------------------------


class DuplicateDocumentError(LensError):
    """A document with these exact bytes is already in the library.

    Carries the existing `doc_id`, so the caller can point at what is already there.
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

    Vectors from two models occupy different spaces, so mixing them wrecks retrieval
    while everything reports success. Nothing else in Lens fails this quietly.
    """

    code = "embed_model_mismatch"


class VectorStoreError(LensError):
    """The vector store could not be opened or written."""

    code = "vector_store_error"


class GenerationFailedError(LensError):
    """The answer model could not be reached, or returned nothing usable.

    Never reported as an abstention: saying the documents do not cover a question
    when the provider was down would be a lie.
    """

    code = "generation_failed"


class ConversationNotFoundError(LensError):
    """No conversation with this id."""

    code = "conversation_not_found"


class EmptyScopeError(LensError):
    """A chat was given no documents to search.

    Refused rather than stored: every question would then be refused, and the user
    would read that as the app being broken rather than as their own setting.
    """

    code = "empty_scope"


class StoreMismatchError(LensError):
    """The document registry and the vector store disagree about what exists.

    The registry can list documents whose chunks are gone, and every question then
    returns "not found in your documents" for a library that visibly contains them.
    """

    code = "store_mismatch"


class PageNotFoundError(LensError):
    """The requested page is outside this document."""

    code = "page_not_found"


class RenderFailedError(LensError):
    """The page could not be turned into an image.

    Distinct from a missing page: the file itself is gone or unreadable, so a
    citation into it can no longer be checked.
    """

    code = "render_failed"


class UnreadableDocumentError(LensError):
    """The PDF has pages but no readable text, even after OCR.

    Rejected rather than indexed: it would sit in the library answering nothing,
    which a user cannot tell from the system simply not finding an answer.
    """

    code = "unreadable_document"
