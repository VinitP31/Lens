"""The shapes the API accepts and returns.

A rejection returns a `code` as well as a message: the UI decides from the code, and
a reworded message must never change which case it thinks it is in.

Diagnostics come back on every answered turn rather than on request, because
without them "why did it say that?" means reproducing the question.
"""

from pydantic import BaseModel, Field

from backend.storage import conversations


class ErrorResponse(BaseModel):
    """A rejection, in the one shape every failing route returns.

    `code` comes from the exception type and is stable; `message` may be reworded.
    """

    code: str
    message: str


# --- documents -----------------------------------------------------------


class DocumentSummary(BaseModel):
    """One row of the library."""

    doc_id: str
    display_name: str
    status: str
    page_count: int | None = None
    chunk_count: int = 0
    table_count: int = 0
    image_count: int = 0
    size_bytes: int
    ocr_applied: bool = False
    uploaded_at: str


class UploadAccepted(BaseModel):
    """The reply to an upload, sent once validation has passed.

    Carries a job id because indexing has not started yet and the caller polls.
    """

    doc_id: str
    job_id: str
    display_name: str
    page_count: int


class IngestStatus(BaseModel):
    """How far an upload has got.

    `stage` is one of the ingestion statuses, so the UI maps it to wording itself.
    """

    doc_id: str
    job_id: str | None = None
    stage: str
    progress: float
    message: str | None = None
    finished: bool = False
    failure_reason: str | None = None


# --- conversations -------------------------------------------------------


class CreateConversation(BaseModel):
    scope_mode: str = conversations.SCOPE_LIBRARY
    scope_doc_ids: list[str] | None = None
    title: str | None = None


class UpdateConversation(BaseModel):
    """Both fields optional: a caller may rename, rescope, or both."""

    title: str | None = None
    scope_mode: str | None = None
    scope_doc_ids: list[str] | None = None


class ConversationSummary(BaseModel):
    """One row of the sidebar."""

    conv_id: str
    title: str | None = None
    title_is_auto: bool = True
    scope_mode: str
    scope_doc_ids: list[str] | None = None
    created_at: str
    updated_at: str


class CitationOut(BaseModel):
    """A citation as the UI renders it.

    Every field was resolved at answer time and stored on the message, never re-read
    when a chat is reopened.
    """

    n: int
    chunk_id: str
    doc_id: str
    display_name: str
    page: int
    section_path: str = ""
    element_type: str = "text"
    snippet: str = ""
    bboxes: list[list[float]] = Field(default_factory=list)


class MessageOut(BaseModel):
    msg_id: str
    role: str
    content: str
    created_at: str
    citations: list[CitationOut] = Field(default_factory=list)
    intent: str | None = None
    abstained: bool = False


class ConversationDetail(ConversationSummary):
    messages: list[MessageOut] = Field(default_factory=list)


# --- chat ----------------------------------------------------------------


class SendMessage(BaseModel):
    message: str


class Diagnostics(BaseModel):
    """Why this answer looks the way it does.

    `top_score` beside `gate_threshold` explains a refusal without reproducing the
    question, and `rejected_citations` says whether the model is inventing sources.
    """

    top_score: float | None = None
    gate_threshold: float
    retrieved: int = 0
    used: int = 0
    rejected_citations: int = 0
    latency_ms: int = 0
    intent: str | None = None
    # Set when the searched text differs from what was typed, so the UI can show
    # what was actually looked for rather than quietly substituting it.
    searched_as: str | None = None
    condensed: bool = False
    # The model answered part of the question and reported the rest as absent.
    partly_absent: bool = False


class ChatDone(BaseModel):
    """The final event of a streamed answer."""

    message_id: str
    answer: str
    abstained: bool
    reason: str | None = None
    citations: list[CitationOut] = Field(default_factory=list)
    diagnostics: Diagnostics


# --- health --------------------------------------------------------------


class Health(BaseModel):
    """Enough to tell a working backend from one that will fail on first use."""

    status: str
    documents: int
    chunks: int
    embedding_model: str
    answer_model: str
    embed_model_matches: bool
    gate_threshold: float
