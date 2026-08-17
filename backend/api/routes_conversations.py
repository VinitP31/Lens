"""Chat endpoints: create, list, read, rename or rescope, delete.

Reading a chat returns its messages *and* its scope. Restoring only the messages
would mean the same follow-up asked tomorrow searches a different set of
documents and gives a different answer, which reads as a bug rather than as a
setting.

Citations come back exactly as they were stored when the answer was given. They
are never re-resolved against the library, so a document deleted since cannot
change or break an old answer.
"""

from fastapi import APIRouter, Request

from backend.api.schemas import (
    CitationOut,
    ConversationDetail,
    ConversationSummary,
    CreateConversation,
    MessageOut,
    UpdateConversation,
)
from backend.storage import conversations

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _summary(conversation) -> ConversationSummary:
    return ConversationSummary(
        conv_id=conversation.conv_id,
        title=conversation.title,
        title_is_auto=conversation.title_is_auto,
        scope_mode=conversation.scope_mode,
        scope_doc_ids=conversation.scope_doc_ids,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _message(message) -> MessageOut:
    return MessageOut(
        msg_id=message.msg_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        citations=[CitationOut(**citation) for citation in message.citations],
        intent=message.intent,
        # A refusal is stored with no text: the wording belongs to the UI, which
        # alone knows whether suggesting a wider selection would be honest. So an
        # empty assistant turn is a refusal and anything else is an answer.
        #
        # Deliberately not "an assistant turn with no citations". A greeting has
        # none either, and marking it a refusal made the app look as though it
        # had failed to answer "hello".
        abstained=message.role == conversations.ROLE_ASSISTANT and not message.content.strip(),
    )


@router.post("", response_model=ConversationSummary, status_code=201)
async def create(request: Request, body: CreateConversation) -> ConversationSummary:
    """Start a chat. Defaults to searching the whole library."""
    return _summary(
        conversations.create(
            request.app.state.db,
            scope_mode=body.scope_mode,
            scope_doc_ids=body.scope_doc_ids,
            title=body.title,
        )
    )


@router.get("", response_model=list[ConversationSummary])
async def list_all(request: Request) -> list[ConversationSummary]:
    """The sidebar, newest first."""
    return [_summary(c) for c in conversations.list_conversations(request.app.state.db)]


@router.get("/{conv_id}", response_model=ConversationDetail)
async def read(request: Request, conv_id: str) -> ConversationDetail:
    """One chat, with its messages and the scope it was left searching."""
    db = request.app.state.db
    conversation = conversations.get(db, conv_id)
    return ConversationDetail(
        **_summary(conversation).model_dump(),
        messages=[_message(m) for m in conversations.messages(db, conv_id)],
    )


@router.patch("/{conv_id}", response_model=ConversationSummary)
async def update(request: Request, conv_id: str, body: UpdateConversation) -> ConversationSummary:
    """Rename a chat, change what it searches, or both."""
    db = request.app.state.db
    if body.title is not None:
        conversations.rename(db, conv_id, body.title)
    if body.scope_mode is not None:
        conversations.set_scope(db, conv_id, body.scope_mode, body.scope_doc_ids)
    return _summary(conversations.get(db, conv_id))


@router.delete("/{conv_id}", status_code=204)
async def delete(request: Request, conv_id: str) -> None:
    """Remove a chat and its messages. Documents are untouched."""
    conversations.delete(request.app.state.db, conv_id)
