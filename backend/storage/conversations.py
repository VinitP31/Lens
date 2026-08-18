"""Chats and their messages, kept in SQLite.

Streamlit wipes session state on refresh, so none of this can live in memory.

Two stored things look redundant and are not. A conversation stores its scope, so
reopening it searches the same documents as before rather than answering a
follow-up differently tomorrow. And a message stores its citations as JSON exactly
as resolved at the time, never looked up again: a document deleted next week must
not change or break an answer from today.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from backend.errors import ConversationNotFoundError, EmptyScopeError
from config import settings

ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# Whole library, or a chosen subset. There is no third mode: an empty subset is
# rejected rather than stored, because it would refuse every question and look
# broken instead of looking like a setting.
SCOPE_LIBRARY = "library"
SCOPE_SUBSET = "subset"

INTENT_GREETING = "greeting"
INTENT_META = "meta"
INTENT_QUESTION = "question"


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class Conversation:
    conv_id: str
    title: str | None
    title_is_auto: bool
    scope_mode: str
    scope_doc_ids: list[str] | None
    created_at: str
    updated_at: str

    @property
    def is_library_wide(self) -> bool:
        return self.scope_mode == SCOPE_LIBRARY


@dataclass(frozen=True)
class Message:
    msg_id: str
    conv_id: str
    role: str
    content: str
    created_at: str
    citations: list[dict] = field(default_factory=list)
    scope_snapshot: list[str] | None = None
    intent: str | None = None
    gate_passed: bool | None = None
    top_score: float | None = None
    latency_ms: int | None = None


def _to_conversation(row: sqlite3.Row) -> Conversation:
    return Conversation(
        conv_id=row["conv_id"],
        title=row["title"],
        title_is_auto=bool(row["title_is_auto"]),
        scope_mode=row["scope_mode"],
        scope_doc_ids=json.loads(row["scope_doc_ids"]) if row["scope_doc_ids"] else None,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _to_message(row: sqlite3.Row) -> Message:
    return Message(
        msg_id=row["msg_id"],
        conv_id=row["conv_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        citations=json.loads(row["citations"]) if row["citations"] else [],
        scope_snapshot=json.loads(row["scope_snapshot"]) if row["scope_snapshot"] else None,
        intent=row["intent"],
        gate_passed=None if row["gate_passed"] is None else bool(row["gate_passed"]),
        top_score=row["top_score"],
        latency_ms=row["latency_ms"],
    )


def _check_scope(mode: str, doc_ids: list[str] | None) -> list[str] | None:
    """Refuse a scope that cannot answer anything.

    An empty subset is the one invalid state. Every question against it would be
    refused for having nothing to search, and the user would read that as the app
    being broken rather than as the setting they chose.
    """
    if mode == SCOPE_LIBRARY:
        # A library-wide chat stores no list. Keeping one would freeze the scope
        # at the documents present when the chat started, so a document uploaded
        # later would never be searched by it.
        return None
    if not doc_ids:
        raise EmptyScopeError("select at least one document, or search the whole library")
    # Order is kept as given, and duplicates dropped. The UI shows this list back
    # to the user, and a repeated entry would look like a bug.
    seen: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def create(
    connection: sqlite3.Connection,
    *,
    scope_mode: str = SCOPE_LIBRARY,
    scope_doc_ids: list[str] | None = None,
    title: str | None = None,
) -> Conversation:
    """Start a chat. Defaults to searching the whole library."""
    scope = _check_scope(scope_mode, scope_doc_ids)
    now = _now()
    conversation = Conversation(
        conv_id=uuid.uuid4().hex[:12],
        title=title,
        # A title supplied by the caller was chosen deliberately, so it is never
        # overwritten by the automatic one.
        title_is_auto=title is None,
        scope_mode=scope_mode,
        scope_doc_ids=scope,
        created_at=now,
        updated_at=now,
    )
    with connection:
        connection.execute(
            "INSERT INTO conversations"
            " (conv_id, title, title_is_auto, scope_mode, scope_doc_ids, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation.conv_id,
                conversation.title,
                int(conversation.title_is_auto),
                conversation.scope_mode,
                json.dumps(scope) if scope is not None else None,
                now,
                now,
            ),
        )
    return conversation


def get(connection: sqlite3.Connection, conv_id: str) -> Conversation:
    """One chat. Raises rather than returning None, as every caller needs it."""
    row = connection.execute("SELECT * FROM conversations WHERE conv_id = ?", (conv_id,)).fetchone()
    if row is None:
        raise ConversationNotFoundError(f"no conversation {conv_id!r}")
    return _to_conversation(row)


def list_conversations(connection: sqlite3.Connection) -> list[Conversation]:
    """Newest first, which is the order the sidebar shows them in."""
    rows = connection.execute("SELECT * FROM conversations ORDER BY updated_at DESC").fetchall()
    return [_to_conversation(row) for row in rows]


def rename(connection: sqlite3.Connection, conv_id: str, title: str) -> None:
    """Set a title chosen by the user.

    Marks the title as no longer automatic, so the first-question titling can
    never overwrite a name somebody typed.
    """
    get(connection, conv_id)
    with connection:
        connection.execute(
            "UPDATE conversations SET title = ?, title_is_auto = 0, updated_at = ?"
            " WHERE conv_id = ?",
            (title.strip(), _now(), conv_id),
        )


def set_auto_title(connection: sqlite3.Connection, conv_id: str, question: str) -> None:
    """Name a chat after its first real question.

    Does nothing if the chat already has a title the user chose, and nothing if
    an automatic title is already set - the name comes from the *first* question,
    so later ones must not rewrite it.

    Greetings and questions about the app never reach here. A chat called "hi"
    tells nobody anything.
    """
    conversation = get(connection, conv_id)
    if conversation.title is not None or not conversation.title_is_auto:
        return

    flat = " ".join(question.split())
    if len(flat) > settings.TITLE_MAX_CHARS:
        cut = flat[: settings.TITLE_MAX_CHARS]
        # Cut at a word boundary so the title does not end mid-word.
        if " " in cut:
            cut = cut[: cut.rindex(" ")]
        flat = cut + "…"

    with connection:
        connection.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE conv_id = ?",
            (flat, _now(), conv_id),
        )


def set_scope(
    connection: sqlite3.Connection,
    conv_id: str,
    scope_mode: str,
    scope_doc_ids: list[str] | None = None,
) -> None:
    """Change which documents this chat searches."""
    get(connection, conv_id)
    scope = _check_scope(scope_mode, scope_doc_ids)
    with connection:
        connection.execute(
            "UPDATE conversations SET scope_mode = ?, scope_doc_ids = ?, updated_at = ?"
            " WHERE conv_id = ?",
            (scope_mode, json.dumps(scope) if scope is not None else None, _now(), conv_id),
        )


def delete(connection: sqlite3.Connection, conv_id: str) -> None:
    """Remove a chat and its messages. Documents are untouched.

    A real delete, not a soft one. A conversation is the user's own writing; a
    document is evidence other answers may still cite.
    """
    get(connection, conv_id)
    with connection:
        connection.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
        connection.execute("DELETE FROM conversations WHERE conv_id = ?", (conv_id,))


def add_message(
    connection: sqlite3.Connection,
    conv_id: str,
    role: str,
    content: str,
    *,
    citations: list[dict] | None = None,
    scope_snapshot: list[str] | None = None,
    intent: str | None = None,
    gate_passed: bool | None = None,
    top_score: float | None = None,
    latency_ms: int | None = None,
) -> Message:
    """Append a turn.

    `citations` are stored exactly as they were resolved when the answer was
    given, and never looked up again. `scope_snapshot` records what was actually
    searched, because the chat's scope can change later and history would
    otherwise be impossible to read back.
    """
    get(connection, conv_id)
    message = Message(
        msg_id=uuid.uuid4().hex[:12],
        conv_id=conv_id,
        role=role,
        content=content,
        created_at=_now(),
        citations=citations or [],
        scope_snapshot=scope_snapshot,
        intent=intent,
        gate_passed=gate_passed,
        top_score=top_score,
        latency_ms=latency_ms,
    )
    with connection:
        connection.execute(
            "INSERT INTO messages (msg_id, conv_id, role, content, citations, scope_snapshot,"
            " intent, gate_passed, top_score, latency_ms, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                message.msg_id,
                conv_id,
                role,
                content,
                json.dumps(message.citations) if message.citations else None,
                json.dumps(scope_snapshot) if scope_snapshot is not None else None,
                intent,
                None if gate_passed is None else int(gate_passed),
                top_score,
                latency_ms,
                message.created_at,
            ),
        )
        # The sidebar is ordered by this, so a chat rises to the top when it is
        # used rather than when it was renamed.
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE conv_id = ?",
            (message.created_at, conv_id),
        )
    return message


def messages(connection: sqlite3.Connection, conv_id: str) -> list[Message]:
    """Every turn in order, oldest first."""
    rows = connection.execute(
        "SELECT * FROM messages WHERE conv_id = ? ORDER BY created_at, rowid", (conv_id,)
    ).fetchall()
    return [_to_message(row) for row in rows]


def recent_turns(connection: sqlite3.Connection, conv_id: str) -> list[Message]:
    """The last few turns, for rewriting a follow-up into a standalone question.

    Bounded on purpose. Unbounded history grows the prompt without limit, and
    turns from far enough back start describing a different subject, which makes
    the rewrite worse rather than better.
    """
    rows = connection.execute(
        "SELECT * FROM messages WHERE conv_id = ? ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (conv_id, settings.HISTORY_WINDOW_TURNS * 2),
    ).fetchall()
    return [_to_message(row) for row in reversed(rows)]
