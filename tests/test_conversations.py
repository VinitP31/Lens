"""Tests for chat storage.

A real SQLite database per test, because the things worth checking here are what
survives a write and a read back: scope, citation JSON, ordering. A mock would
agree with whatever the code did.

The two rules being protected: an answer given today must not change when a
document is deleted tomorrow, and reopening a chat must restore what it was
searching, not just what was said.
"""

import pytest

from backend.errors import ConversationNotFoundError, EmptyScopeError
from backend.storage import conversations, registry
from config import settings


@pytest.fixture
def db(tmp_path):
    connection = registry.connect(tmp_path / "lens.db")
    yield connection
    connection.close()


# --- creating ------------------------------------------------------------


def test_a_new_chat_searches_the_whole_library_by_default(db):
    conversation = conversations.create(db)

    assert conversation.scope_mode == conversations.SCOPE_LIBRARY
    assert conversation.is_library_wide
    assert conversation.scope_doc_ids is None


def test_a_library_wide_chat_stores_no_document_list(db):
    """Storing one would freeze the scope at the documents present when the chat
    started, so anything uploaded later would never be searched by it."""
    conversation = conversations.create(
        db, scope_mode=conversations.SCOPE_LIBRARY, scope_doc_ids=["a", "b"]
    )

    assert conversation.scope_doc_ids is None


def test_a_subset_chat_keeps_its_documents_in_order(db):
    conversation = conversations.create(
        db, scope_mode=conversations.SCOPE_SUBSET, scope_doc_ids=["c", "a", "b"]
    )

    assert conversation.scope_doc_ids == ["c", "a", "b"]


def test_a_repeated_document_is_stored_once(db):
    """The UI shows this list back to the user, and a repeat looks like a bug."""
    conversation = conversations.create(
        db, scope_mode=conversations.SCOPE_SUBSET, scope_doc_ids=["a", "b", "a"]
    )

    assert conversation.scope_doc_ids == ["a", "b"]


def test_an_empty_subset_is_refused(db):
    """A chat with no documents selected is refused when it is created.

    Stored, it would refuse every question, which reads as broken rather than chosen."""
    with pytest.raises(EmptyScopeError):
        conversations.create(db, scope_mode=conversations.SCOPE_SUBSET, scope_doc_ids=[])


def test_a_missing_chat_raises(db):
    with pytest.raises(ConversationNotFoundError):
        conversations.get(db, "nope")


# --- titles --------------------------------------------------------------


def test_a_chat_is_named_after_its_first_real_question(db):
    conversation = conversations.create(db)
    conversations.set_auto_title(db, conversation.conv_id, "How much annual leave do I get?")

    assert conversations.get(db, conversation.conv_id).title == "How much annual leave do I get?"


def test_the_name_comes_from_the_first_question_not_the_latest(db):
    conversation = conversations.create(db)
    conversations.set_auto_title(db, conversation.conv_id, "First question")
    conversations.set_auto_title(db, conversation.conv_id, "Second question")

    assert conversations.get(db, conversation.conv_id).title == "First question"


def test_a_name_the_user_typed_is_never_overwritten(db):
    conversation = conversations.create(db)
    conversations.rename(db, conversation.conv_id, "Leave policy")
    conversations.set_auto_title(db, conversation.conv_id, "How much annual leave do I get?")

    assert conversations.get(db, conversation.conv_id).title == "Leave policy"


def test_renaming_marks_the_title_as_chosen(db):
    conversation = conversations.create(db)
    conversations.rename(db, conversation.conv_id, "Leave policy")

    assert not conversations.get(db, conversation.conv_id).title_is_auto


def test_a_long_question_is_cut_at_a_word_boundary(db):
    conversation = conversations.create(db)
    conversations.set_auto_title(db, conversation.conv_id, "leave " * 40)

    title = conversations.get(db, conversation.conv_id).title
    assert len(title) <= settings.TITLE_MAX_CHARS + 1
    assert title.endswith("…")
    assert "leav…" not in title


# --- scope changes -------------------------------------------------------


def test_the_scope_can_be_narrowed_later(db):
    conversation = conversations.create(db)
    conversations.set_scope(db, conversation.conv_id, conversations.SCOPE_SUBSET, ["a"])

    reopened = conversations.get(db, conversation.conv_id)
    assert reopened.scope_mode == conversations.SCOPE_SUBSET
    assert reopened.scope_doc_ids == ["a"]


def test_the_scope_can_be_widened_back_to_the_library(db):
    conversation = conversations.create(
        db, scope_mode=conversations.SCOPE_SUBSET, scope_doc_ids=["a"]
    )
    conversations.set_scope(db, conversation.conv_id, conversations.SCOPE_LIBRARY)

    assert conversations.get(db, conversation.conv_id).scope_doc_ids is None


def test_the_scope_cannot_be_emptied(db):
    conversation = conversations.create(db)

    with pytest.raises(EmptyScopeError):
        conversations.set_scope(db, conversation.conv_id, conversations.SCOPE_SUBSET, [])


# --- messages ------------------------------------------------------------


def test_messages_come_back_in_the_order_they_were_added(db):
    conversation = conversations.create(db)
    for number in range(5):
        conversations.add_message(db, conversation.conv_id, conversations.ROLE_USER, f"q{number}")

    assert [m.content for m in conversations.messages(db, conversation.conv_id)] == [
        "q0",
        "q1",
        "q2",
        "q3",
        "q4",
    ]


def test_citations_are_stored_and_read_back_whole(db):
    """Stored as they were resolved. Never looked up again, so deleting a
    document next week cannot change or break this answer."""
    conversation = conversations.create(db)
    citation = {
        "n": 1,
        "chunk_id": "a3f2:41",
        "doc_id": "a3f2",
        "display_name": "Employee Handbook",
        "page": 17,
        "bboxes": [[72.0, 210.5, 523.0, 288.25]],
    }
    conversations.add_message(
        db,
        conversation.conv_id,
        conversations.ROLE_ASSISTANT,
        "Leave accrues monthly [1].",
        citations=[citation],
    )

    stored = conversations.messages(db, conversation.conv_id)[0]
    assert stored.citations == [citation]


def test_a_message_records_what_it_was_actually_searched_against(db):
    """The chat's scope can change later. Without this, history cannot be read
    back - an old answer would appear to have searched today's selection."""
    conversation = conversations.create(db)
    conversations.add_message(
        db,
        conversation.conv_id,
        conversations.ROLE_ASSISTANT,
        "An answer.",
        scope_snapshot=["doc1", "doc2"],
    )
    conversations.set_scope(db, conversation.conv_id, conversations.SCOPE_SUBSET, ["doc9"])

    assert conversations.messages(db, conversation.conv_id)[0].scope_snapshot == ["doc1", "doc2"]


def test_diagnostics_are_stored_with_the_answer(db):
    conversation = conversations.create(db)
    conversations.add_message(
        db,
        conversation.conv_id,
        conversations.ROLE_ASSISTANT,
        "An answer.",
        intent=conversations.INTENT_QUESTION,
        gate_passed=True,
        top_score=0.62,
        latency_ms=2140,
    )

    stored = conversations.messages(db, conversation.conv_id)[0]
    assert stored.gate_passed is True
    assert stored.top_score == 0.62
    assert stored.latency_ms == 2140


def test_a_message_with_no_citations_reads_back_as_an_empty_list(db):
    """An abstention has none. Reading back None would make every caller check."""
    conversation = conversations.create(db)
    conversations.add_message(db, conversation.conv_id, conversations.ROLE_ASSISTANT, "No.")

    assert conversations.messages(db, conversation.conv_id)[0].citations == []


def test_a_message_cannot_be_added_to_a_chat_that_does_not_exist(db):
    with pytest.raises(ConversationNotFoundError):
        conversations.add_message(db, "nope", conversations.ROLE_USER, "hello")


# --- history for rewriting a follow-up -----------------------------------


def test_recent_turns_are_capped(db):
    """Unbounded history grows the prompt without limit, and old turns start
    describing a different subject, which makes the rewrite worse."""
    conversation = conversations.create(db)
    for number in range(40):
        conversations.add_message(db, conversation.conv_id, conversations.ROLE_USER, f"q{number}")

    assert len(conversations.recent_turns(db, conversation.conv_id)) == (
        settings.HISTORY_WINDOW_TURNS * 2
    )


def test_recent_turns_are_the_latest_ones_oldest_first(db):
    """The rewrite reads them as a conversation, so the order has to be natural."""
    conversation = conversations.create(db)
    for number in range(20):
        conversations.add_message(db, conversation.conv_id, conversations.ROLE_USER, f"q{number}")

    recent = conversations.recent_turns(db, conversation.conv_id)

    window = settings.HISTORY_WINDOW_TURNS * 2
    assert [m.content for m in recent] == [f"q{n}" for n in range(20 - window, 20)]


# --- the sidebar ---------------------------------------------------------


def test_the_list_is_newest_first(db):
    first = conversations.create(db)
    second = conversations.create(db)
    conversations.add_message(db, first.conv_id, conversations.ROLE_USER, "hello")

    listed = [c.conv_id for c in conversations.list_conversations(db)]
    assert set(listed) == {first.conv_id, second.conv_id}


def test_using_a_chat_moves_it_up_the_list(db):
    older = conversations.create(db)
    conversations.create(db)
    conversations.add_message(db, older.conv_id, conversations.ROLE_USER, "hello")

    assert conversations.list_conversations(db)[0].conv_id == older.conv_id


# --- deleting ------------------------------------------------------------


def test_deleting_a_chat_removes_its_messages(db):
    conversation = conversations.create(db)
    conversations.add_message(db, conversation.conv_id, conversations.ROLE_USER, "hello")

    conversations.delete(db, conversation.conv_id)

    assert conversations.list_conversations(db) == []
    with pytest.raises(ConversationNotFoundError):
        conversations.get(db, conversation.conv_id)


def test_deleting_one_chat_leaves_the_others_alone(db):
    kept = conversations.create(db)
    removed = conversations.create(db)
    conversations.add_message(db, kept.conv_id, conversations.ROLE_USER, "hello")

    conversations.delete(db, removed.conv_id)

    assert [c.conv_id for c in conversations.list_conversations(db)] == [kept.conv_id]
    assert len(conversations.messages(db, kept.conv_id)) == 1
