"""Tests for message classification and follow-up rewriting.

The model call is injected, so nothing here touches the network and no test
depends on what a model would actually say. Each test supplies the exact reply it
needs and checks what the code does with it.

The important tests are the ones about failure. This step sits in front of search
and can only make things worse if it fails in the wrong direction, so every
degraded path has to end in "search it as typed".
"""

from backend.retrieval import analyzer
from backend.storage.conversations import (
    INTENT_GREETING,
    INTENT_META,
    INTENT_QUESTION,
    ROLE_ASSISTANT,
    ROLE_USER,
    Message,
)


def replying(text: str):
    def chat(_messages):
        return text

    return chat


def json_reply(intent: str, standalone: str):
    return replying(f'{{"intent": "{intent}", "standalone": "{standalone}"}}')


def turn(role: str, content: str, scope: list[str] | None = None) -> Message:
    return Message(
        msg_id="m1",
        conv_id="c1",
        role=role,
        content=content,
        created_at="2026-01-01T00:00:00+00:00",
        scope_snapshot=scope,
    )


# --- what kind of message ------------------------------------------------


def test_a_greeting_is_recognised_and_does_not_search():
    """Otherwise "hi" is searched, matches nothing, and comes back as "I could not
    find that in your documents" - correct by the rules, absurd to a person."""
    result = analyzer.analyze("hi there", chat=json_reply(INTENT_GREETING, "hi there"))

    assert result.intent == INTENT_GREETING
    assert not result.needs_search


def test_a_question_about_the_app_does_not_search():
    result = analyzer.analyze(
        "which documents are loaded?", chat=json_reply(INTENT_META, "which documents are loaded?")
    )

    assert result.intent == INTENT_META
    assert not result.needs_search


def test_a_real_question_searches():
    result = analyzer.analyze(
        "how much leave do I get?", chat=json_reply(INTENT_QUESTION, "how much leave do I get?")
    )

    assert result.needs_search


# --- rewriting a follow-up -----------------------------------------------


def test_a_bare_follow_up_is_rewritten_to_stand_alone():
    result = analyzer.analyze(
        "and for part-time?",
        history=[turn(ROLE_USER, "how much annual leave do I get?")],
        chat=json_reply(
            INTENT_QUESTION, "What is the annual leave entitlement for part-time staff?"
        ),
    )

    assert result.standalone == "What is the annual leave entitlement for part-time staff?"
    assert result.was_rewritten


def test_the_users_own_words_are_kept_as_the_original():
    """The user sees what they typed. The rewrite is only what gets searched."""
    result = analyzer.analyze(
        "and for part-time?",
        chat=json_reply(INTENT_QUESTION, "What leave do part-time staff get?"),
    )

    assert result.original == "and for part-time?"


def test_a_question_that_already_stands_alone_is_not_marked_as_rewritten():
    """The UI shows the searched form only when it differs, so the user is never
    told their words were changed when they were not."""
    question = "What is the minimum password length?"
    result = analyzer.analyze(question, chat=json_reply(INTENT_QUESTION, question))

    assert not result.was_rewritten


def test_a_greeting_is_never_rewritten_whatever_the_model_returns():
    """There is nothing to search, so a rewrite could only mislead the user about
    what was done with their words."""
    result = analyzer.analyze(
        "hi", chat=json_reply(INTENT_GREETING, "What is the annual leave policy?")
    )

    assert result.standalone == "hi"
    assert not result.was_rewritten


# --- history -------------------------------------------------------------


def test_history_is_passed_to_the_model():
    seen = {}

    def chat(messages):
        seen["content"] = messages[1]["content"]
        return f'{{"intent": "{INTENT_QUESTION}", "standalone": "rewritten"}}'

    analyzer.analyze(
        "and for part-time?",
        history=[
            turn(ROLE_USER, "how much annual leave do I get?"),
            turn(ROLE_ASSISTANT, "Twenty days a year."),
        ],
        chat=chat,
    )

    assert "how much annual leave do I get?" in seen["content"]
    assert "Twenty days a year." in seen["content"]


def test_a_change_of_documents_is_marked_in_the_history():
    """A chat about one document that switches to another, then gets a bare
    follow-up, would otherwise have that follow-up rewritten into a question
    about the old subject and searched in the new documents."""
    seen = {}

    def chat(messages):
        seen["content"] = messages[1]["content"]
        return f'{{"intent": "{INTENT_QUESTION}", "standalone": "x"}}'

    analyzer.analyze(
        "and that one?",
        history=[
            turn(ROLE_USER, "what is the notice period?", scope=["doc1"]),
            turn(ROLE_ASSISTANT, "One month.", scope=["doc1"]),
            turn(ROLE_USER, "what about overtime?", scope=["doc2"]),
        ],
        chat=chat,
    )

    assert "changed which documents" in seen["content"]


def test_an_unchanged_scope_is_not_marked():
    seen = {}

    def chat(messages):
        seen["content"] = messages[1]["content"]
        return f'{{"intent": "{INTENT_QUESTION}", "standalone": "x"}}'

    analyzer.analyze(
        "and overtime?",
        history=[
            turn(ROLE_USER, "notice period?", scope=["doc1"]),
            turn(ROLE_ASSISTANT, "One month.", scope=["doc1"]),
        ],
        chat=chat,
    )

    assert "changed which documents" not in seen["content"]


def test_no_history_still_works():
    result = analyzer.analyze("first question", chat=json_reply(INTENT_QUESTION, "first question"))

    assert result.needs_search


# --- degrading safely ----------------------------------------------------


def test_an_unreachable_model_falls_back_to_searching_as_typed():
    """Searching something that did not need it costs one lookup and an honest
    refusal. Not searching something that did would swallow a real question."""

    def failing(_messages):
        raise RuntimeError("provider down")

    result = analyzer.analyze("how much leave do I get?", chat=failing)

    assert result.intent == INTENT_QUESTION
    assert result.standalone == "how much leave do I get?"
    assert result.degraded


def test_a_reply_that_is_not_json_falls_back():
    result = analyzer.analyze("a question", chat=replying("I think this is a question!"))

    assert result.intent == INTENT_QUESTION
    assert result.degraded


def test_an_unknown_intent_falls_back():
    """A value outside the three cannot be acted on, and guessing which one was
    meant would be worse than searching."""
    result = analyzer.analyze("a question", chat=json_reply("chitchat", "a question"))

    assert result.intent == INTENT_QUESTION
    assert result.degraded


def test_json_wrapped_in_prose_is_still_read():
    """Models occasionally add a sentence or a code fence around the object."""
    result = analyzer.analyze(
        "a question",
        chat=replying(
            f'Sure:\n```json\n{{"intent": "{INTENT_GREETING}", "standalone": "hi"}}\n```'
        ),
    )

    assert result.intent == INTENT_GREETING
    assert not result.degraded


def test_a_missing_rewrite_uses_the_message_itself():
    """Recoverable, unlike a missing intent: there is always something to search."""
    result = analyzer.analyze("a question", chat=replying(f'{{"intent": "{INTENT_QUESTION}"}}'))

    assert result.standalone == "a question"
    assert not result.degraded


def test_a_successful_analysis_is_not_marked_degraded():
    result = analyzer.analyze("a question", chat=json_reply(INTENT_QUESTION, "a question"))

    assert not result.degraded
