"""Tests for shortening a very long message before searching with it.

Everything here is about not making things worse. Condensing is an optimisation,
so every failure path has to end with the original message being searched rather
than with an error or with something shorter but wrong.
"""

from backend.retrieval import condenser
from config import settings


def replying(text: str):
    def chat(_messages):
        return text

    return chat


def long_message(words: str = "background ") -> str:
    return words * (settings.CONDENSE_CHAR_THRESHOLD // len(words) + 10)


# --- when it runs at all -------------------------------------------------


def test_a_short_message_is_left_alone():
    result = condenser.condense("How much leave do I get?", chat=replying("should not be called"))

    assert result.text == "How much leave do I get?"
    assert not result.was_condensed


def test_a_short_message_costs_no_call():
    called = []

    def chat(_messages):
        called.append(1)
        return "x"

    condenser.condense("short question", chat=chat)

    assert called == []


def test_a_message_at_the_threshold_is_not_condensed():
    """The comparison is greater-than, so the threshold is the longest message
    left untouched rather than the shortest one reduced."""
    at_limit = "a" * settings.CONDENSE_CHAR_THRESHOLD

    assert not condenser.needed(at_limit)


def test_a_message_over_the_threshold_is_condensed():
    result = condenser.condense(long_message(), chat=replying("What is the leave entitlement?"))

    assert result.text == "What is the leave entitlement?"
    assert result.was_condensed


# --- what the user still sees --------------------------------------------


def test_the_original_message_is_kept():
    """The reduction is a search aid. The user's own words are what they see and
    what the answer model is asked."""
    original = long_message()
    result = condenser.condense(original, chat=replying("A short question?"))

    assert result.original == original


# --- degrading safely ----------------------------------------------------


def test_an_unreachable_model_falls_back_to_the_original():
    """A weaker search is a far better outcome than refusing to answer."""

    def failing(_messages):
        raise RuntimeError("provider down")

    original = long_message()
    result = condenser.condense(original, chat=failing)

    assert result.text == original
    assert not result.was_condensed


def test_an_empty_reply_falls_back_to_the_original():
    original = long_message()
    result = condenser.condense(original, chat=replying("   "))

    assert result.text == original
    assert not result.was_condensed


def test_a_reply_that_is_not_shorter_is_rejected():
    """Not a reduction, so using it would make the search worse than doing
    nothing at all."""
    original = long_message()
    result = condenser.condense(original, chat=replying(original + " and more"))

    assert result.text == original
    assert not result.was_condensed
