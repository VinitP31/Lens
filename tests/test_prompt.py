"""Tests for prompt assembly.

The prompt is the only place in Lens where behaviour is written in English, so
these tests pin the properties that behaviour depends on: the order of the
blocks, the fact that the abstention marker is the one in settings, and that the
model is given source labels it is told never to write itself.

None of this asserts on phrasing. The wording is expected to change at Stage 5
and after; the structure is not.
"""

from backend.retrieval import prompt
from backend.storage.vector_store import Hit
from config import settings


def hit(chunk_id: str = "doc1:4", doc_id: str = "doc1", section: str = "4.2 Leave") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page=17,
        section_path=section,
        element_type="text",
        text="Employees who have completed twelve months of service accrue leave monthly.",
        bboxes=[(1.0, 2.0, 3.0, 4.0)],
        similarity=0.7,
        raw_distance=0.7,
    )


NAMES = {"doc1": "Employee Handbook", "doc2": "Onboarding Guide"}


# --- the fixed order -----------------------------------------------------


def test_the_instruction_blocks_come_in_the_specified_order():
    """Role and scope, grounding, citation contract, abstention contract.

    Order matters for cost, because a provider only discounts a byte-identical
    prefix, and for behaviour, because the model is told what it may use before
    it is shown what it has.
    """
    text = prompt.SYSTEM_PROMPT

    scope = text.index("numbered passages")
    grounding = text.index("only from the numbered passages")
    citation = text.index("Cite every passage")
    abstention = text.index(settings.ABSTENTION_MARKER)

    assert scope < grounding < citation < abstention


def test_the_instructions_never_mention_a_document_from_the_corpus():
    """No document-specific logic anywhere, the prompt included. A prompt naming
    a sample document would work on the corpus and fail on an unseen PDF."""
    lowered = prompt.SYSTEM_PROMPT.lower()

    assert "handbook" not in lowered
    assert "troy" not in lowered
    assert "rfp" not in lowered


def test_the_abstention_marker_is_the_one_code_checks_for():
    """The prompt asks for the exact marker code checks for. Two copies that drift
    abstains and then reports the abstention to the user as an answer."""
    assert settings.ABSTENTION_MARKER in prompt.SYSTEM_PROMPT


def test_the_system_prompt_is_stable_between_calls():
    """A prefix rebuilt per call cannot be discounted and cannot be reasoned
    about. It is a constant, and this fails if it becomes a function of input."""
    first = prompt.assemble("one question", [hit()], NAMES)
    second = prompt.assemble("an entirely different question", [hit()], NAMES)

    assert first[0]["content"] == second[0]["content"]


# --- passages ------------------------------------------------------------


def test_passages_are_numbered_from_one():
    """One, because that is what the model is asked to cite and what a reader
    sees. The offset from a list index is why resolution is written once."""
    text = prompt.passages([hit(), hit(chunk_id="doc2:1", doc_id="doc2")], NAMES)

    assert text.startswith("[1] ")
    assert "[2] " in text
    assert "[0]" not in text


def test_a_passage_carries_its_document_section_and_page():
    text = prompt.passages([hit()], NAMES)

    assert "Employee Handbook" in text
    assert "4.2 Leave" in text
    assert f"{settings.CONTEXT_PAGE_PREFIX}17" in text


def test_a_passage_with_no_section_still_labels_its_document_and_page():
    """Heading detection degrades rather than failing, so a chunk can arrive with
    no section path. The label must not then read as an empty field."""
    text = prompt.passages([hit(section="")], NAMES)

    assert "Employee Handbook" in text
    assert settings.CONTEXT_SEPARATOR * 2 not in text


def test_the_label_uses_the_same_format_as_a_chunk_context_header():
    """Two nearly-matching formats is one more thing to get wrong later."""
    text = prompt.label(hit(), "Employee Handbook")

    assert settings.CONTEXT_SEPARATOR in text
    assert settings.CONTEXT_PAGE_PREFIX in text


def test_an_unknown_document_id_falls_back_to_the_id():
    text = prompt.passages([hit(doc_id="unregistered")], {})

    assert "unregistered" in text


# --- assembly ------------------------------------------------------------


def test_the_question_comes_after_the_passages():
    """A question placed before the evidence is answered from memory of it."""
    messages = prompt.assemble("How much leave accrues?", [hit()], NAMES)
    user = messages[1]["content"]

    assert user.index("[1]") < user.index("How much leave accrues?")


def test_assembly_produces_a_system_message_then_a_user_message():
    messages = prompt.assemble("A question", [hit()], NAMES)

    assert [message["role"] for message in messages] == ["system", "user"]


def test_the_passage_text_itself_reaches_the_model():
    messages = prompt.assemble("A question", [hit()], NAMES)

    assert "accrue leave monthly" in messages[1]["content"]
