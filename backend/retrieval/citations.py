"""Validating and resolving the citations in an answer.

The model cites a number; code decides whether that number was supplied and looks
up the document, section, page and coordinates. An invented source is not unlikely,
it is unavailable.

An answer with no valid citation left is shown as "I don't know".
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from backend.ingestion.chunk import TYPE_TABLE
from backend.storage.vector_store import Hit
from config import settings

# `[1]`, `[12]`. Only a bracketed number counts, so ordinary brackets in prose
# are left alone and a bare "passage 2" is not treated as a citation - the
# contract in the prompt is explicit, and guessing at near-misses would let a
# malformed reference resolve to a real source.
CITATION = re.compile(r"\[(\d+)\]")

# The same thing with any space in front of it, for removing markers from text
# rather than reading them.
SPACED_CITATION = re.compile(r"[ \t]*\[\d+\]")


@dataclass(frozen=True)
class Citation:
    """One validated citation, resolved to something a reader can open.

    Everything here is looked up by code from the passage the number refers to.
    None of it is taken from the model's reply.

    Stored on the message as it is at answer time, never re-resolved when the
    conversation is displayed again: a document soft-deleted next week must not
    change or break an answer given today.
    """

    number: int
    chunk_id: str
    doc_id: str
    document_name: str
    page: int
    section_path: str
    element_type: str
    snippet: str
    bboxes: list[tuple[float, float, float, float]]


@dataclass(frozen=True)
class Validated:
    """An answer's citations after checking, and the working behind the decision."""

    citations: list[Citation]
    # Numbers the model cited that were never supplied to it. Kept rather than
    # discarded silently: this is the one number that says whether the model is
    # inventing sources, and it belongs in the trace log from the first answer.
    fabricated: list[int]

    @property
    def grounded(self) -> bool:
        """Whether anything the model claimed can be checked against a source."""
        return bool(self.citations)


def parse(answer: str) -> list[int]:
    """Citation numbers in the order they first appear.

    Order is the model's, so citations read in the order the answer uses them.
    Duplicates collapse: a passage cited three times is one source, and listing
    it three times beneath the answer suggests three pieces of evidence.
    """
    seen: list[int] = []
    for match in CITATION.finditer(answer):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    return seen


def _snippet(text: str, element_type: str = "") -> str:
    """The opening of a passage, cut at a word boundary.

    Prose is collapsed onto one line, because a passage's own line breaks are an
    artifact of the page width and reproducing them in a chat reply looks broken.

    A table keeps its line breaks. Collapsing them turns rows into one run of
    pipes and dashes that nothing can read back into a table - and a table is
    exactly the passage a reader most wants to check, because a number belongs to
    the row it sits in and to nothing else.
    """
    if element_type == TYPE_TABLE:
        return _table_snippet(text)

    collapsed = " ".join(text.split())
    if len(collapsed) <= settings.CITATION_SNIPPET_CHARS:
        return collapsed

    cut = collapsed[: settings.CITATION_SNIPPET_CHARS]
    # Prefer the end of a sentence. A passage that stops mid-sentence reads as a
    # half answer, and the reader cannot tell whether the rest mattered.
    sentence = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "))
    if sentence > settings.CITATION_SNIPPET_CHARS // 2:
        return cut[: sentence + 1]
    if " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut + settings.SNIPPET_TRUNCATED_MARK


def _table_snippet(text: str) -> str:
    """A table cut only between rows, never inside one.

    A half row states a label with no value - "Maximum age | 90" with the rest
    gone - and a reader takes what is shown as what the document says. Whole rows
    or nothing, and a mark on its own line when rows were left behind, so the
    screen can say there is more rather than let the table look complete.
    """
    rows = [" ".join(line.split()) for line in text.splitlines()]
    rows = [row for row in rows if row]

    kept: list[str] = []
    used = 0
    for row in rows:
        if used + len(row) > settings.CITATION_TABLE_SNIPPET_CHARS and kept:
            kept.append(settings.SNIPPET_TRUNCATED_MARK)
            break
        kept.append(row)
        used += len(row) + 1

    return "\n".join(kept)


def validate(answer: str, hits: Sequence[Hit], names: dict[str, str]) -> Validated:
    """Check the cited numbers and resolve the ones that were supplied.

    `names` maps document id to display name, resolved by the caller at answer
    time. A missing name falls back to the id rather than raising: an answer that
    is otherwise good should not be thrown away because one document row could
    not be read.
    """
    citations: list[Citation] = []
    fabricated: list[int] = []

    for number in parse(answer):
        # The model was given passages numbered from 1. Anything outside that
        # range it invented, whatever it looks like.
        if not 1 <= number <= len(hits):
            fabricated.append(number)
            continue

        hit = hits[number - 1]
        citations.append(
            Citation(
                number=number,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                document_name=names.get(hit.doc_id, hit.doc_id),
                page=hit.page,
                section_path=hit.section_path,
                element_type=hit.element_type,
                snippet=_snippet(hit.text, hit.element_type),
                bboxes=list(hit.bboxes),
            )
        )

    return Validated(citations=citations, fabricated=fabricated)


def strip_markers(answer: str) -> str:
    """The answer without its citation markers, for a plain-text copy.

    Used where the markers would be noise rather than links. The answer shown in
    the UI keeps them, because they are what makes it checkable.

    The space before a marker goes with it. Removing the marker alone leaves
    "monthly ." wherever a sentence ended with a citation, which is every
    sentence this system produces.
    """
    return " ".join(SPACED_CITATION.sub("", answer).split())
