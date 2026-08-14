"""Validating and resolving the citations in an answer.

This is the module that makes the product's central claim true. The model cites a
number; code decides whether that number was supplied, and code looks up the
document, section, page and coordinates behind it. A citation to a document that
was never retrieved cannot be rendered, because the model never had the means to
write one.

The whole design rests on the model handling only the part it cannot get
structurally wrong. It can be wrong about which passage supports its sentence -
nothing here can detect that, and the answer to it is that the source page is one
click away. It cannot be wrong about what document `[2]` refers to, because it
does not decide.

An answer with no valid citation left is not an answer. It is either an
abstention the model phrased as prose, or an ungrounded claim, and both are shown
as "I don't know" rather than as an answer nobody can check.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

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


def _snippet(text: str) -> str:
    """The opening of a passage, cut at a word boundary."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= settings.CITATION_SNIPPET_CHARS:
        return collapsed
    cut = collapsed[: settings.CITATION_SNIPPET_CHARS]
    # Break at the last space so the snippet never ends mid-word, which reads as
    # if the passage itself were truncated.
    if " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut + "…"


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
                snippet=_snippet(hit.text),
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
