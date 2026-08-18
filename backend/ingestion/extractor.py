"""Docling extraction with page and coordinate provenance.

Turns a PDF into a flat, ordered list of elements. Every element knows its text,
its page, where it sits on that page, what kind of thing it is, and which section
it came from. That provenance is what makes citations clickable, so it is
captured here or not at all.

Nothing in this module knows anything about a specific document. Structure is
read from Docling's own labels, never from filenames, titles or page offsets.
"""

import re
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pymupdf
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    OcrMode,
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.document import (
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.labels import DocItemLabel

from backend.errors import EmptyDocumentError, ExtractionFailedError
from backend.ingestion.chunk import TYPE_FIGURE_CAPTION, TYPE_TABLE, TYPE_TEXT
from config import settings

# The three type names above are defined in `chunk`, not here, so that retrieval
# can tell a table from a sentence without importing this module and, with it,
# Docling. They are re-exported because every caller already reads them from the
# extractor.


# Labels dropped before anything downstream sees them.
#
# PAGE_HEADER / PAGE_FOOTER: "Confidential - Page 12 of 84" on every page would
# be embedded into every chunk, making all chunks slightly more alike, which
# compresses the range of similarity scores and corrupts the confidence gate.
#
# DOCUMENT_INDEX: a contents page holds the vocabulary of every topic and the
# answer to none. It scores well against many questions, occupies an evidence
# slot, and can pass the gate while carrying no usable text.
#
# Captions are NOT dropped here. They usually arrive through their parent figure
# or table, but not always, and a caption that did not is the only record of that
# figure in the text. They are kept and de-duplicated afterwards instead.
DROPPED_LABELS = frozenset(
    {
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        DocItemLabel.DOCUMENT_INDEX,
    }
)

SECTION_SEPARATOR = " > "

# A contents entry looks like "Attendance Policy ......... 18". Docling labels
# most of them DOCUMENT_INDEX, but not always every fragment, so leaders left
# behind on a page already identified as contents are dropped too.
#
# This pattern is deliberately only applied to pages Docling has already flagged.
# A requirements list such as "FR-01 ....... 12" is typographically identical to
# a contents entry, and deleting real content is far worse than keeping a
# contents page, so the pattern is never allowed to decide on its own.
DOTTED_LEADER = re.compile(r"\.\s?\.\s?\.\s?\.")

# Characters that carry no meaning in extracted text. Symbol fonts render
# bullets from the Unicode private use area, so they arrive as unreadable
# glyphs that would be embedded as noise. Control characters and the object
# replacement character are dropped for the same reason.
JUNK_CHARACTERS = re.compile("[\ue000-\uf8ff\ufffc\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]")

# True when the text holds at least one letter, in any alphabet.
HAS_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)

# A paragraph that opens with its own section title: capitals, at least two
# words, an optional aside in brackets, a separator, then prose. Anything from
# "BEFORE DAY ONE - Ensure..." to "THE FIRST SIX MONTHS (180-day check in).
# Continue..." reads the same way to a person.
#
# Deliberately strict about the capitals, because inventing a heading is worse
# than missing one. Across the sample corpus this matched six titles and nothing
# else in 178 pages.
RUN_IN_HEADING = re.compile(
    r"^([A-Z][A-Z0-9'&/\-\. ]{4,58}?)"  # the title itself, set in capitals
    r"(?:\s*\([^)]{0,40}\))?"  # an optional aside, any case
    r"\s*(?:-{1,2}|\u2013|\u2014|\.|:)\s+"  # dash, full stop or colon
    r"(?=[A-Za-z])"  # and then prose
)

# The same shape in ordinary case, marked by a colon: "Section D: Secure Hosting
# Facility Profile: Details of...". Kept to a few words so that an instruction
# ending in a colon is not mistaken for a label.
RUN_IN_LABEL = re.compile(r"^([A-Z][A-Za-z0-9'()/&.\- ]{2,58}?):\s+(?=[A-Za-z])")

# A bare value: a number on its own, optionally with a currency sign or a short
# unit. "30 pts", "15%", "$5,000.00", "12". Deliberately allows nothing else,
# because this shape is what licenses joining the element to the one before it.
BARE_VALUE = re.compile(r"^[$€£¥]?\d[\d,]*(?:\.\d+)?\s*(?:%|[A-Za-z]{1,4}\.?)?$")

# A bare label: a few words of ordinary text naming a thing, with no sentence
# punctuation and no value of its own. "Fees", "Total weight", "Annual leave:".
BARE_LABEL = re.compile(r"^[A-Za-z][A-Za-z&/\-' ]*:?$")

# Characters that are rarely what a document really says in running text, and are
# a common sign that a symbol was mis-decoded. Their presence only triggers a
# check against a second reader; it never implies a particular replacement.
GLYPH_SUSPECTS = re.compile("[\u2020\u2021\u00a4\u00bf]")


@dataclass(frozen=True)
class Element:
    """One extracted piece of a document, in reading order."""

    text: str
    page: int
    element_type: str
    section_path: str
    # One box per provenance region, as (left, top, right, bottom) in PDF points
    # with a TOP-LEFT origin, matching how PyMuPDF later draws the highlight.
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractedDocument:
    """Everything extraction learned about one PDF."""

    page_count: int
    elements: list[Element]
    table_count: int
    picture_count: int
    dropped_count: int
    heading_count: int
    seconds: float
    # Pages Docling identified as a table of contents. Recorded so a report can
    # say a page is empty *because it was excluded*, rather than leaving a reader
    # to guess whether the pipeline lost it.
    contents_pages: frozenset[int] = frozenset()

    @property
    def char_count(self) -> int:
        return sum(len(element.text) for element in self.elements)

    @property
    def chars_per_page(self) -> int:
        """Average characters per page. Drives the conditional OCR decision."""
        if self.page_count == 0:
            return 0
        return self.char_count // self.page_count

    @property
    def needs_ocr(self) -> bool:
        """True when the text layer is too thin to be real text."""
        return self.chars_per_page < settings.OCR_TRIGGER_CHARS_PER_PAGE


@lru_cache(maxsize=2)
def _converter(with_ocr: bool = False) -> DocumentConverter:
    """Build a Docling converter once per mode.

    Construction loads layout and table models, which costs seconds and a few
    hundred megabytes, so each is shared across documents for the process
    lifetime. Two are cached rather than one: a document that needs OCR is read
    twice, once each way, and rebuilding the models for the second pass would
    double an already slow path.

    OCR is off in the ordinary converter. Running it on a PDF that already has a
    text layer roughly triples the time and replaces exact text with a guess.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = with_ocr
    pipeline_options.do_table_structure = settings.DOCLING_DETECT_TABLES
    pipeline_options.table_structure_options.do_cell_matching = settings.DOCLING_MATCH_TABLE_CELLS
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice(settings.DOCLING_ACCELERATOR.lower())
    )

    if with_ocr:
        # Only the parts of a page that carry no text are read by the engine.
        # `FULL_PAGE` would discard the exact text on a document that is mostly
        # digital and only partly scanned, replacing it with a guess.
        pipeline_options.ocr_options = RapidOcrOptions(mode=OcrMode.PDF_AWARE_LAYOUT_REGIONS)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def _bboxes_top_left(
    item, doc: DoclingDocument, only_page: int | None = None
) -> list[tuple[float, float, float, float]]:
    """Return the item's boxes converted to a top-left origin.

    Docling reports coordinates from the bottom-left, the PDF convention.
    PyMuPDF draws from the top-left, the screen convention. Converting once
    here means no renderer downstream has to remember which way up it is.

    A paragraph that runs across a page break has a box on each page. Passing
    `only_page` keeps just that page's boxes, because a highlight drawn from
    another page's coordinates would land on the wrong part of the wrong page.
    """
    boxes: list[tuple[float, float, float, float]] = []
    for prov in item.prov or []:
        if only_page is not None and prov.page_no != only_page:
            continue
        page = doc.pages.get(prov.page_no)
        if page is None or prov.bbox is None:
            continue
        bbox = prov.bbox.to_top_left_origin(page.size.height)
        boxes.append((bbox.l, bbox.t, bbox.r, bbox.b))
    return boxes


def _heading_height(item) -> float | None:
    """Height of a heading's text in PDF points, a proxy for its font size."""
    if not item.prov or item.prov[0].bbox is None:
        return None
    bbox = item.prov[0].bbox
    return abs(bbox.t - bbox.b)


def _run_in_heading(text: str) -> tuple[str, int] | None:
    """The title a paragraph opens with, and how deep it sits, if it has one.

    Some documents put a title and its explanation in one block: "BEFORE DAY ONE
    - Ensure everything is in place...". The title then never becomes a heading,
    so the pages beneath inherit whatever came before.

    Capitals introduce a section. Ordinary case with a colon is a label, and sits
    at ordinary heading depth so it is a sibling rather than a parent.
    """
    flat = re.sub(r"\s+", " ", text)

    match = RUN_IN_HEADING.match(flat)
    if match:
        title = match.group(1).strip()
        body = flat[match.end() :]
        if (
            len(title.split()) >= 2
            and title.upper() == title
            and len(body) >= settings.RUN_IN_HEADING_MIN_BODY_CHARS
        ):
            return title, settings.RUN_IN_HEADING_LEVEL

    match = RUN_IN_LABEL.match(flat)
    if match:
        title = match.group(1).strip()
        body = flat[match.end() :]
        words = len(title.split())
        if (
            2 <= words <= settings.RUN_IN_LABEL_MAX_WORDS
            and len(body) >= settings.RUN_IN_HEADING_MIN_BODY_CHARS
        ):
            return title, settings.RUN_IN_LABEL_LEVEL

    return None


def _is_really_a_heading(text: str) -> bool:
    """Whether text classified as a heading actually reads like one.

    Layout models label a lead-in sentence such as "The vendor must clearly
    identify all third-party components and describe:" as a heading. Accepting
    that has two costs: the sentence stops being indexed as content, because
    headings live in the section path rather than in the text, and every list
    item beneath it is filed under a section that does not exist.

    Short labels ending in a colon, such as "PERFORMANCE BONDS:", are genuine
    headings and stay.
    """
    if len(text) > settings.HEADING_MAX_CHARS:
        return False
    if not text.endswith((".", ":", ";")):
        return True
    # Either measure is enough. "The vendor must describe in detail:" is only 35
    # characters but six words, and it introduces a list rather than naming a
    # section.
    return (
        len(text) <= settings.HEADING_SENTENCE_CHARS
        and len(text.split()) < settings.HEADING_SENTENCE_WORDS
    )


def _heading_levels(doc: DoclingDocument) -> dict[str, int]:
    """Work out a depth for every heading, keyed by its element reference.

    Docling reports level 1 for every heading, so depth comes from the height of
    the heading text against this document's most common heading height. A
    document whose headings are all one size gets a flat path, which is honest:
    if the PDF does not distinguish parent from child, nothing can.
    """
    measured: dict[str, tuple[float, int, bool]] = {}
    for item, _level in doc.iterate_items():
        if not isinstance(item, SectionHeaderItem | TitleItem):
            continue
        height = _heading_height(item)
        if height is not None:
            text = item.text.strip()
            measured[item.self_ref] = (height, len(text), text.endswith(":"))

    if not measured:
        return {}

    # The modal height is "an ordinary heading" in this document. Rounding to
    # whole points groups headings that differ only by a rendering fraction.
    modal = Counter(round(height) for height, _length, _label in measured.values()).most_common(1)[
        0
    ][0]

    levels: dict[str, int] = {}
    for ref, (height, length, is_label) in measured.items():
        delta = height - modal
        can_be_parent = length <= settings.HEADING_MAX_PARENT_CHARS
        if can_be_parent and delta >= settings.HEADING_LEVEL_1_DELTA:
            level = 1
        elif can_be_parent and delta >= settings.HEADING_LEVEL_2_DELTA:
            level = 2
        elif delta >= settings.HEADING_LEVEL_3_DELTA:
            level = 3
        else:
            level = 4
        # A label ending in a colon belongs under the heading above it, even at
        # the same size. Left as siblings they replace one another and the parent
        # is lost from every path beneath it.
        if is_label and settings.HEADING_DEMOTE_TRAILING_COLON:
            level = min(4, level + 1)
        levels[ref] = level
    return levels


def _clean(text: str) -> str:
    """Remove characters that carry no meaning, and tidy the whitespace."""
    text = JUNK_CHARACTERS.sub(" ", text)
    # Collapse runs of spaces and tabs but keep line breaks, which carry table
    # and list structure.
    text = re.sub(r"[ \t]+", " ", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _page_texts(pdf_path: Path) -> dict[int, str]:
    """Plain text per page, read with PyMuPDF, for checking suspect characters."""
    try:
        with pymupdf.open(pdf_path) as pdf:
            return {number: page.get_text() for number, page in enumerate(pdf, start=1)}
    except Exception:  # noqa: BLE001 - repair is optional, extraction is not
        return {}


def _repair_glyphs(text: str, page_text: str) -> str:
    """Restore a character that was mis-decoded, using a second reader.

    A threshold written "≥ 4 hours before arrival" can arrive as "‡ 4 hours
    before arrival", which changes a requirement rather than merely looking odd.

    Nothing is assumed about what the character ought to be. The passage is
    located in the page text as PyMuPDF read it, and whatever character sits
    there is used. If both readers agree, or the passage cannot be located
    exactly once, the text is left as it is - a document that really does use a
    dagger keeps it.
    """
    if not page_text or not GLYPH_SUSPECTS.search(text):
        return text

    def flatten(value: str) -> str:
        # Table markup is ours, not the document's, so it is removed from both
        # sides before comparing. Otherwise a cell's pipes stop the passage from
        # ever matching the plain text of the page.
        return re.sub(r"\s+", " ", value.replace("|", " ")).strip()

    reference = flatten(page_text)
    repaired = text
    for match in GLYPH_SUSPECTS.finditer(text):
        following = flatten(text[match.end() :])
        words = [word for word in following.split(" ") if word.strip("-")][
            : settings.GLYPH_REPAIR_CONTEXT_WORDS
        ]
        if len(words) < 2:
            continue
        found = [m.start() for m in re.finditer(re.escape(" ".join(words)), reference)]
        if not found:
            continue

        # The passage may occur several times on the page - the same threshold
        # repeated down a column of a table. That is only a problem if the
        # occurrences disagree about the character. Where they all report the
        # same one, which occurrence this text came from does not matter.
        candidates = {reference[:start].rstrip()[-1:] for start in found}
        candidates.discard("")
        if len(candidates) != 1:
            continue

        actual = candidates.pop()
        # Only a symbol is accepted. Landing on a letter or a digit means the
        # passage was not lined up, not that the character is a symbol.
        if actual == match.group() or actual.isalnum():
            continue
        repaired = repaired[: match.start()] + actual + repaired[match.start() + 1 :]
    return repaired


def _repair_spacing(text: str, page_text: str) -> str:
    """Restore spacing inside a word, using the second reader.

    Layout analysis sometimes breaks a word: "Checklist" arrives as "Checklis t",
    which no longer matches a search for the word it is. It can also close a gap
    that belonged there.

    The two readers must agree on every character for a repair to happen - only
    the spaces may differ. That makes the change safe: nothing is added, removed
    or guessed, and the passage is rewritten only when it is the same passage.
    """
    if not page_text or "|" in text or len(text) < 8:
        return text

    squeezed_text = re.sub(r"\s+", "", text)
    if not squeezed_text:
        return text

    # Index every non-space character of the page so a match can be mapped back
    # to the original text, spacing included.
    positions = [index for index, char in enumerate(page_text) if not char.isspace()]
    squeezed_page = "".join(page_text[index] for index in positions)

    at = squeezed_page.find(squeezed_text)
    if at < 0 or squeezed_page.find(squeezed_text, at + 1) >= 0:
        # Not found, or found more than once and so not identifiable.
        return text

    start = positions[at]
    end = positions[at + len(squeezed_text) - 1] + 1
    return _clean(page_text[start:end])


def _contents_pages(doc: DoclingDocument) -> set[int]:
    """Pages Docling identified as holding a table of contents.

    Used to scope the dotted-leader rule. Any page already labelled by Docling is
    a page where an entry-shaped fragment is safe to drop; anywhere else, the
    same shape might be a numbered requirement and must be kept.
    """
    pages: set[int] = set()
    headings: dict[str, set[int]] = {}
    per_page: dict[int, list[str]] = {}

    for item, _level in doc.iterate_items(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    ):
        if not item.prov:
            continue
        page = item.prov[0].page_no
        if getattr(item, "label", None) is DocItemLabel.DOCUMENT_INDEX:
            pages.add(page)
        if isinstance(item, SectionHeaderItem | TitleItem):
            headings.setdefault(_index_key(item.text), set()).add(page)
        elif isinstance(item, TextItem):
            per_page.setdefault(page, []).append(_index_key(item.text))

    # Where the label is missing, a contents page still gives itself away: it is
    # a list of the document's own headings and nothing else. Matching against
    # headings the document actually has means a page of ordinary prose cannot
    # qualify, however it is typeset.
    page_count = len(doc.pages) or 1
    cutoff = max(2, round(page_count * settings.CONTENTS_MAX_PAGE_RATIO))
    for page, entries in per_page.items():
        if page in pages or page > cutoff or len(entries) < settings.CONTENTS_MIN_ENTRIES:
            continue
        # An entry counts only when the matching heading is on some other page,
        # so a page cannot be judged a contents page by its own headings.
        matches = sum(
            1
            for entry in entries
            if entry and any(other != page for other in headings.get(entry, ()))
        )
        if matches / len(entries) >= settings.CONTENTS_HEADING_MATCH_RATIO:
            pages.add(page)
    return pages


def _index_key(text: str) -> str:
    """Reduce a line to what a contents entry and its heading have in common.

    A contents page lists "About This Manual" while the heading itself reads
    "1. About This Manual", and the entry may trail a page number. Stripping the
    numbering from both ends lets the two be compared without guessing at
    typography.
    """
    text = _clean(text).casefold()
    text = re.sub(r"^[\divxlc]+[.)\s-]+", "", text)
    text = re.sub(r"[.\s]*\d*$", "", text)
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def _repeated_headings(doc: DoclingDocument) -> set[str]:
    """Heading text that recurs across the document, which makes it furniture.

    A running title set in heading type appears on page after page. Left on the
    stack it becomes the parent of every section beneath it, so a citation reads
    "New Employee Onboarding Supervisor Guide > 3-Month Review > ..." where the
    first part says nothing. A real heading does not repeat on a quarter of the
    document's pages.
    """
    seen: dict[str, list[tuple[int, float]]] = {}
    for item, _level in doc.iterate_items():
        if isinstance(item, SectionHeaderItem | TitleItem) and item.prov:
            seen.setdefault(_clean(item.text), []).append(_position_of(item, doc))

    page_count = len(doc.pages) or 1
    limit = max(3, round(page_count * settings.REPEATED_HEADING_PAGE_RATIO))

    furniture: set[str] = set()
    for text, spots in seen.items():
        if len({page for page, _top in spots}) < limit:
            continue
        # Printed at the same height on every page means it is part of the page
        # frame. A heading that moves down the page is a real heading each time,
        # however often it recurs.
        tops = [top for _page, top in spots]
        if max(tops) - min(tops) <= settings.REPEATED_HEADING_POSITION_SPREAD:
            furniture.add(text)
    return furniture


def _is_indexable(text: str, page: int, contents_pages: set[int]) -> bool:
    """Whether this text is worth storing as a chunk.

    Two ways to fail. Text with no letters at all - a stray page number, a run of
    underscores from a form - can never answer a question. And on a contents page,
    a leftover entry with dot leaders is part of the contents rather than content.
    """
    if not HAS_LETTER.search(text):
        return False
    return not (page in contents_pages and DOTTED_LEADER.search(text))


def _page_of(item) -> int | None:
    """The page an item sits on, or None when it has no provenance."""
    if not item.prov:
        return None
    return item.prov[0].page_no


def _position_of(item, doc: DoclingDocument) -> tuple[int, float]:
    """Where an item sits in the document, as (page, distance down the page).

    Only boxes on the item's own page are considered. A paragraph continuing
    across a page break also has a box near the top of the next page, and
    measuring from that would place the paragraph at the top of the page it
    started on - above the heading it actually sits beneath.
    """
    page = _page_of(item) or 0
    boxes = _bboxes_top_left(item, doc, only_page=page)
    if not boxes:
        return (page, 0.0)
    return (page, min(box[1] for box in boxes))


def _heading_stacks(
    doc: DoclingDocument,
    levels: dict[str, int],
    contents_pages: set[int],
    repeated: set[str],
    picture_pages: set[int],
) -> tuple[list[tuple[int, float]], list[dict[int, str]], list[tuple]]:
    """Build the heading stack as it stands at each heading, in visual order.

    Section paths are resolved by position rather than by the order Docling
    emitted things, because Docling sometimes emits a page's main heading after
    the paragraphs beneath it. Walking in emission order would then hand those
    paragraphs the previous page's heading, which is worse than no heading at
    all: a citation would name a section the text does not belong to.

    Only the label is decided here. Element order is left exactly as Docling
    produced it, so a mistake in this function can never scramble reading order.
    """
    headings: list[tuple[tuple[int, float], int, str, list]] = []
    for item, _level in doc.iterate_items():
        is_heading = isinstance(item, SectionHeaderItem | TitleItem)
        # A title that opens its own paragraph is registered as a heading too,
        # so that it labels the pages beneath it. The paragraph is left alone.
        run_in = (
            _run_in_heading(_clean(item.text))
            if isinstance(item, TextItem) and not is_heading and item.prov
            else None
        )
        if run_in:
            title, level = run_in
            headings.append(
                (
                    _position_of(item, doc),
                    level,
                    title,
                    _bboxes_top_left(item, doc, only_page=_page_of(item)),
                )
            )
            continue
        # A caption on a page with no picture is not describing an image. In the
        # documents tested these are note and section headings, so they are
        # allowed to label the text beneath them like any other heading.
        is_misfiled_heading = (
            getattr(item, "label", None) is DocItemLabel.CAPTION
            and bool(item.prov)
            and item.prov[0].page_no not in picture_pages
        )
        if not (is_heading or is_misfiled_heading):
            continue
        text = _clean(item.text)
        position = _position_of(item, doc)
        # A "Table of Contents" heading would otherwise stay on the stack and
        # label the following pages, which belong to a real section instead.
        if (
            not text
            or position[0] in contents_pages
            or text in repeated
            or not _is_really_a_heading(text)
        ):
            continue
        headings.append((position, levels.get(item.self_ref, 3), text, _bboxes_top_left(item, doc)))

    headings.sort(key=lambda entry: entry[0])

    keys: list[tuple[int, float]] = []
    stacks: list[dict[int, str]] = []
    records: list[tuple] = []
    stack: dict[int, str] = {}
    for position, level, text, bboxes in headings:
        # A heading replaces anything at its level or deeper, and keeps what
        # sits above it.
        stack = {lvl: existing for lvl, existing in stack.items() if lvl < level}
        # The path recorded for the heading itself is its ancestors only, so
        # that a rescued heading is not filed underneath its own text.
        parent = SECTION_SEPARATOR.join(
            stack[lvl] for lvl in sorted(stack)[-settings.HEADING_MAX_DEPTH :]
        )
        stack[level] = text
        keys.append(position)
        stacks.append(dict(stack))
        records.append((position, text, bboxes, parent))
    return keys, stacks, records


def _section_path_at(
    position: tuple[int, float],
    keys: list[tuple[int, float]],
    stacks: list[dict[int, str]],
) -> str:
    """The section path in force at a position: the last heading above it."""
    index = bisect_right(keys, position) - 1
    if index < 0:
        return ""
    stack = stacks[index]
    # Keep the deepest headings: they are the most specific description of where
    # this text sits. Distant ancestors add words without adding meaning.
    levels = sorted(stack)[-settings.HEADING_MAX_DEPTH :]
    return SECTION_SEPARATOR.join(stack[level] for level in levels)


def _without_duplicate_captions(elements: list[Element]) -> list[Element]:
    """Drop a caption that its figure or table already carries.

    A caption is kept when it stands alone, because then it is the only record
    of that figure in the text. When its parent already includes the same
    sentence, keeping both would embed it twice and put two citations on the
    same passage.
    """

    def key(value: str) -> str:
        # Compared without spacing or case, because the two copies are not
        # always byte-identical: a caption standing on its own can have its
        # spacing repaired against the page text, while the copy bound into a
        # table cannot. Raw containment then misses the duplicate.
        return re.sub(r"\s+", "", value).casefold()

    kept: list[Element] = []
    seen: set[tuple[int, str]] = set()
    for element in elements:
        if element.element_type == TYPE_FIGURE_CAPTION:
            caption = key(element.text)
            # Keep the first copy and drop later ones. Asking only "is this
            # contained in something else" would delete both, since two
            # identical strings each contain the other.
            if (element.page, caption) in seen:
                continue
            swallowed = any(
                other.page == element.page
                and len(other.text) > len(element.text)
                and caption in key(other.text)
                for other in elements
            )
            if swallowed:
                continue
            seen.add((element.page, caption))
        kept.append(element)
    return kept


def _with_orphan_headings(
    elements: list[Element],
    headings: list[tuple[tuple[int, float], str, list[tuple[float, float, float, float]], str]],
) -> list[Element]:
    """Keep a heading that has no text beneath it, as text in its own right.

    A heading normally survives inside the section path of the text below it.
    When two headings follow one another with nothing in between - a title above
    a subtitle, a label above an address - the first has no text to attach to and
    would disappear from the document altogether.

    That is real content loss: on one sample it removed the document's own
    reference number, "Request for Proposal #26-004", which is exactly the kind
    of thing somebody would search for.
    """
    occupied = {
        (element.page, round(min(b[1] for b in element.bboxes), 1))
        for element in elements
        if element.bboxes
    }
    positions = sorted(
        {
            (element.page, min(b[1] for b in element.bboxes))
            for element in elements
            if element.bboxes
        }
    )

    rescued: list[Element] = []
    for index, (position, text, bboxes, parent_path) in enumerate(headings):
        following = headings[index + 1][0] if index + 1 < len(headings) else (10**6, 0.0)
        has_body = any(position < spot < following for spot in positions)
        if has_body or (position[0], round(position[1], 1)) in occupied:
            continue
        rescued.append(
            Element(
                text=text,
                page=position[0],
                element_type=TYPE_TEXT,
                section_path=parent_path,
                bboxes=bboxes,
            )
        )
    return elements + rescued


def _looks_like_prose_columns(page_elements: list[Element]) -> bool:
    """Whether a page is two columns of running prose rather than a grid.

    Both look like two columns of text. The difference is alignment: a grid puts
    a label beside its value on the same line, so items pair up across the gap,
    while two columns of an article flow independently and rarely line up.

    Reading a grid row by row is correct and reading an article row by row is
    nonsense, so this decides which pages may be re-ordered at all.
    """
    boxes = [
        (
            min(b[0] for b in e.bboxes),
            max(b[2] for b in e.bboxes),
            max(b[3] for b in e.bboxes) - min(b[1] for b in e.bboxes),
        )
        for e in page_elements
        if e.bboxes
    ]
    if len(boxes) < 4:
        return False

    content_width = max(right for _left, right, _height in boxes) - min(
        left for left, _right, _height in boxes
    )
    if content_width <= 0:
        return False

    # Is there a clear vertical channel splitting the page into two columns?
    ordered = sorted(boxes, key=lambda box: (box[0] + box[1]) / 2)
    centres = [(left + right) / 2 for left, right, _height in ordered]
    gap, split = max(
        ((centres[i + 1] - centres[i], i) for i in range(len(centres) - 1)),
        default=(0.0, 0),
    )
    if gap < content_width * settings.COLUMN_GAP_RATIO:
        return False

    left_side, right_side = ordered[: split + 1], ordered[split + 1 :]
    if len(left_side) < 2 or len(right_side) < 2:
        return False

    # Columns stand side by side without overlapping. A page of full-width
    # paragraphs and tables can be split at its widest gap into two groups of
    # similar width, but those groups sit on top of one another rather than
    # beside one another, and treating that page as columns would leave its
    # elements in whatever order they arrived.
    if max(right for _left, right, _height in left_side) > min(
        left for left, _right, _height in right_side
    ):
        return False

    def span(side: list[tuple[float, float, float]]) -> float:
        return max(right for _left, right, _height in side) - min(
            left for left, _right, _height in side
        )

    # Two columns of an article are set to the same width. A grid pairing a
    # short label with a long value is lopsided.
    widths = sorted((span(left_side), span(right_side)))
    if widths[1] > 0 and widths[0] / widths[1] >= settings.COLUMN_WIDTH_SIMILARITY:
        return True

    # Failing that, prose gives itself away by being built from multi-line
    # blocks, where a grid cell holds a single line.
    multiline = sum(
        1 for _left, _right, height in boxes if height > settings.MULTILINE_HEIGHT_POINTS
    )
    return multiline / len(boxes) >= settings.PROSE_COLUMN_RATIO


def _in_reading_order(elements: list[Element]) -> list[Element]:
    """Put each page's elements into visual order, where that is safe to do.

    Docling sometimes emits an element far from where it sits on the page, which
    separates a date from the event beside it or moves a table away from the
    text it belongs to. Sorting by position repairs that.

    Pages that look like two columns of prose are left exactly as Docling
    produced them, because reading those row by row would interleave the columns.
    Reading order is the one thing that must never be scrambled, so where there
    is any doubt the original order stands.
    """
    by_page: dict[int, list[Element]] = {}
    for element in elements:
        by_page.setdefault(element.page, []).append(element)

    ordered: list[Element] = []
    for page in sorted(by_page):
        group = by_page[page]
        if _looks_like_prose_columns(group):
            ordered.extend(group)
            continue

        def key(item: tuple[int, Element]) -> tuple[float, float, int]:
            index, element = item
            if not element.bboxes:
                # Nothing to place it by, so it keeps its position in the stream.
                return (float("inf"), 0.0, index)
            top = min(box[1] for box in element.bboxes)
            left = min(box[0] for box in element.bboxes)
            # Banding the top means two cells set fractionally apart still read
            # left to right rather than one above the other.
            return (round(top / settings.ROW_BAND), left, index)

        ordered.extend(element for _index, element in sorted(enumerate(group), key=key))
    return ordered


def _envelope(element: Element) -> tuple[float, float, float, float] | None:
    """The one box that contains all of an element's boxes."""
    if not element.bboxes:
        return None
    return (
        min(box[0] for box in element.bboxes),
        min(box[1] for box in element.bboxes),
        max(box[2] for box in element.bboxes),
        max(box[3] for box in element.bboxes),
    )


def _is_pair(label: Element, value: Element) -> bool:
    """Whether these two elements are one label-and-value fact split in two.

    Both the wording and the geometry have to agree. Wording alone would join a
    heading to the first number under it, anywhere on the page; geometry alone
    would join any two things that happen to sit close together.
    """
    if label.element_type != TYPE_TEXT or value.element_type != TYPE_TEXT:
        return False
    if label.page != value.page:
        return False

    text = label.text.strip()
    if not BARE_LABEL.match(text) or not 1 <= len(text.split()) <= settings.LABEL_MAX_WORDS:
        return False
    if not BARE_VALUE.match(value.text.strip()):
        return False

    first, second = _envelope(label), _envelope(value)
    if first is None or second is None:
        return False

    # Beneath its label, or beside it. Which one depends on whether the boxes
    # share a horizontal or a vertical span; anything diagonal is not a pair.
    if first[0] < second[2] and second[0] < first[2]:
        gap = second[1] - first[3]
    elif first[1] < second[3] and second[1] < first[3]:
        gap = second[0] - first[2]
    else:
        return False
    return 0 <= gap <= settings.LABEL_VALUE_MAX_GAP_POINTS


def _with_paired_values(elements: list[Element]) -> list[Element]:
    """Rejoin a value to the label it belongs to.

    Run after reading order is settled, so that "next to" means next on the page
    rather than next in the order Docling happened to emit.

    Nothing is added or removed: the two texts are joined with a colon, and the
    boxes of both are kept so the citation highlights the whole fact.
    """
    joined: list[Element] = []
    index = 0
    while index < len(elements):
        current = elements[index]
        following = elements[index + 1] if index + 1 < len(elements) else None
        if following is not None and _is_pair(current, following):
            joined.append(
                Element(
                    text=f"{current.text.rstrip(':')}: {following.text}",
                    page=current.page,
                    element_type=TYPE_TEXT,
                    section_path=current.section_path,
                    bboxes=current.bboxes + following.bboxes,
                )
            )
            index += 2
            continue
        joined.append(current)
        index += 1
    return joined


def extract(pdf_path: Path, with_ocr: bool = False) -> ExtractedDocument:
    """Extract one PDF into ordered elements with full provenance.

    Raises:
        ExtractionFailedError: Docling could not process the file.
        EmptyDocumentError: the file has no pages.

    An empty element list is not an error. A scanned PDF has pages but no text
    layer, and it is the OCR stage that decides what to do about that.
    """
    started = time.perf_counter()
    try:
        result = _converter(with_ocr).convert(pdf_path)
    except Exception as exc:  # noqa: BLE001 - any Docling failure is one outcome
        raise ExtractionFailedError(f"{type(exc).__name__}: {exc}") from exc
    seconds = time.perf_counter() - started

    doc = result.document
    page_count = len(doc.pages)
    if page_count == 0:
        raise EmptyDocumentError("the PDF reports zero pages")

    elements: list[Element] = []
    contents_pages = _contents_pages(doc)
    page_texts = _page_texts(pdf_path)
    picture_pages = {
        item.prov[0].page_no
        for item, _level in doc.iterate_items()
        if isinstance(item, PictureItem) and item.prov
    }
    heading_keys, heading_stacks, heading_records = _heading_stacks(
        doc, _heading_levels(doc), contents_pages, _repeated_headings(doc), picture_pages
    )
    headings_seen = len(heading_keys)
    pictures = 0

    # Running headers and footers sit in Docling's FURNITURE layer, which
    # iterate_items excludes by default. They are counted separately so the
    # profiler can report what was removed rather than leaving it invisible.
    dropped = sum(
        1 for _item, _level in doc.iterate_items(included_content_layers={ContentLayer.FURNITURE})
    )

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        if label in DROPPED_LABELS:
            dropped += 1
            continue

        page = _page_of(item)
        if page is None:
            continue

        if isinstance(item, SectionHeaderItem | TitleItem):
            # Headings are structure, not content: they label the text beneath
            # them through the section path. A heading emitted as its own
            # element would answer nothing while scoring well and occupying an
            # evidence slot.
            #
            # Unless it is not really a heading, in which case it is a sentence
            # that would otherwise be lost entirely.
            text = _clean(item.text)
            if not text or _is_really_a_heading(text):
                continue
            element_type = TYPE_TEXT
        elif label is DocItemLabel.CAPTION:
            # Captions normally arrive through their figure or table. One that
            # did not is the only record of that figure, so it is kept rather
            # than dropped; duplicates are removed afterwards.
            #
            # It is only labelled a figure caption when the page actually holds a
            # picture. Layout models also label a note heading as a caption, and
            # calling that a figure would tell the user to look at an image that
            # is not there.
            text = _clean(item.text)
            element_type = TYPE_FIGURE_CAPTION if page in picture_pages else TYPE_TEXT
        elif isinstance(item, TableItem):
            # A table is kept whole. Half a table means nothing.
            text = item.export_to_markdown(doc).strip()
            caption = item.caption_text(doc).strip()
            # The markdown export sometimes already carries the caption. Adding
            # it unconditionally would repeat the whole sentence twice in the
            # chunk, wasting context and embedding the same words twice.
            if caption and caption not in text:
                text = f"{caption}\n\n{text}"
            element_type = TYPE_TABLE
        elif isinstance(item, PictureItem):
            pictures += 1
            # The image itself is never read. Its caption is indexed so the
            # figure is findable, and the citation is labelled as a figure so
            # the user knows to look at the page rather than trust the text.
            text = item.caption_text(doc).strip()
            element_type = TYPE_FIGURE_CAPTION
        elif isinstance(item, TextItem):
            text = item.text.strip()
            element_type = TYPE_TEXT
        else:
            continue

        reference = page_texts.get(page, "")
        text = _repair_spacing(_repair_glyphs(_clean(text), reference), reference)
        if not _is_indexable(text, page, contents_pages):
            dropped += 1
            continue

        elements.append(
            Element(
                text=text,
                page=page,
                element_type=element_type,
                section_path=_section_path_at(
                    _position_of(item, doc), heading_keys, heading_stacks
                ),
                bboxes=_bboxes_top_left(item, doc, only_page=page),
            )
        )

    elements = _without_duplicate_captions(elements)
    elements = _with_orphan_headings(elements, heading_records)
    elements = _in_reading_order(elements)
    elements = _with_paired_values(elements)

    return ExtractedDocument(
        page_count=page_count,
        elements=elements,
        table_count=sum(1 for e in elements if e.element_type == TYPE_TABLE),
        picture_count=pictures,
        dropped_count=dropped,
        heading_count=headings_seen,
        seconds=seconds,
        contents_pages=frozenset(contents_pages),
    )
