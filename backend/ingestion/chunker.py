"""Turn extracted elements into chunks ready to embed.

Structure first, size second.

Structure first because a chunk that lines up with a real section has one
subject, one useful citation, and a heading that describes it. Size second
because a section is often far larger than a useful chunk, and a vector that
averages several topics is close to nothing in particular.

Nothing here knows anything about any particular document. Chunks are built
from the section paths and element types the extractor produced, so an unseen
PDF chunks the same way with no code change.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

from backend.ingestion.extractor import TYPE_TEXT, Element, ExtractedDocument
from config import settings

# A token counter takes text and returns how many tokens the embedding model
# would see. Injected so tests run offline and instantly, and so the real
# tokenizer is loaded once rather than per chunk.
TokenCounter = Callable[[str], int]

# Paragraph, then sentence, then word: the order in which an over-long element
# is broken up, weakest boundary last. Splitting mid-word is never allowed.
PARAGRAPH_BREAK = re.compile(r"\n\s*\n|\n")
SENTENCE_BREAK = re.compile(r"(?<=[.!?:;])\s+")
WORD_BREAK = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    """One unit of retrieval.

    `text` is the body alone, because that is what a user is shown when they
    open the source of an answer, and a header they did not write would read as
    if the document contained it.

    `embed_text` is what actually gets embedded: the context header followed by
    the body. The header is cheap and does real work, because a question asked
    in a heading's words then matches a chunk whose body uses different words.
    """

    index: int
    text: str
    page: int
    section_path: str
    element_type: str
    token_count: int
    # Boxes for the highlight. All of them are on `page`, because a chunk never
    # holds text from more than one page: a citation resolves to one page and one
    # set of coordinates on it, so text from the next page filed under this
    # chunk would open a page that text is not on.
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    # Set once the document has a name. Held on the chunk rather than rebuilt
    # later so that what was embedded is exactly what is stored.
    context_header: str = ""

    @property
    def embed_text(self) -> str:
        return f"{self.context_header}\n{self.text}" if self.context_header else self.text


@lru_cache(maxsize=1)
def _encoding():
    """The tokenizer the embedding model itself uses.

    Loaded lazily and once. Measuring chunk size with any other tokenizer
    produces chunks that are the wrong size in the only unit that counts.
    """
    import tiktoken

    return tiktoken.encoding_for_model(settings.EMBEDDING_MODEL)


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def context_header(title: str, section_path: str, page: int) -> str:
    """`[Handbook > 4.2 Parental Leave > p.17]`

    Whatever is known is included and whatever is missing is left out, so a
    document with no detected headings still gets a title and a page rather
    than an empty bracket or a gap between separators.
    """
    parts = [part for part in (title, section_path) if part]
    parts.append(f"{settings.CONTEXT_PAGE_PREFIX}{page}")
    return "[" + settings.CONTEXT_SEPARATOR.join(parts) + "]"


def _groups(elements: list[Element]) -> list[list[Element]]:
    """Consecutive elements sharing a section path.

    Consecutive on purpose. The same heading can legitimately recur far apart
    in a document - a role name under each phase of a guide, for instance - and
    joining those into one chunk would place unrelated text under one citation.
    """
    grouped: list[list[Element]] = []
    for element in elements:
        if grouped and grouped[-1][0].section_path == element.section_path:
            grouped[-1].append(element)
        else:
            grouped.append([element])
    return grouped


def _body_budget() -> int:
    """How much new text a chunk may hold, once room is left for the overlap.

    The target is the size of the whole chunk, not of its new content, so the
    repeated tail has to be paid for out of the same budget. Without this every
    chunk after the first comes out one overlap over target.
    """
    return max(1, settings.CHUNK_TARGET_TOKENS - settings.CHUNK_OVERLAP_TOKENS)


def _split_oversized(text: str, counter: TokenCounter) -> list[str]:
    """Break text that exceeds the hard ceiling at the strongest boundary available.

    Paragraphs first, then sentences, then words. A word is never split: half a
    token sequence is not text, and a broken word embeds as neither of the two
    words it came from.
    """
    if counter(text) <= settings.CHUNK_MAX_TOKENS:
        return [text]

    budget = _body_budget()

    for pattern in (PARAGRAPH_BREAK, SENTENCE_BREAK, WORD_BREAK):
        pieces = [piece for piece in pattern.split(text) if piece.strip()]
        if len(pieces) < 2:
            continue

        out: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece
            if current and counter(candidate) > budget:
                out.append(current)
                current = piece
            else:
                current = candidate
        if current:
            out.append(current)

        # Words are the last resort, so accept whatever it produced. Otherwise
        # only accept a split that actually got everything under the ceiling,
        # and fall through to a weaker boundary if it did not.
        if pattern is WORD_BREAK or all(counter(part) <= settings.CHUNK_MAX_TOKENS for part in out):
            return out

    return [text]


def _tail(text: str, counter: TokenCounter) -> str:
    """The last CHUNK_OVERLAP_TOKENS or so of text, on a sentence boundary if possible.

    Overlap exists so an answer sitting across a boundary is whole in at least
    one chunk. Taking it at a sentence boundary keeps the repeated fragment
    readable, which matters because it is shown to a user as part of a source.
    """
    if counter(text) <= settings.CHUNK_OVERLAP_TOKENS:
        return text

    sentences = [piece for piece in SENTENCE_BREAK.split(text) if piece.strip()]
    kept: list[str] = []
    for sentence in reversed(sentences):
        candidate = " ".join([sentence, *kept])
        if kept and counter(candidate) > settings.CHUNK_OVERLAP_TOKENS:
            break
        kept.insert(0, sentence)
    if kept:
        return " ".join(kept)

    # One sentence longer than the overlap budget: fall back to whole words.
    words = WORD_BREAK.split(text)
    while words and counter(" ".join(words)) > settings.CHUNK_OVERLAP_TOKENS:
        words.pop(0)
    return " ".join(words)


@dataclass
class _Pending:
    """A chunk being accumulated, before it is frozen into a Chunk."""

    texts: list[str]
    page: int
    section_path: str
    element_type: str
    bboxes: list[tuple[float, float, float, float]]

    @property
    def text(self) -> str:
        return "\n".join(self.texts)


def _is_separator_row(line: str) -> bool:
    """A markdown table's rule line, `|---|:---:|`. Structure, not data."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= set("|-: ")


def _split_unembeddable(text: str, counter: TokenCounter) -> list[str]:
    """Split an atomic element that is too large for the embedding model to accept.

    Tables are never split for size convenience: half a table is a column
    heading with no data, or data with no heading. This is the one exception,
    and it is not a size preference - above the model's hard input ceiling the
    chunk cannot be embedded at all, so the choice is between a split table and
    no table in the index.

    The split is at row boundaries and the header rows are repeated into every
    part, so each part still says what its columns mean. A single row larger
    than the ceiling is left intact: breaking inside a row would produce values
    with no column, which is worse than a chunk that fails to embed.
    """
    if counter(text) <= settings.EMBED_MAX_INPUT_TOKENS:
        return [text]

    lines = text.split("\n")
    header = lines[:2] if len(lines) > 2 and _is_separator_row(lines[1]) else []
    body = lines[len(header) :]

    parts: list[str] = []
    current = list(header)
    for line in body:
        candidate = "\n".join([*current, line])
        if len(current) > len(header) and counter(candidate) > settings.EMBED_MAX_INPUT_TOKENS:
            parts.append("\n".join(current))
            current = [*header, line]
        else:
            current.append(line)
    if len(current) > len(header):
        parts.append("\n".join(current))

    return parts or [text]


def _chunk_group(group: list[Element], counter: TokenCounter) -> list[_Pending]:
    """Chunk one section.

    Tables and figure captions are emitted whole and alone: half a table states
    a column heading with no data or data with no heading, and a caption
    belongs to a figure rather than to the prose around it.
    """
    out: list[_Pending] = []
    current: _Pending | None = None

    def close() -> None:
        nonlocal current
        if current is not None:
            out.append(current)
            current = None

    for element in group:
        if element.element_type != TYPE_TEXT:
            close()
            for part in _split_unembeddable(element.text, counter):
                out.append(
                    _Pending(
                        texts=[part],
                        page=element.page,
                        section_path=element.section_path,
                        element_type=element.element_type,
                        bboxes=list(element.bboxes),
                    )
                )
            continue

        # A chunk belongs to exactly one page, because a citation resolves to one
        # page and one set of coordinates on it. Merging text from the next page
        # into this chunk would file that text under this chunk's page number,
        # and clicking the citation would open a page the text is not on.
        if current is not None and element.page != current.page:
            close()

        for part in _split_oversized(element.text, counter):
            if current is not None:
                candidate = f"{current.text}\n{part}"
                if counter(candidate) > settings.CHUNK_TARGET_TOKENS:
                    # Full, including whatever overlap it opened with.
                    close()
                else:
                    current.texts.append(part)
                    current.bboxes.extend(element.bboxes)
                    continue

            starter = [part]
            # Overlap only within a page. Repeating the previous page's closing
            # sentences here would put text on this chunk that is not on the page
            # this chunk cites, and it would also make the previous chunk look
            # like a redundant copy of this one.
            if out and out[-1].element_type == TYPE_TEXT and out[-1].page == element.page:
                tail = _tail(out[-1].text, counter)
                # Only worth repeating if it does not swallow the whole budget.
                if tail and counter(f"{tail}\n{part}") <= settings.CHUNK_MAX_TOKENS:
                    starter.insert(0, tail)
            current = _Pending(
                texts=starter,
                page=element.page,
                section_path=element.section_path,
                element_type=TYPE_TEXT,
                bboxes=list(element.bboxes),
            )

    close()
    return out


def _merge_short_tail(pending: list[_Pending], counter: TokenCounter) -> list[_Pending]:
    """Fold an undersized trailing prose chunk back into the one before it.

    A section often ends in a sentence or two. Stored alone that fragment has
    lost its subject, and it competes in search against the chunk that actually
    carries the section. A genuinely short section is left alone rather than
    dropped: degrade, never lose text.
    """
    if len(pending) < 2:
        return pending

    last, previous = pending[-1], pending[-2]
    if last.element_type != TYPE_TEXT or previous.element_type != TYPE_TEXT:
        return pending
    if counter(last.text) >= settings.CHUNK_MIN_TOKENS:
        return pending
    # Never across a page break: the merged chunk would carry the earlier page's
    # number while holding text from the later page, so its citation would open
    # the wrong page.
    if last.page != previous.page:
        return pending

    merged = f"{previous.text}\n{last.text}"
    if counter(merged) > settings.CHUNK_MAX_TOKENS:
        return pending

    previous.texts.append(last.text)
    previous.bboxes.extend(last.bboxes)
    return pending[:-1]


def _normalised(text: str) -> str:
    """For comparing two pieces of text for sameness, ignoring layout."""
    return " ".join(text.split()).casefold()


def _drop_redundant(pending: list[_Pending], counter: TokenCounter) -> list[_Pending]:
    """Remove undersized chunks whose text is already carried by a neighbour.

    Two shapes turn up in real documents. A heading is emitted as an element in
    its own right and is also the section path of the elements beneath it, so
    left alone it becomes a three-token chunk saying "17. Administration" while
    the chunk after it already carries those words in its context header. A bare
    figure reference behaves the same way against its own caption.

    Such a chunk cannot be retrieved usefully - it holds a title and no fact -
    and it competes for a slot against the chunk that answers the question. It
    is dropped only when its text demonstrably appears in an adjacent chunk, so
    no wording is lost from the document, only repeated wording from the index.

    Deliberately not a rule about short chunks in general. A short chunk that is
    a whole small subsection is real content, and with its context header it
    still states its own subject. Whether those cost retrieval is a question for
    measurement at the evaluation stage, not for a guess here.
    """
    keep: list[_Pending] = []
    for index, item in enumerate(pending):
        if counter(item.text) >= settings.CHUNK_MIN_TOKENS:
            keep.append(item)
            continue

        text = _normalised(item.text)
        # The previous surviving chunk, and the next one as it stands. Comparing
        # against what survived means two identical neighbours leave one copy
        # rather than cancelling each other out.
        neighbours = [neighbour for neighbour in (keep[-1] if keep else None,) if neighbour]
        neighbours += pending[index + 1 : index + 2]

        if text and any(_is_carried_by(text, other) for other in neighbours):
            continue
        keep.append(item)
    return keep


def _is_carried_by(text: str, other: _Pending) -> bool:
    """Whether a neighbour already holds this wording.

    Containment must be strict. Two sections can legitimately hold the identical
    short sentence - a guide repeating "Collaborate with the Onboarding
    Coordinator" under each phase - and those are separate statements that each
    deserve their own citation, not duplicates. Only text that a neighbour holds
    *in addition to* its own content is genuinely repeated.
    """
    body = _normalised(other.text)
    if text in _normalised(other.section_path):
        return True
    return text in body and len(body) > len(text)


def chunk(
    document: ExtractedDocument,
    title: str = "",
    counter: TokenCounter | None = None,
) -> list[Chunk]:
    """Chunk a whole extracted document, in reading order.

    `title` is the document's display name, used only in the context header. It
    is passed in rather than read from the file, because the name shown in a
    citation is the registry's to decide, not the extractor's.
    """
    counter = counter or count_tokens

    # A contents page holds the vocabulary of every topic in the document and the
    # answer to none of them, so it scores respectably against many questions and
    # satisfies none. The extractor identifies those pages but leaves their
    # elements in place, because whether to index them is a chunking decision,
    # not an extraction one.
    elements = [
        element for element in document.elements if element.page not in document.contents_pages
    ]

    pending: list[_Pending] = []
    for group in _groups(elements):
        pending.extend(_merge_short_tail(_chunk_group(group, counter), counter))
    pending = _drop_redundant(pending, counter)

    chunks: list[Chunk] = []
    for index, item in enumerate(pending):
        text = item.text
        chunks.append(
            Chunk(
                index=index,
                text=text,
                page=item.page,
                section_path=item.section_path,
                element_type=item.element_type,
                token_count=counter(text),
                bboxes=item.bboxes,
                context_header=context_header(title, item.section_path, item.page),
            )
        )
    return chunks
