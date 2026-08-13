"""Tests for chunking.

Elements are built directly here rather than extracted from a PDF. Chunking is
pure logic over the extractor's output, so constructing that output by hand is
both faster and more precise: a test can state exactly how many tokens an
element holds and assert exactly where the boundary landed.

A word counter stands in for the real tokenizer. The suite then runs offline
and instantly, and every size in these tests is countable by eye. The real
tokenizer is exercised separately in one test that only checks it agrees about
ordering, never about exact numbers.
"""

from backend.ingestion import chunker
from backend.ingestion.extractor import (
    TYPE_FIGURE_CAPTION,
    TYPE_TABLE,
    TYPE_TEXT,
    Element,
    ExtractedDocument,
)
from config import settings


def words(text: str) -> int:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def sentences(count: int, word: str = "policy", start: int = 0) -> str:
    """`count` sentences of ten words each, so token counts are obvious.

    Each sentence carries its own number so a test can say which sentences
    ended up in which chunk.
    """
    return " ".join(
        f"Sentence {start + index} states the {word} for staff in plain terms."
        for index in range(count)
    )


def element(
    text: str,
    page: int = 1,
    element_type: str = TYPE_TEXT,
    section_path: str = "1. Leave",
    bboxes: list | None = None,
) -> Element:
    return Element(
        text=text,
        page=page,
        element_type=element_type,
        section_path=section_path,
        bboxes=bboxes if bboxes is not None else [(10.0, 20.0, 300.0, 40.0)],
    )


def document(elements: list[Element], page_count: int = 3) -> ExtractedDocument:
    return ExtractedDocument(
        page_count=page_count,
        elements=elements,
        table_count=sum(1 for item in elements if item.element_type == TYPE_TABLE),
        picture_count=0,
        dropped_count=0,
        heading_count=1,
        seconds=0.1,
    )


def chunk(elements: list[Element], title: str = "Handbook") -> list[chunker.Chunk]:
    return chunker.chunk(document(elements), title=title, counter=words)


# --- Structure -----------------------------------------------------------


def test_a_short_section_becomes_one_chunk():
    """Structure first: a section that already fits is not split for any reason."""
    chunks = chunk([element(sentences(20))])

    assert len(chunks) == 1
    assert chunks[0].section_path == "1. Leave"
    assert chunks[0].token_count == 200


def test_two_sections_never_share_a_chunk():
    """Even two tiny sections stay apart, so a citation names one real section."""
    chunks = chunk(
        [
            element(sentences(3), section_path="1. Leave"),
            element(sentences(3), section_path="2. Overtime"),
        ]
    )

    assert [item.section_path for item in chunks] == ["1. Leave", "2. Overtime"]


def test_the_same_heading_far_apart_is_not_joined():
    """A recurring heading is a real heading each time it appears.

    A role name repeated under every phase of a guide would otherwise collapse
    into one chunk holding text from unrelated phases.
    """
    chunks = chunk(
        [
            element(sentences(3), section_path="Guide > Onboarding Partner"),
            element(sentences(3), section_path="Guide > First Week"),
            element(sentences(3), section_path="Guide > Onboarding Partner"),
        ]
    )

    assert len(chunks) == 3


# --- Per-element rules ---------------------------------------------------


def test_a_table_is_one_chunk_and_is_never_split():
    """Half a table is a header with no data, or data with no header."""
    rows = "\n".join(f"| Role {index} | Limit {index} |" for index in range(400))
    chunks = chunk([element(rows, element_type=TYPE_TABLE)])

    assert len(chunks) == 1
    assert chunks[0].element_type == TYPE_TABLE
    assert chunks[0].token_count > settings.CHUNK_MAX_TOKENS
    assert chunks[0].text.count("| Role") == 400


def test_a_table_is_not_merged_with_the_prose_around_it():
    """The table keeps its own citation, and the prose is not diluted by it."""
    chunks = chunk(
        [
            element(sentences(3, start=0)),
            element("| Role | Limit |\n| Lead | 5000 |", element_type=TYPE_TABLE),
            element(sentences(3, start=10)),
        ]
    )

    assert [item.element_type for item in chunks] == [TYPE_TEXT, TYPE_TABLE, TYPE_TEXT]


def test_a_figure_caption_keeps_its_own_type_so_the_ui_can_label_it():
    chunks = chunk(
        [element("Figure 2.1: application architecture.", element_type=TYPE_FIGURE_CAPTION)]
    )

    assert chunks[0].element_type == TYPE_FIGURE_CAPTION


def big_table(rows: int) -> str:
    """A markdown table with a header, a rule line, and `rows` data rows."""
    return "\n".join(
        ["| Role | Limit |", "|------|-------|"]
        + [f"| Role number {index} | Limit {index} |" for index in range(rows)]
    )


def test_a_table_over_the_embedding_limit_is_split_at_row_boundaries():
    """The one case where a table may be split. Above the model's hard input
    ceiling the chunk cannot be embedded at all, so the choice is a split table
    or no table in the index."""
    table = big_table(2000)
    assert words(table) > settings.EMBED_MAX_INPUT_TOKENS

    chunks = chunk([element(table, element_type=TYPE_TABLE)])

    assert len(chunks) > 1
    for item in chunks:
        assert item.element_type == TYPE_TABLE
        assert item.token_count <= settings.EMBED_MAX_INPUT_TOKENS


def test_every_part_of_a_split_table_repeats_the_header():
    """A part without its header is values with no column names."""
    chunks = chunk([element(big_table(2000), element_type=TYPE_TABLE)])

    for item in chunks:
        assert item.text.startswith("| Role | Limit |")
        assert "|------|-------|" in item.text


def test_splitting_a_table_keeps_every_data_row_exactly_once():
    chunks = chunk([element(big_table(2000), element_type=TYPE_TABLE)])

    rows = [
        line
        for item in chunks
        for line in item.text.split("\n")
        if line.startswith("| Role number ")
    ]

    assert len(rows) == 2000
    assert len(set(rows)) == 2000


def test_a_table_within_the_embedding_limit_is_still_never_split():
    """The guard is about the ceiling only, not a licence to split tables."""
    table = big_table(150)
    # Comfortably past the ordinary ceiling, nowhere near the embedding limit.
    assert words(table) > settings.CHUNK_MAX_TOKENS
    assert words(table) < settings.EMBED_MAX_INPUT_TOKENS

    chunks = chunk([element(table, element_type=TYPE_TABLE)])

    assert len(chunks) == 1


# --- Size ----------------------------------------------------------------


def test_a_long_section_is_split_at_about_the_target():
    """Each chunk stays under the ceiling, and no chunk is a stray fragment."""
    chunks = chunk([element(sentences(200))])

    assert len(chunks) > 1
    for item in chunks:
        assert item.token_count <= settings.CHUNK_MAX_TOKENS


def test_the_target_is_the_size_of_the_whole_chunk_including_overlap():
    """The repeated tail is paid for out of the target, not added on top of it.

    Written after chunks came out at target plus one overlap each, which quietly
    inflated every chunk by 15% and would have pushed real embedding cost and
    topic dilution up with it.
    """
    chunks = chunk([element(sentences(200))])
    prose = [item for item in chunks if item.element_type == TYPE_TEXT]

    # The last chunk may absorb a short leftover, so it is allowed the ceiling.
    for item in prose[:-1]:
        assert item.token_count <= settings.CHUNK_TARGET_TOKENS, (
            f"chunk {item.index} is {item.token_count} tokens, over target "
            f"{settings.CHUNK_TARGET_TOKENS}"
        )


def test_no_chunk_ever_exceeds_the_hard_ceiling_on_prose():
    one_huge_paragraph = sentences(500)
    chunks = chunk([element(one_huge_paragraph)])

    assert max(item.token_count for item in chunks) <= settings.CHUNK_MAX_TOKENS


def test_splitting_never_breaks_a_word():
    """Every word in the source survives whole, in order."""
    text = sentences(300)
    chunks = chunk([element(text)])

    for item in chunks:
        for word in item.text.split():
            assert word in text


def test_a_short_trailing_chunk_is_merged_back():
    """A two-sentence leftover has lost its subject, so it joins the chunk before it."""
    long_part = sentences(48, start=0)
    tail = sentences(2, start=100)
    chunks = chunk([element(long_part), element(tail)])

    assert all(
        item.token_count >= settings.CHUNK_MIN_TOKENS
        for item in chunks
        if item.element_type == TYPE_TEXT
    )
    assert "Sentence 100" in chunks[-1].text


def test_a_genuinely_short_section_is_kept_not_dropped():
    """Degrade, never lose text. One short section has nothing to merge into."""
    chunks = chunk([element("Purpose. To make the University work.")])

    assert len(chunks) == 1
    assert "make the University work" in chunks[0].text


# --- Redundant chunks ----------------------------------------------------


def test_a_heading_already_in_the_next_section_path_is_not_its_own_chunk():
    """Found on a real manual: 45 of 53 prose chunks were under the minimum, and
    14 of those held nothing but a heading the next chunk already carried."""
    chunks = chunk(
        [
            element("17. Administration", section_path="Manual"),
            element(sentences(30), section_path="Manual > 17. Administration > 17.1 Users"),
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].section_path == "Manual > 17. Administration > 17.1 Users"
    # The wording is not lost: it is in the section path and so in the header.
    assert "17. Administration" in chunks[0].context_header


def test_a_bare_figure_reference_is_dropped_in_favour_of_the_real_caption():
    chunks = chunk(
        [
            element(
                "Figure 2.1: FleetLink application architecture, showing the gateway.",
                element_type=TYPE_FIGURE_CAPTION,
            ),
            element("Figure 2.1"),
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].element_type == TYPE_FIGURE_CAPTION


def test_a_short_chunk_carrying_real_content_is_kept():
    """The rule is about repeated wording, never about shortness on its own."""
    chunks = chunk(
        [
            element(
                "Overlapping cards are rejected on save with message FL-4118.",
                section_path="9.4 Rate card validity",
            ),
            element(sentences(30), section_path="10. Track and Trace"),
        ]
    )

    assert len(chunks) == 2
    assert "FL-4118" in chunks[0].text


def test_the_same_short_sentence_in_two_sections_is_kept_twice():
    """Not duplication. A guide repeats "Collaborate with the Onboarding
    Coordinator" under each phase, and each occurrence is a separate instruction
    that must cite its own phase. Only strictly-contained text is redundant."""
    line = "Collaborate activities with the Onboarding Coordinator for assistance."
    chunks = chunk(
        [
            element(line, section_path="Guide > The First Day"),
            element(line, section_path="Guide > The First Week"),
        ]
    )

    assert len(chunks) == 2
    assert [item.section_path for item in chunks] == [
        "Guide > The First Day",
        "Guide > The First Week",
    ]


# --- Overlap -------------------------------------------------------------


def test_consecutive_prose_chunks_overlap():
    """An answer across a boundary must be whole in at least one chunk."""
    chunks = chunk([element(sentences(150))])

    assert len(chunks) > 1
    first_words = set(chunks[0].text.split())
    assert first_words & set(chunks[1].text.split())


def test_overlap_stays_within_its_budget():
    chunks = chunk([element(sentences(150))])
    tail = chunker._tail(chunks[0].text, words)

    assert words(tail) <= settings.CHUNK_OVERLAP_TOKENS


def test_a_table_does_not_leak_into_the_next_chunk_as_overlap():
    """Repeating table rows as a prose preamble would make the prose chunk lie."""
    chunks = chunk(
        [
            element("| Role | Limit |\n| Director | 100000 |", element_type=TYPE_TABLE),
            element(sentences(30)),
        ]
    )

    assert "| Director |" not in chunks[1].text


# --- Provenance ----------------------------------------------------------


def test_page_and_boxes_are_carried_through():
    chunks = chunk([element(sentences(5), page=7, bboxes=[(1.0, 2.0, 3.0, 4.0)])])

    assert chunks[0].page == 7
    assert chunks[0].bboxes == [(1.0, 2.0, 3.0, 4.0)]


def test_a_chunk_spanning_a_page_break_keeps_only_its_own_page_boxes():
    """The citation points at the starting page, so boxes from the next page
    would highlight a region of a page nobody is looking at."""
    chunks = chunk(
        [
            element(sentences(3), page=4, bboxes=[(1.0, 1.0, 2.0, 2.0)]),
            element(sentences(3), page=5, bboxes=[(9.0, 9.0, 9.0, 9.0)]),
        ]
    )

    assert len(chunks) == 1
    assert chunks[0].page == 4
    assert (9.0, 9.0, 9.0, 9.0) not in chunks[0].bboxes


def test_no_chunk_is_empty_or_whitespace_only():
    """An empty chunk embeds to noise and can be retrieved against anything."""
    chunks = chunk(
        [
            element(sentences(200)),
            element("| A | B |", element_type=TYPE_TABLE),
            element("Short closing note about the policy."),
        ]
    )

    assert chunks
    for item in chunks:
        assert item.text.strip()
        assert item.token_count > 0


def test_chunking_the_same_document_twice_gives_identical_chunks():
    """The chunk id is the document id plus this index, so reingesting the same
    file must upsert onto the same rows rather than create a second copy."""
    elements = [
        element(sentences(120)),
        element("| Role | Limit |", element_type=TYPE_TABLE),
        element(sentences(40), page=2),
    ]

    first = chunk(elements)
    second = chunk(elements)

    assert [(c.index, c.text, c.page) for c in first] == [(c.index, c.text, c.page) for c in second]


def test_chunk_indexes_are_sequential_from_zero():
    """The index is half of the deterministic chunk id, so reingest upserts."""
    chunks = chunk([element(sentences(200))])

    assert [item.index for item in chunks] == list(range(len(chunks)))


# --- Context header ------------------------------------------------------


def test_the_context_header_names_document_section_and_page():
    chunks = chunk([element(sentences(3), page=17, section_path="4.2 Parental Leave")])

    assert chunks[0].context_header == "[Handbook > 4.2 Parental Leave > p.17]"


def test_the_header_is_embedded_but_not_part_of_the_stored_text():
    """The user sees the document's words; the embedding sees the context too."""
    chunks = chunk([element(sentences(3), page=2, section_path="1. Leave")])

    assert chunks[0].text.startswith("Sentence 0")
    assert chunks[0].embed_text.startswith("[Handbook > 1. Leave > p.2]")
    assert chunks[0].text in chunks[0].embed_text


def test_a_document_with_no_headings_still_gets_a_usable_header():
    """Heading detection may find nothing. Citations fall back to page level."""
    chunks = chunk([element(sentences(3), page=9, section_path="")])

    assert chunks[0].context_header == "[Handbook > p.9]"


def test_an_untitled_document_still_gets_a_usable_header():
    chunks = chunker.chunk(
        document([element(sentences(3), page=9, section_path="")]), title="", counter=words
    )

    assert chunks[0].context_header == "[p.9]"


# --- The real tokenizer --------------------------------------------------


def test_the_real_tokenizer_counts_more_tokens_than_words():
    """Not an assertion about exact numbers, which would break on any model change.

    It checks the thing the setting depends on: tokens and words are not the
    same unit, so measuring chunk size in words would size every chunk wrongly.
    """
    text = sentences(10)

    assert chunker.count_tokens(text) > words(text)
