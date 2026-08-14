"""Tests for backend.ingestion.extractor.

Extraction is slow (seconds per page), so each PDF is converted once per session
and the result is shared across assertions.
"""

import pytest

from backend.errors import ExtractionFailedError
from backend.ingestion import extractor
from config import settings
from tests.conftest import (
    CHAPTER_TITLE,
    GRID_ROWS,
    HEADING_TEXT,
    LEFT_COLUMN,
    PAGE_MARKERS,
    RIGHT_COLUMN,
    RUNNING_FOOTER,
    RUNNING_HEADER,
    SECTION_TITLE,
)


@pytest.fixture(scope="session")
def simple(simple_pdf):
    return extractor.extract(simple_pdf)


@pytest.fixture(scope="session")
def table(table_pdf):
    return extractor.extract(table_pdf)


def test_page_count_matches_the_file(simple):
    assert simple.page_count == 3


def test_every_page_body_is_extracted(simple):
    extracted = " ".join(element.text for element in simple.elements)
    for marker in PAGE_MARKERS:
        assert marker in extracted, f"missing from extraction: {marker}"


def test_text_is_attributed_to_the_correct_page(simple):
    """The most important test in the suite.

    An off-by-one page map makes every citation in the finished app quietly
    wrong, and nothing downstream would ever reveal it.
    """
    for expected_page, marker in enumerate(PAGE_MARKERS, start=1):
        pages = [element.page for element in simple.elements if marker in element.text]
        assert pages, f"text for page {expected_page} was not extracted at all"
        assert pages == [expected_page], f"expected page {expected_page}, got {pages}"


def test_pages_are_numbered_from_one(simple):
    assert min(element.page for element in simple.elements) == 1


def test_every_element_carries_a_bounding_box(simple):
    """Without a box there is nothing to highlight, so citations stop being clickable."""
    missing = [element.text[:40] for element in simple.elements if not element.bboxes]
    assert not missing, f"elements with no bbox: {missing}"


def test_boxes_use_a_top_left_origin(simple):
    """Top must be above bottom, and boxes must sit inside the page."""
    for element in simple.elements:
        for left, top, right, bottom in element.bboxes:
            assert top < bottom, f"box is upside down: {(left, top, right, bottom)}"
            assert left < right
            assert top >= 0 and left >= 0


def test_running_header_and_footer_are_dropped(simple):
    extracted = " ".join(element.text for element in simple.elements)
    assert RUNNING_HEADER not in extracted
    for page in range(1, 4):
        assert RUNNING_FOOTER.format(page=page) not in extracted


def test_headings_become_a_section_path_not_an_element(simple):
    """A heading on its own answers nothing. It labels the text beneath it."""
    texts = [element.text for element in simple.elements]
    assert HEADING_TEXT not in texts

    first_body = next(e for e in simple.elements if PAGE_MARKERS[0] in e.text)
    assert HEADING_TEXT in first_body.section_path


def test_no_element_is_empty(simple):
    assert all(element.text.strip() for element in simple.elements)


def test_element_types_are_from_the_known_set(simple, table):
    allowed = {extractor.TYPE_TEXT, extractor.TYPE_TABLE, extractor.TYPE_FIGURE_CAPTION}
    for document in (simple, table):
        assert {element.element_type for element in document.elements} <= allowed


def test_a_table_is_extracted_as_one_element(table):
    tables = [e for e in table.elements if e.element_type == extractor.TYPE_TABLE]
    assert len(tables) == 1, "a table must never be split across elements"
    assert table.table_count == 1


def test_table_keeps_its_rows_together(table):
    body = next(e.text for e in table.elements if e.element_type == extractor.TYPE_TABLE)
    for value in ("Team lead", "5000", "Director", "100000"):
        assert value in body


def test_character_density_is_reported(simple):
    assert simple.char_count > 0
    assert simple.chars_per_page == simple.char_count // simple.page_count


def test_a_text_layer_does_not_ask_for_ocr(simple):
    assert simple.needs_ocr is False


def test_a_page_with_no_text_layer_asks_for_ocr(imageonly_pdf):
    """Zero elements is not a failure. It is a document that needs OCR."""
    result = extractor.extract(imageonly_pdf)
    assert result.page_count == 2
    assert result.char_count == 0
    assert result.needs_ocr is True


def test_a_corrupt_file_raises_a_typed_error(corrupt_pdf):
    with pytest.raises(ExtractionFailedError) as raised:
        extractor.extract(corrupt_pdf)
    assert raised.value.code == "extraction_failed"


# --- section path depth --------------------------------------------------
# Docling reports level 1 for every heading in every document tested, so depth is
# derived from type size instead. These check that hierarchy survives.


def test_a_larger_heading_becomes_the_parent_of_a_smaller_one(hierarchy_pdf):
    result = extractor.extract(hierarchy_pdf)
    body = next(e for e in result.elements if "thirty days" in e.text)
    assert body.section_path == f"{CHAPTER_TITLE}{extractor.SECTION_SEPARATOR}{SECTION_TITLE}", (
        f"expected the chapter title above the section title, got {body.section_path!r}"
    )


def test_section_paths_stay_within_the_configured_depth(simple, table):
    for document in (simple, table):
        for element in document.elements:
            if element.section_path:
                depth = len(element.section_path.split(extractor.SECTION_SEPARATOR))
                assert depth <= settings.HEADING_MAX_DEPTH


# --- what is worth indexing ----------------------------------------------
# Tested as pure rules rather than through a generated PDF, because whether
# Docling labels a synthetic contents page is a property of the model, not of
# these rules, and a test should fail for its own reason.


def test_text_with_no_letters_is_never_indexed():
    for junk in ("3", "20", "___________________", "__/_ /_", ". . .", "  "):
        assert extractor._is_indexable(junk, page=5, contents_pages=set()) is False, junk


def test_a_form_field_label_survives_even_next_to_its_blank():
    assert extractor._is_indexable("Employee's Name:", page=5, contents_pages=set()) is True


def test_a_contents_entry_is_dropped_on_a_contents_page():
    entry = "Attendance Policy .......................... 18"
    assert extractor._is_indexable(entry, page=2, contents_pages={2}) is False


def test_the_same_shape_survives_anywhere_else():
    """A numbered requirement looks exactly like a contents entry.

    Deleting real content is far worse than keeping a contents page, so the
    dotted-leader rule is only allowed to act on a page Docling already
    identified as contents.
    """
    requirement = "FR-01 The system shall record receipts ............ 12"
    assert extractor._is_indexable(requirement, page=14, contents_pages={2}) is True


# --- reading order -------------------------------------------------------
# Docling sometimes emits an element far from where it sits on the page, which
# separates a value from its label. Pages are re-ordered by position, except
# where doing so would interleave two columns of prose.


def test_a_grid_is_read_row_by_row(grid_pdf):
    """Each date must be followed by the event it sits level with.

    Read any other way, a date can end up attached to the wrong event, and an
    answer would state the wrong deadline while citing the right page.
    """
    result = extractor.extract(grid_pdf)
    # Read as one stream, because a layout model may legitimately return the
    # whole grid as a single element. What matters is the order the text ends
    # up in, not how many elements it was split across.
    stream = "\n".join(element.text for element in result.elements)

    for index, (date, event) in enumerate(GRID_ROWS):
        assert date in stream, f"missing from extraction: {date}"
        assert event in stream, f"missing from extraction: {event}"
        assert stream.index(event) > stream.index(date), (
            f"{event!r} should follow {date!r}, not precede it"
        )
        if index + 1 < len(GRID_ROWS):
            next_date = GRID_ROWS[index + 1][0]
            assert stream.index(event) < stream.index(next_date), (
                f"{event!r} belongs to {date!r}, but it came after {next_date!r}"
            )


def test_two_columns_of_prose_are_not_interleaved(two_column_pdf):
    """The case that makes re-ordering dangerous.

    Reading a two-column article across the page mixes the columns into
    nonsense, so pages that look like columns of prose keep the order the
    layout model gave them.
    """
    result = extractor.extract(two_column_pdf)
    order = [e.text for e in result.elements]

    def position(sentence: str) -> int:
        return next(i for i, text in enumerate(order) if sentence[:40] in text)

    left = [position(paragraph) for paragraph in LEFT_COLUMN]
    right = [position(paragraph) for paragraph in RIGHT_COLUMN]

    # Every paragraph of one column must come before every paragraph of the
    # other. Interleaving shows up as the two ranges overlapping.
    assert max(left) < min(right) or max(right) < min(left), (
        f"columns were interleaved: left at {left}, right at {right}"
    )


# --- headings that are not headings --------------------------------------


def test_a_lead_in_sentence_is_not_treated_as_a_heading():
    """A layout model labels these as headings. Accepting that loses them.

    Headings live in the section path rather than in the indexed text, so a
    sentence promoted to a heading stops being searchable, and every list item
    beneath it is filed under a section that does not exist.
    """
    assert not extractor._is_really_a_heading(
        "The vendor must clearly identify all third-party components and describe:"
    )
    assert not extractor._is_really_a_heading(
        "Troy University reserves the right, at its sole discretion, to:"
    )


def test_a_short_label_ending_in_a_colon_is_still_a_heading():
    assert extractor._is_really_a_heading("PERFORMANCE BONDS:")
    assert extractor._is_really_a_heading("Purpose")


def test_a_heading_with_nothing_beneath_it_is_still_indexed(hierarchy_pdf):
    """Two headings in a row leave the first with no text to attach to.

    It would otherwise vanish from the document entirely. On a real sample that
    removed the document's own reference number, which is exactly what somebody
    would search for.
    """
    result = extractor.extract(hierarchy_pdf)
    everything = " ".join(f"{e.text} {e.section_path}" for e in result.elements)
    assert CHAPTER_TITLE in everything
    assert SECTION_TITLE in everything


def test_a_colon_label_sits_under_the_heading_above_it():
    """Both are set in the same size, so size alone cannot separate them.

    "Contract Requirements" names a section; "PERFORMANCE BONDS:" introduces the
    paragraph beneath it. Treated as siblings they replace one another and the
    parent vanishes from every path below it.
    """
    assert extractor._is_really_a_heading("PERFORMANCE BONDS:")
    assert extractor._is_really_a_heading("Contract Requirements")
    # The label is demoted, so it nests rather than replacing its parent.
    assert settings.HEADING_DEMOTE_TRAILING_COLON is True


# --- titles that open their own paragraph --------------------------------
# Some documents set a section title as the first words of its paragraph. Layout
# analysis is right to call that one block, but the title then never becomes a
# heading and every page beneath it inherits whatever came before.


def test_a_title_opening_a_paragraph_is_recognised():
    text = (
        "BEFORE DAY ONE - Ensure everything is in place to welcome the new "
        "employee before their first day in the department."
    )
    assert extractor._run_in_heading(text) == ("BEFORE DAY ONE", settings.RUN_IN_HEADING_LEVEL)


def test_a_title_with_an_aside_and_a_full_stop_is_recognised():
    """The separator varies between documents, and sometimes within one.

    "THE FIRST SIX MONTHS (180-day check in). Continue to..." reads exactly like
    the dash form to a person, and must behave the same.
    """
    text = (
        "THE FIRST SIX MONTHS (180-day check in). Continue to promote collaboration "
        "and network building for the new employee."
    )
    title, level = extractor._run_in_heading(text)
    assert title == "THE FIRST SIX MONTHS"
    assert level == settings.RUN_IN_HEADING_LEVEL


def test_a_lettered_item_becomes_a_sibling_not_a_child():
    """Where one item of a list is typeset as a heading and the rest are not, the
    one that is would otherwise become the parent of its own siblings.

    Section A to N are peers. "Section C" happened to be set as a real heading,
    so D onwards were filed underneath it.
    """
    text = (
        "Section D: Secure Hosting Facility Profile: Details of data warehousing "
        "hosting site, number of facilities and their locations."
    )
    title, level = extractor._run_in_heading(text)
    assert title == "Section D"
    # Ordinary heading depth, so it sits beside a heading rather than under it.
    assert level == settings.RUN_IN_LABEL_LEVEL


def test_an_instruction_ending_in_a_colon_is_not_a_label():
    """Making it a heading would file the checklist items after it underneath it."""
    text = (
        "Establish preferred method of communication: Agree with the new employee "
        "how you will keep in touch during the first week."
    )
    assert extractor._run_in_heading(text) is None


def test_an_ordinary_sentence_is_not_a_title():
    """Inventing a heading is worse than missing one, so the rule stays strict."""
    for text in (
        "The vendor must - as noted above - describe the integration in detail.",
        "Employees who have completed twelve months of continuous service qualify.",
        "NOTE - see below",  # one word, and no prose after it
        "BEFORE DAY ONE - short",  # nothing substantial follows
    ):
        assert extractor._run_in_heading(text) is None, text


def test_a_recognised_title_does_not_change_the_paragraph():
    """The rule only adds a label. It never rewrites the text it found."""
    text = "THE FIRST DAY - This is the new employee's first real impression of the team."
    assert extractor._run_in_heading(text)[0] == "THE FIRST DAY"
    # the paragraph is untouched; the title is still part of it
    assert text.startswith("THE FIRST DAY - This is")


# --- repeated headings ---------------------------------------------------
# A running title set in heading type must not become the parent of unrelated
# sections. But repetition alone does not make a heading furniture: a role label
# recurs under every phase of a guide and is a real heading each time.


def test_a_running_title_and_a_recurring_heading_are_told_apart():
    """Position separates them. Measured on a real guide, a running title varied
    by 0.0pt across eleven pages while a recurring role heading varied by 360pt.
    """
    running_title_tops = [40.1, 40.1, 40.1, 40.1, 40.1]
    role_heading_tops = [130.9, 255.9, 484.8, 116.8, 109.9]

    spread = max(running_title_tops) - min(running_title_tops)
    assert spread <= settings.REPEATED_HEADING_POSITION_SPREAD

    spread = max(role_heading_tops) - min(role_heading_tops)
    assert spread > settings.REPEATED_HEADING_POSITION_SPREAD


# --- duplicate captions --------------------------------------------------
# A caption can arrive twice: bound into its table, and again as an element of
# its own. Both copies indexed means the same sentence embedded twice and two
# citations on one passage.


def _caption(text: str, page: int = 1) -> object:
    return extractor.Element(
        text=text,
        page=page,
        element_type=extractor.TYPE_FIGURE_CAPTION,
        section_path="",
        bboxes=[(0.0, 0.0, 1.0, 1.0)],
    )


def _table(text: str, page: int = 1) -> object:
    return extractor.Element(
        text=text,
        page=page,
        element_type=extractor.TYPE_TABLE,
        section_path="",
        bboxes=[(0.0, 0.0, 1.0, 1.0)],
    )


def test_a_caption_bound_into_its_table_is_not_repeated():
    caption = "Chart 1: Revenue from operations and EBITDA margin."
    kept = extractor._without_duplicate_captions(
        [_table(f"{caption}\n\n| Year | Revenue |\n| 2025 | 3489 |"), _caption(caption)]
    )
    assert len(kept) == 1
    assert kept[0].element_type == extractor.TYPE_TABLE


def test_the_duplicate_is_found_even_when_spacing_differs():
    """One copy can be spacing-repaired while the copy inside a table cannot."""
    kept = extractor._without_duplicate_captions(
        [
            _table("Figure 2: Indicative dock  and staging layout.\n\n| Bay | Use |"),
            _caption("Figure 2: Indicative dock and staging layout."),
        ]
    )
    assert len(kept) == 1


def test_a_caption_standing_alone_is_kept():
    """With no figure or table carrying it, the caption is the only record."""
    caption = "Figure 8.1: Trip status lifecycle."
    kept = extractor._without_duplicate_captions([_caption(caption)])
    assert len(kept) == 1
    assert kept[0].text == caption


def test_the_same_caption_on_a_different_page_is_not_a_duplicate():
    caption = "Figure 1: Process flow."
    kept = extractor._without_duplicate_captions([_caption(caption, 3), _caption(caption, 9)])
    assert len(kept) == 2


# --- contents pages ------------------------------------------------------


def test_a_contents_entry_matches_its_numbered_heading():
    """The entry reads "About This Manual", the heading reads "1. About This Manual"."""
    assert extractor._index_key("  About This Manual") == extractor._index_key(
        "1. About This Manual"
    )
    assert extractor._index_key("Attendance Policy ....... 18") == extractor._index_key(
        "Attendance Policy"
    )


def test_unrelated_lines_do_not_match():
    assert extractor._index_key("Annual leave is eighteen days") != extractor._index_key(
        "Sick leave notification"
    )


# --- mis-decoded characters ----------------------------------------------
# A threshold written "≥ 4 hours" can arrive as "‡ 4 hours", which changes what
# the document requires rather than merely looking wrong. A second reader is
# asked what the character actually is.


def test_a_misdecoded_comparison_sign_is_restored():
    page = "Inbound plan report ≥ 4 hours before arrival Gate entry follows"
    assert extractor._repair_glyphs("‡ 4 hours before arrival", page) == "≥ 4 hours before arrival"


def test_repair_works_through_table_markup():
    """The pipes are ours, not the document's, so they must not block the match."""
    page = "Inbound plan report ≥ 4 hours before arrival"
    repaired = extractor._repair_glyphs("| ‡ 4 hours before arrival |", page)
    assert "≥ 4 hours" in repaired


def test_a_real_dagger_is_left_alone():
    """A document that genuinely uses a dagger keeps it.

    Nothing is assumed about what the character should be: if both readers agree,
    the text does not change.
    """
    page = "See the note marked † for the exceptions that apply"
    assert extractor._repair_glyphs("† for the exceptions that apply", page).startswith("†")


def test_a_passage_repeated_with_the_same_sign_is_still_repaired():
    """A threshold repeated down a column of a table is not ambiguous.

    Which row this text came from does not matter when every row reports the
    same character.
    """
    page = "row one ≥ 4 hours before arrival row two ≥ 4 hours before arrival"
    assert extractor._repair_glyphs("‡ 4 hours before arrival", page) == "≥ 4 hours before arrival"


def test_no_repair_when_the_repeated_passages_disagree():
    """Here the answer really is unknown, and guessing would change a requirement."""
    page = "≥ 4 hours before arrival and also ± 4 hours before arrival"
    text = "‡ 4 hours before arrival"
    assert extractor._repair_glyphs(text, page) == text


def test_no_repair_without_a_second_reading():
    text = "‡ 4 hours before arrival"
    assert extractor._repair_glyphs(text, "") == text


# --- broken words --------------------------------------------------------
# Layout analysis sometimes splits a word: "Checklist" arrives as "Checklis t",
# which no longer matches a search for the word it is.


def test_a_split_word_is_rejoined():
    page = "Supervisor See the Onboard Supervisor Checklist on page 10-12."
    text = "Supervisor See the Onboard Supervisor Checklis t on page 10-12."
    assert "Checklist" in extractor._repair_spacing(text, page)


def test_spacing_repair_leaves_a_table_alone():
    """Table markup is ours, so its text will never match the page as written."""
    text = "| Role | Limit |\n| Staff | 90 |"
    assert extractor._repair_spacing(text, "Role Limit Staff 90") == text


def test_spacing_repair_needs_the_same_characters():
    """Only the spaces may differ. A different word is a different passage."""
    text = "Supervisor See the Onboard Supervisor Handbook"
    assert extractor._repair_spacing(text, "Supervisor See the Onboard Checklist") == text


def test_spacing_repair_skips_an_ambiguous_passage():
    page = "the annual review process the annual review process"
    text = "theannual review process"
    assert extractor._repair_spacing(text, page) == text


# --- cleaning ------------------------------------------------------------


def test_private_use_glyphs_are_stripped():
    """Symbol-font bullets arrive as unreadable private use area characters."""
    cleaned = extractor._clean(" To get the new employee the tools they need")
    assert "" not in cleaned
    assert cleaned.startswith("To get the new employee")


def test_cleaning_keeps_line_breaks_but_collapses_spaces():
    """Line breaks carry table and list structure, so they must survive."""
    cleaned = extractor._clean("| Role   |  Limit |\n|  Staff |  90    |")
    assert "\n" in cleaned
    assert "  " not in cleaned


# --- label and value pairing ---------------------------------------------
# A fact stated as a short label above or beside a bare value arrives as two
# elements. Left apart, several such facts read as a list of unattached words
# and numbers, and an answer can pair the wrong number with the wrong label.


def _line(text: str, left: float, top: float, width: float = 40.0, page: int = 1) -> object:
    return extractor.Element(
        text=text,
        page=page,
        element_type=extractor.TYPE_TEXT,
        section_path="Evaluation",
        bboxes=[(left, top, left + width, top + 12.0)],
    )


def test_a_value_below_its_label_is_joined():
    joined = extractor._with_paired_values(
        [_line("Fees", 297.6, 210.6), _line("30 pts", 292.8, 227.8)]
    )

    assert len(joined) == 1
    assert joined[0].text == "Fees: 30 pts"


def test_a_value_beside_its_label_is_joined():
    joined = extractor._with_paired_values(
        [_line("Total weight", 72.0, 300.0), _line("18 kg", 120.0, 300.0)]
    )

    assert [element.text for element in joined] == ["Total weight: 18 kg"]


def test_columns_a_page_apart_are_not_joined():
    """The boundary of this rule, stated so it is not mistaken for a bug.

    A wide two-column grid - label at the left margin, value near the right -
    is left alone. Anything on the same line would qualify at that distance,
    including two unrelated cells of a three-column row, and inventing a pairing
    is worse than leaving one unstated.
    """
    joined = extractor._with_paired_values(
        [_line("Fees", 72.0, 300.0), _line("30 pts", 400.0, 300.0)]
    )

    assert len(joined) == 2


def test_the_joined_element_keeps_both_boxes():
    """The citation highlights the whole fact, not half of it."""
    joined = extractor._with_paired_values(
        [_line("Fees", 297.6, 210.6), _line("30 pts", 292.8, 227.8)]
    )

    assert len(joined[0].bboxes) == 2


def test_a_label_keeps_its_own_section_path():
    joined = extractor._with_paired_values(
        [_line("Fees", 297.6, 210.6), _line("30 pts", 292.8, 227.8)]
    )

    assert joined[0].section_path == "Evaluation"


def test_two_values_in_a_row_are_not_joined():
    """A column of numbers under one heading is not a label and a value. Joining
    them would state a pairing the document does not make."""
    joined = extractor._with_paired_values(
        [_line("55 pts", 292.8, 176.5), _line("30 pts", 292.8, 194.0)]
    )

    assert len(joined) == 2


def test_a_sentence_is_not_treated_as_a_label():
    """Otherwise any paragraph would swallow the first number beneath it."""
    joined = extractor._with_paired_values(
        [
            _line("Proposals will be scored on the following weights", 125.5, 75.9, width=370.0),
            _line("15 pts", 292.8, 92.0),
        ]
    )

    assert len(joined) == 2


def test_a_value_carrying_words_is_not_a_bare_value():
    """ "30 pts payable in advance" states the pairing itself and needs no help."""
    joined = extractor._with_paired_values(
        [_line("Fees", 297.6, 210.6), _line("30 pts payable in advance", 292.8, 227.8, width=120.0)]
    )

    assert len(joined) == 2


def test_a_label_and_a_value_far_apart_are_not_joined():
    """Two items at opposite ends of a page are not a pair, however they read."""
    joined = extractor._with_paired_values(
        [_line("Fees", 297.6, 90.0), _line("30 pts", 292.8, 640.0)]
    )

    assert len(joined) == 2


def test_a_diagonal_pair_is_not_joined():
    """Neither above nor beside: the boxes share no span, so nothing lines up."""
    joined = extractor._with_paired_values(
        [_line("Fees", 72.0, 200.0), _line("30 pts", 400.0, 215.0)]
    )

    assert len(joined) == 2


def test_a_pair_split_across_a_page_break_is_not_joined():
    joined = extractor._with_paired_values(
        [_line("Fees", 297.6, 700.0), _line("30 pts", 292.8, 72.0, page=2)]
    )

    assert len(joined) == 2


def test_a_label_is_used_once():
    """Consuming both elements stops the value being paired again with the next
    label, which would repeat it in two places."""
    joined = extractor._with_paired_values(
        [_line("Fees", 297.6, 210.6), _line("30 pts", 292.8, 227.8), _line("Bond", 297.6, 245.0)]
    )

    assert [element.text for element in joined] == ["Fees: 30 pts", "Bond"]
