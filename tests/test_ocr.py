"""Tests for the conditional OCR path.

Two of these convert a real image-only PDF and take a few seconds each. That cost
is worth paying: OCR is the one stage whose behaviour cannot be checked by
inspecting a decision, only by seeing whether words came back out of a picture.

The rest are the decision itself, which is arithmetic and instant.
"""

from dataclasses import replace

import pytest

from backend.errors import UnreadableDocumentError
from backend.ingestion import extractor, ocr
from config import settings
from tests.conftest import SCANNED_LINES


def document(chars: int, pages: int = 2) -> extractor.ExtractedDocument:
    """An extracted document with a chosen character density."""
    body = "x" * chars
    return extractor.ExtractedDocument(
        page_count=pages,
        elements=[
            extractor.Element(text=body, page=page, element_type="text", section_path="")
            for page in range(1, pages + 1)
        ],
        table_count=0,
        picture_count=0,
        dropped_count=0,
        heading_count=0,
        seconds=1.0,
    )


# --- the decision --------------------------------------------------------


def test_a_document_with_plenty_of_text_does_not_need_ocr():
    """Running it would triple the time and replace exact text with a guess."""
    assert not ocr.needs_ocr(document(chars=2000))


def test_a_document_with_almost_no_text_needs_ocr():
    assert ocr.needs_ocr(document(chars=5))


def test_the_floor_is_the_lowest_density_accepted_without_ocr():
    """Stated because this is the only value anyone will ever check by hand."""
    at_floor = document(chars=settings.OCR_TRIGGER_CHARS_PER_PAGE)

    assert not ocr.needs_ocr(at_floor)
    assert ocr.needs_ocr(replace(at_floor, page_count=at_floor.page_count + 1))


def test_density_is_averaged_over_the_whole_document():
    """A good report with a few full-page charts has several near-empty pages and
    is not scanned. Judging per page would send it down the slow path for
    nothing."""
    mostly_full = extractor.ExtractedDocument(
        page_count=4,
        elements=[
            extractor.Element(text="x" * 4000, page=1, element_type="text", section_path=""),
            # Three pages of charts, no text on any of them.
        ],
        table_count=0,
        picture_count=3,
        dropped_count=0,
        heading_count=0,
        seconds=1.0,
    )

    assert ocr.density(mostly_full) == 1000
    assert not ocr.needs_ocr(mostly_full)


def test_a_document_below_the_index_floor_is_unreadable():
    assert ocr.is_unreadable(document(chars=0))


def test_a_document_above_the_index_floor_is_readable():
    assert not ocr.is_unreadable(document(chars=settings.MIN_CHARS_PER_PAGE + 1))


def test_an_empty_document_reports_zero_rather_than_dividing_by_zero():
    empty = extractor.ExtractedDocument(
        page_count=0,
        elements=[],
        table_count=0,
        picture_count=0,
        dropped_count=0,
        heading_count=0,
        seconds=0.0,
    )

    assert ocr.density(empty) == 0


# --- reading a real document ---------------------------------------------


def test_a_document_with_a_text_layer_is_read_without_ocr(simple_pdf):
    """The text layer is exactly what the author typed. OCR is a model's reading
    of a picture of it, so given both, the text layer wins."""
    result, applied = ocr.read(simple_pdf)

    assert not applied
    assert result.chars_per_page > settings.OCR_TRIGGER_CHARS_PER_PAGE


def test_ocr_recovers_the_words_from_a_scan(scanned_pdf):
    """The one test that cannot be replaced by checking a decision: words have to
    come back out of a picture."""
    result, applied = ocr.read(scanned_pdf)

    assert applied
    assert result.chars_per_page >= settings.MIN_CHARS_PER_PAGE

    recovered = " ".join(element.text for element in result.elements).lower()
    # Not an exact match: OCR is a reading, and asserting one would make this test
    # about the engine's version rather than about the pipeline.
    for phrase in ("cold store", "insulated gloves", "forty-five minutes"):
        assert phrase in recovered, f"{phrase!r} not recovered from the scan"


def test_a_scan_with_nothing_on_it_is_rejected(blank_scan_pdf):
    """Rejected rather than indexed. A document that yielded almost nothing would
    sit in the library answering nothing, and a user could not tell that from the
    system simply not finding their answer."""
    with pytest.raises(UnreadableDocumentError):
        ocr.read(blank_scan_pdf)


def test_the_rejection_says_what_was_measured(blank_scan_pdf):
    """A rejection a user cannot act on is worth little. This one names the
    density found and the density needed."""
    with pytest.raises(UnreadableDocumentError, match=str(settings.MIN_CHARS_PER_PAGE)):
        ocr.read(blank_scan_pdf)


def test_the_scanned_fixture_really_has_no_text_layer(scanned_pdf):
    """Guards the fixture itself. If it accidentally kept a text layer, the OCR
    test above would pass without OCR ever running."""
    plain = extractor.extract(scanned_pdf)

    assert plain.chars_per_page < settings.OCR_TRIGGER_CHARS_PER_PAGE
    assert SCANNED_LINES  # the words the fixture draws, used by the test above
