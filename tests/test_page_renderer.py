"""Tests for turning a cited page into a picture.

A highlight has to land on the words it names. A box in the wrong place is worse than
no box: the user is shown a sentence and told it is the source.

So these tests check pixels. A box over known text must darken that region and leave
the rest alone, and a box that does not belong on the page must not be moved onto
words that were never cited.
"""

import pymupdf
import pytest

from backend.errors import PageNotFoundError, RenderFailedError
from backend.rendering import page_renderer
from config import settings

# Where the fixture writes its text, in PDF points with a top-left origin - the
# same convention extraction stores boxes in.
LINE_ONE = (72.0, 92.0, 400.0, 108.0)
LINE_TWO = (72.0, 192.0, 400.0, 208.0)


@pytest.fixture
def pdf(tmp_path):
    """Two pages, each with text at a known position."""
    document = pymupdf.open()
    for number in (1, 2):
        page = document.new_page()
        page.insert_text((72, 104), f"First line on page {number}.", fontsize=12)
        page.insert_text((72, 204), f"Second line on page {number}.", fontsize=12)
    path = tmp_path / "fixture.pdf"
    document.save(path)
    document.close()
    return path


def darkness(image: bytes, box) -> float:
    """Mean ink in a region of the rendered image, 0 (white) to 1 (black).

    Sampled from the PNG rather than from the page, because what matters is what
    the user is shown.
    """
    pixmap = pymupdf.Pixmap(image)
    scale = settings.RENDER_DPI / 72.0
    left, top, right, bottom = (int(value * scale) for value in box)

    total = 0
    count = 0
    for y in range(max(0, top), min(pixmap.height, bottom)):
        for x in range(max(0, left), min(pixmap.width, right)):
            pixel = pixmap.pixel(x, y)
            total += sum(pixel[:3]) / 3
            count += 1
    return 1.0 - (total / count / 255) if count else 0.0


# --- the page renders ----------------------------------------------------


def test_a_page_renders_as_a_png(pdf):
    image = page_renderer.render(pdf, 1)

    assert image[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_image_is_rendered_at_the_configured_density(pdf):
    image = page_renderer.render(pdf, 1)
    pixmap = pymupdf.Pixmap(image)

    with pymupdf.open(pdf) as document:
        expected = document[0].rect.width * settings.RENDER_DPI / 72.0

    assert abs(pixmap.width - expected) <= 2


def test_each_page_renders_its_own_content(pdf):
    """An off-by-one here shows the user the wrong page and tells them it is the
    source."""
    first = page_renderer.render(pdf, 1)
    second = page_renderer.render(pdf, 2)

    assert first != second


def test_a_page_without_boxes_still_renders(pdf):
    """A citation whose coordinates were lost is worth less than one with them,
    but the page is still the source."""
    assert page_renderer.render(pdf, 1, None)
    assert page_renderer.render(pdf, 1, [])


# --- the highlight lands where it should ---------------------------------


def test_a_highlight_darkens_the_region_it_names(pdf):
    plain = page_renderer.render(pdf, 1)
    marked = page_renderer.render(pdf, 1, [LINE_ONE])

    assert darkness(marked, LINE_ONE) > darkness(plain, LINE_ONE)


def test_a_highlight_leaves_the_rest_of_the_page_alone(pdf):
    """The second line was not cited, so it must look exactly as it did."""
    plain = page_renderer.render(pdf, 1)
    marked = page_renderer.render(pdf, 1, [LINE_ONE])

    before = darkness(plain, LINE_TWO)
    after = darkness(marked, LINE_TWO)

    assert abs(after - before) < 0.01


def test_the_highlighted_text_is_still_readable(pdf):
    """Translucent on purpose. The point is to show where the answer came from,
    not to cover it."""
    marked = page_renderer.render(pdf, 1, [LINE_ONE])

    # Fully opaque fill would push this region close to solid.
    assert darkness(marked, LINE_ONE) < 0.9


def test_several_boxes_are_all_drawn(pdf):
    """A chunk spanning three lines stores three boxes, and all three are part of
    what the answer used."""
    marked = page_renderer.render(pdf, 1, [LINE_ONE, LINE_TWO])
    plain = page_renderer.render(pdf, 1)

    assert darkness(marked, LINE_ONE) > darkness(plain, LINE_ONE)
    assert darkness(marked, LINE_TWO) > darkness(plain, LINE_TWO)


def test_a_box_the_wrong_way_round_is_still_drawn(pdf):
    """Stored coordinates are occasionally inverted on one axis, which PyMuPDF
    treats as an empty rectangle and silently draws nothing."""
    left, top, right, bottom = LINE_ONE
    inverted = (right, bottom, left, top)

    marked = page_renderer.render(pdf, 1, [inverted])
    plain = page_renderer.render(pdf, 1)

    assert darkness(marked, LINE_ONE) > darkness(plain, LINE_ONE)


def test_a_box_outside_the_page_is_skipped_not_moved(pdf):
    """Clamping would put the highlight on words that were never cited, and tell
    the user those words were the source."""
    far_away = (2000.0, 3000.0, 2400.0, 3100.0)

    marked = page_renderer.render(pdf, 1, [far_away])
    plain = page_renderer.render(pdf, 1)

    assert darkness(marked, LINE_ONE) == pytest.approx(darkness(plain, LINE_ONE), abs=0.01)
    assert darkness(marked, LINE_TWO) == pytest.approx(darkness(plain, LINE_TWO), abs=0.01)


def test_a_malformed_box_does_not_break_the_render(pdf):
    """One unusable box must not cost the whole page."""
    image = page_renderer.render(pdf, 1, [("x", None, 1, 2), LINE_ONE])

    assert darkness(image, LINE_ONE) > darkness(page_renderer.render(pdf, 1), LINE_ONE)


# --- failures ------------------------------------------------------------


def test_a_page_beyond_the_document_is_reported(pdf):
    with pytest.raises(PageNotFoundError):
        page_renderer.render(pdf, 3)


def test_page_zero_is_reported(pdf):
    """Pages are 1-based everywhere they are stored and everywhere a user sees
    them, so zero is a caller mistake rather than the first page."""
    with pytest.raises(PageNotFoundError):
        page_renderer.render(pdf, 0)


def test_a_missing_file_is_reported(tmp_path):
    """The original is what makes a citation checkable. If it is gone, say so."""
    with pytest.raises(RenderFailedError):
        page_renderer.render(tmp_path / "never-existed.pdf", 1)


def test_a_file_that_is_not_a_pdf_is_reported(tmp_path):
    path = tmp_path / "not.pdf"
    path.write_text("this is not a PDF")

    with pytest.raises(RenderFailedError):
        page_renderer.render(path, 1)
