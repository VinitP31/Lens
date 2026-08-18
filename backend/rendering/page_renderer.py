"""Turning a cited page into a picture with the cited text highlighted.

What makes an answer checkable rather than merely attributable: you look at the
page and see the sentence.

An image rather than an embedded viewer - browsers treat a page anchor in an iframe
inconsistently, and no viewer will draw a box over an arbitrary region.

Coordinates need no conversion: extraction stored every box with a top-left origin
because that is how PyMuPDF draws.
"""

import logging
from pathlib import Path

import pymupdf

from backend.errors import PageNotFoundError, RenderFailedError
from config import settings

log = logging.getLogger(__name__)

# A box in PDF points, top-left origin: (left, top, right, bottom).
Box = tuple[float, float, float, float]


def _highlight(page: pymupdf.Page, boxes: list[Box]) -> int:
    """Draw the cited regions. Returns how many were actually drawn.

    A box that falls outside the page is skipped rather than clamped. Clamping
    would move the highlight onto words it does not belong to, which is worse
    than not drawing it: the user would be shown the wrong sentence and told it
    was the source.
    """
    drawn = 0
    limit = page.rect

    for box in boxes:
        try:
            left, top, right, bottom = (float(value) for value in box)
        except (TypeError, ValueError):
            continue

        # Stored boxes are occasionally the wrong way round on one axis, which
        # PyMuPDF treats as an empty rectangle and silently draws nothing.
        rect = pymupdf.Rect(
            min(left, right) - settings.HIGHLIGHT_PADDING,
            min(top, bottom) - settings.HIGHLIGHT_PADDING,
            max(left, right) + settings.HIGHLIGHT_PADDING,
            max(top, bottom) + settings.HIGHLIGHT_PADDING,
        )
        if rect.is_empty or not rect.intersects(limit):
            continue

        page.draw_rect(
            rect,
            color=settings.HIGHLIGHT_BORDER,
            fill=settings.HIGHLIGHT_FILL,
            fill_opacity=settings.HIGHLIGHT_OPACITY,
            width=settings.HIGHLIGHT_BORDER_WIDTH,
        )
        drawn += 1

    return drawn


def render(pdf_path: Path | str, page_number: int, boxes: list[Box] | None = None) -> bytes:
    """A PNG of one page, with the cited regions marked.

    `page_number` is 1-based, as it is everywhere a user sees it and everywhere it
    is stored.

    Raises:
        PageNotFoundError: the page is outside this document.
        RenderFailedError: the file is missing or cannot be read.

    A page with no boxes still renders: the page is still the source.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise RenderFailedError(f"the original file is no longer on disk: {path.name}")

    try:
        document = pymupdf.open(path)
    except Exception as error:  # noqa: BLE001 - any failure to open is one outcome
        raise RenderFailedError(f"could not open {path.name}: {type(error).__name__}") from error

    try:
        if not 1 <= page_number <= document.page_count:
            raise PageNotFoundError(
                f"page {page_number} is outside this document, which has "
                f"{document.page_count} pages"
            )

        page = document[page_number - 1]

        if boxes:
            drawn = _highlight(page, boxes)
            if not drawn:
                # Worth a line in the log: it means a citation resolved to
                # coordinates that are not on the page it names, which would be a
                # real bug in extraction rather than a rendering problem.
                log.warning(
                    "no highlight drawn on page %d of %s from %d box(es)",
                    page_number,
                    path.name,
                    len(boxes),
                )

        return page.get_pixmap(dpi=settings.RENDER_DPI).tobytes("png")
    except PageNotFoundError:
        raise
    except Exception as error:  # noqa: BLE001 - re-raised as a typed error
        raise RenderFailedError(f"could not render page {page_number}: {error}") from error
    finally:
        document.close()
