"""Reading a document, falling back to OCR only when it is needed.

OCR roughly triples the time and its output is a guess where a real text layer is
exact, so a document is read normally first and its character density decides.
Still below the floor after OCR, the document is rejected rather than indexed:
otherwise it sits in the library answering nothing.

Density is averaged over the whole document. A report with a few full-page charts
has near-empty pages and is not a scan.
"""

import logging
from pathlib import Path

from backend.errors import UnreadableDocumentError
from backend.ingestion import extractor
from backend.ingestion.extractor import ExtractedDocument
from config import settings

log = logging.getLogger(__name__)


def density(document: ExtractedDocument) -> int:
    """Characters per page, averaged over the document."""
    return document.chars_per_page


def needs_ocr(document: ExtractedDocument) -> bool:
    """Whether the text layer is too thin to be real text.

    A scanned page yields a handful of characters at most - a stray mark read as
    a letter. A digital page yields well over a thousand. The floor sits far below
    any real document and far above any scan, so the decision is not close.
    """
    return density(document) < settings.OCR_TRIGGER_CHARS_PER_PAGE


def is_unreadable(document: ExtractedDocument) -> bool:
    """Whether there is too little text to index, even after OCR."""
    return density(document) < settings.MIN_CHARS_PER_PAGE


def read(pdf_path: Path) -> tuple[ExtractedDocument, bool]:
    """Read a document, using OCR only if the first pass found almost nothing.

    Returns the document and whether OCR was used, because the registry records
    that per document and a user is entitled to know an answer came from a
    machine's reading of a picture rather than from the file's own text.

    Raises:
        UnreadableDocumentError: still below the floor after OCR.
        ExtractionFailedError, EmptyDocumentError: from extraction itself.
    """
    document = extractor.extract(pdf_path)

    if not needs_ocr(document):
        return document, False

    log.info(
        "%s has %d characters a page, below the %d floor: reading it again with OCR",
        pdf_path.name,
        density(document),
        settings.OCR_TRIGGER_CHARS_PER_PAGE,
    )

    scanned = extractor.extract(pdf_path, with_ocr=True)

    if is_unreadable(scanned):
        raise UnreadableDocumentError(
            f"no readable text found, even with OCR: {density(scanned)} characters a page, "
            f"below the {settings.MIN_CHARS_PER_PAGE} needed to index"
        )

    log.info("OCR recovered %d characters a page from %s", density(scanned), pdf_path.name)
    return scanned, True
