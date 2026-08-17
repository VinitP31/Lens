"""Reading a document, and falling back to OCR only when it is needed.

OCR is conditional, never on by default, for two reasons that both matter.

It is slow: running it over a PDF that already has a text layer roughly triples
the time and buys nothing. And it is a guess. A digital PDF's text layer is
exactly what the author typed; OCR is a model's reading of a picture of it. Given
both, the text layer wins every time.

So a document is read normally first, and the character density decides. Below
the floor, the same document is read again with the engine switched on. Still
below it afterwards, the document is rejected rather than indexed - a document
that yielded almost nothing would sit in the library answering nothing, and the
user would have no way to tell that from the system simply not finding an answer.

The density is averaged over the whole document, not judged per page. A good
report with a few full-page charts has several near-empty pages and is not
scanned; treating it per page would send it down the slow path for nothing.
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
