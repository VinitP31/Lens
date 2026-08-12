"""Typed exceptions with stable error codes.

Every failure in Lens raises one of these. The `code` is a contract: the UI maps
codes to user-facing messages, so wording can change freely without breaking
anything. Nothing anywhere may match on exception text.

This file grows one build stage at a time, gaining an exception when the code
that raises it is written.
"""


class LensError(Exception):
    """Base for every Lens failure.

    Subclasses set `code` to a short, stable identifier. `detail` carries the
    specifics of this particular occurrence, such as an actual page count.
    """

    code = "lens_error"

    def __init__(self, detail: str = "") -> None:
        self.detail = detail
        super().__init__(detail or self.code)


# --- Extraction ----------------------------------------------------------


class ExtractionFailedError(LensError):
    """Docling could not process the PDF at all."""

    code = "extraction_failed"


class EmptyDocumentError(LensError):
    """The PDF has no pages, or no extractable content on any page."""

    code = "empty_document"
