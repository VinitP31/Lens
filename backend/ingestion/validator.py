"""Decide whether an uploaded file is worth ingesting, before any slow work starts.

Hash, size, page count, encryption, damage - in that order, cheapest first, because
everything after this point costs seconds per page and money per chunk.

Each failure raises its own type, so nothing downstream has to read a message to know
what happened. Nothing here looks at what the document says.
"""

import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import pymupdf

from backend.errors import (
    CorruptFileError,
    DuplicateDocumentError,
    EmptyDocumentError,
    EncryptedPDFError,
    FileTooLargeError,
    TooManyPagesError,
)
from config import settings

# Answers "have we already got this exact file?". Injected so validation needs no
# database and the whole suite runs without one.
SeenFunction = Callable[[str], str | None]


@dataclass(frozen=True)
class Inspection:
    """What validation learned. All of it cheap, none of it from the text layer."""

    content_hash: str
    size_bytes: int
    page_count: int


def content_hash(data: bytes) -> str:
    """The file's SHA-256, which is what "the same document" means here.

    Deliberately the bytes, not the filename. Two uploads of one file under
    different names are one document; two different files sharing a name are two.
    """
    return hashlib.sha256(data).hexdigest()


def validate(data: bytes, seen: SeenFunction | None = None) -> Inspection:
    """Decide whether this upload is worth ingesting.

    Raises, in the order the checks run:
        DuplicateDocumentError: this exact file is already in the library.
        FileTooLargeError: over the size limit.
        CorruptFileError: not a PDF, or too damaged to open.
        EncryptedPDFError: password protected, so the text cannot be read.
        EmptyDocumentError: opens, but reports no pages.
        TooManyPagesError: over the page limit.

    `seen` maps a hash to the display name of the document already holding it, or
    None. Passed in so this module stays free of the database.
    """
    digest = content_hash(data)

    # First, because it is the one rejection that should cost nothing. Re-hashing
    # a file the library already holds is the entire price of detecting it.
    if seen is not None:
        existing = seen(digest)
        if existing is not None:
            raise DuplicateDocumentError(f"already in the library as {existing!r}")

    # Before opening, because opening a very large file is what the limit exists
    # to avoid paying for.
    if len(data) > settings.MAX_FILE_BYTES:
        raise FileTooLargeError(
            f"{len(data) / 1024 / 1024:.1f} MB, limit is "
            f"{settings.MAX_FILE_BYTES / 1024 / 1024:.0f} MB"
        )

    try:
        with pymupdf.open(stream=data, filetype="pdf") as pdf:
            # Asked before the page count. An encrypted document reports its page
            # count happily and then yields no text, so a page-limit rejection
            # here would tell the user the wrong thing about their file.
            if pdf.needs_pass:
                raise EncryptedPDFError("the PDF is password protected")

            pages = pdf.page_count
    except EncryptedPDFError:
        raise
    except Exception as exc:  # noqa: BLE001 - any failure to open is one outcome
        raise CorruptFileError(f"could not open as a PDF: {type(exc).__name__}") from exc

    if pages == 0:
        raise EmptyDocumentError("the PDF reports zero pages")

    if pages > settings.MAX_PAGES:
        raise TooManyPagesError(f"{pages} pages, limit is {settings.MAX_PAGES}")

    return Inspection(content_hash=digest, size_bytes=len(data), page_count=pages)
