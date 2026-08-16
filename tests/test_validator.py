"""Tests for upload validation.

Each rejection has its own type, because the API maps a type to a message and the
UI must not have to read wording to know what happened. So these tests assert on
the exception class, never on its text.

The order of the checks is tested too. A password-protected 60-page file must be
reported as encrypted rather than as too long: the user can do something about
one of those and not the other.
"""

import pymupdf
import pytest

from backend.errors import (
    CorruptFileError,
    DuplicateDocumentError,
    EmptyDocumentError,
    EncryptedPDFError,
    FileTooLargeError,
    TooManyPagesError,
)
from backend.ingestion import validator
from config import settings


def pdf_bytes(pages: int = 1, password: str | None = None) -> bytes:
    """A real PDF, built in memory. No fixture files, no fixed page counts."""
    document = pymupdf.open()
    for number in range(pages):
        page = document.new_page()
        page.insert_text((72, 100), f"Page {number + 1} of this document.")
    if password:
        # A user password, not only an owner password. An owner password merely
        # restricts what may be done with a document that still opens freely, so
        # it would not exercise the check at all.
        data = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password
        )
    else:
        data = document.tobytes()
    document.close()
    return data


# --- the happy path ------------------------------------------------------


def test_a_valid_pdf_reports_what_was_learned():
    data = pdf_bytes(pages=3)
    inspection = validator.validate(data)

    assert inspection.page_count == 3
    assert inspection.size_bytes == len(data)
    assert len(inspection.content_hash) == 64


def test_the_hash_is_of_the_bytes_not_the_name():
    """Two uploads of one file under different names are one document."""
    data = pdf_bytes()

    assert validator.content_hash(data) == validator.content_hash(data)


def test_different_files_hash_differently():
    assert validator.content_hash(pdf_bytes(1)) != validator.content_hash(pdf_bytes(2))


# --- rejections ----------------------------------------------------------


def test_a_file_already_in_the_library_is_rejected_as_a_duplicate():
    with pytest.raises(DuplicateDocumentError):
        validator.validate(pdf_bytes(), seen=lambda _digest: "Employee Handbook.pdf")


def test_a_file_over_the_size_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "MAX_FILE_BYTES", 10)

    with pytest.raises(FileTooLargeError):
        validator.validate(pdf_bytes())


def test_something_that_is_not_a_pdf_is_rejected_as_corrupt():
    with pytest.raises(CorruptFileError):
        validator.validate(b"this is not a PDF at all")


def test_a_pdf_header_with_a_broken_body_is_rejected_as_corrupt():
    with pytest.raises(CorruptFileError):
        validator.validate(b"%PDF-1.7\nthen nothing that parses")


def test_a_file_that_opens_with_no_pages_is_rejected_as_empty():
    """Distinct from corrupt: it parsed. There is simply nothing in it.

    PyMuPDF repairs a good deal of damage rather than refusing, so a badly
    truncated file often arrives here rather than as a parse failure.
    """
    with pytest.raises(EmptyDocumentError):
        validator.validate(pdf_bytes()[:120])


def test_a_password_protected_pdf_is_rejected_as_encrypted():
    """Distinct from corrupt. The user can supply a decrypted copy."""
    with pytest.raises(EncryptedPDFError):
        validator.validate(pdf_bytes(password="secret"))


def test_a_file_over_the_page_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(settings, "MAX_PAGES", 2)

    with pytest.raises(TooManyPagesError):
        validator.validate(pdf_bytes(pages=3))


# --- the order of the checks ---------------------------------------------


def test_a_duplicate_is_detected_before_anything_expensive(monkeypatch):
    """A duplicate should cost nothing to reject, so it is found before the file
    is measured or opened. Both limits are set to fail here; the duplicate wins."""
    monkeypatch.setattr(settings, "MAX_FILE_BYTES", 1)
    monkeypatch.setattr(settings, "MAX_PAGES", 0)

    with pytest.raises(DuplicateDocumentError):
        validator.validate(pdf_bytes(), seen=lambda _digest: "already here.pdf")


def test_an_encrypted_file_is_reported_as_encrypted_not_as_too_long(monkeypatch):
    """An encrypted PDF reports its page count happily and then yields no text.
    Rejecting it on length would tell the user the wrong thing about their file."""
    monkeypatch.setattr(settings, "MAX_PAGES", 1)

    with pytest.raises(EncryptedPDFError):
        validator.validate(pdf_bytes(pages=5, password="secret"))


def test_a_size_rejection_does_not_require_opening_the_file(monkeypatch):
    """Bytes that are too long are rejected without being parsed, which is the
    point of checking size first."""
    monkeypatch.setattr(settings, "MAX_FILE_BYTES", 4)

    with pytest.raises(FileTooLargeError):
        validator.validate(b"not a pdf but definitely too long")
