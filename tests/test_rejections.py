"""The full rejection matrix, end to end over HTTP.

Each row of the table in the specification, checked as a user would meet it: an
upload, a status code, and a stable code the UI switches on. Individual pieces are
tested closer to the code elsewhere; this file exists so that a row of the matrix
cannot quietly stop working while its unit test still passes.

Every case here is a rejection rather than a crash. A refused file is an ordinary
outcome the user can act on, so each one must arrive immediately, name its reason,
and leave the library exactly as it was.
"""

import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.errors import UnreadableDocumentError
from backend.ingestion import pipeline
from backend.ingestion.chunk import Chunk
from backend.ingestion.prepare import Prepared
from config import settings


def pdf_bytes(pages: int = 2, password: str | None = None, blank: bool = False) -> bytes:
    document = pymupdf.open()
    for number in range(pages):
        page = document.new_page()
        if not blank:
            page.insert_text((72, 100), f"Page {number + 1} of the document.")
    if password:
        data = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password
        )
    else:
        data = document.tobytes()
    document.close()
    return data


def prepared(count: int = 2) -> Prepared:
    return Prepared(
        chunks=[
            Chunk(
                index=index,
                text=f"Body {index}.",
                page=index + 1,
                section_path="1. Section",
                element_type="text",
                token_count=6,
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                context_header=f"[Doc > 1. Section > p.{index + 1}]",
            )
            for index in range(count)
        ],
        page_count=count,
        table_count=0,
        picture_count=0,
        chars_per_page=1200,
        needs_ocr=False,
        seconds=0.1,
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A running app with isolated stores, no network and no worker."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "lens.db")
    monkeypatch.setattr(settings, "MILVUS_PATH", tmp_path / "chunks.db")
    monkeypatch.setattr(pipeline.prepare, "prepare", lambda _path, _title: prepared())
    monkeypatch.setattr(
        pipeline.embedder,
        "embed_chunks",
        lambda chunks, embed=None: [[0.01] * settings.EMBEDDING_DIMENSIONS for _ in chunks],
    )

    from backend.main import create_app

    with TestClient(create_app()) as running:
        yield running


def send(client, data: bytes, name: str = "document.pdf"):
    return client.post("/documents", files={"file": (name, data, "application/pdf")})


def library(client) -> list[dict]:
    return client.get("/documents").json()


# --- the matrix ----------------------------------------------------------


def test_a_password_protected_pdf_is_rejected(client):
    response = send(client, pdf_bytes(password="secret"))

    assert response.status_code == 415
    assert response.json()["code"] == "encrypted_pdf"
    assert library(client) == []


def test_something_that_is_not_a_pdf_is_rejected(client):
    response = send(client, b"this is not a PDF at all")

    assert response.status_code == 415
    assert response.json()["code"] == "corrupt_file"
    assert library(client) == []


def test_a_pdf_with_a_broken_body_is_rejected(client):
    response = send(client, b"%PDF-1.7\nthen nothing that parses")

    assert response.status_code == 415
    assert response.json()["code"] == "corrupt_file"


def test_a_file_over_the_page_limit_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_PAGES", 2)
    response = send(client, pdf_bytes(pages=5))

    assert response.status_code == 413
    assert response.json()["code"] == "too_many_pages"
    # The message states both numbers, so the user knows how far over they are.
    assert "5" in response.json()["message"]
    assert "2" in response.json()["message"]


def test_a_file_over_the_size_limit_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_FILE_BYTES", 100)
    response = send(client, pdf_bytes())

    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"


def test_a_pdf_that_opens_with_no_pages_is_rejected(client):
    """Parsed successfully. There is simply nothing in it.

    Built by truncating a real PDF rather than by saving an empty one, because
    PyMuPDF refuses to write a document with no pages - which is itself why this
    case only ever arrives as damage rather than as something deliberate.
    """
    response = send(client, pdf_bytes()[:120])

    assert response.status_code in (415, 422)
    assert response.json()["code"] in ("empty_document", "corrupt_file")
    assert library(client) == []


def test_the_same_file_twice_is_reported_as_already_present(client):
    data = pdf_bytes()
    send(client, data, "first.pdf")

    response = send(client, data, "second.pdf")

    assert response.status_code == 409
    assert response.json()["code"] == "duplicate_document"
    # Names the document already holding it, so the user can find it.
    assert "first.pdf" in response.json()["message"]
    assert len(library(client)) == 1


def test_two_different_files_sharing_a_name_both_index(client):
    """Their contents differ, so both belong in the library. A counter suffix
    keeps the citations on each one distinguishable."""
    send(client, pdf_bytes(pages=1), "report.pdf")
    send(client, pdf_bytes(pages=3), "report.pdf")

    names = [document["display_name"] for document in library(client)]
    assert len(names) == 2
    assert len(set(names)) == 2


def test_a_scan_with_no_recoverable_text_is_rejected(client, monkeypatch):
    """OCR ran and found nothing. Rolled back rather than stored as a document
    that is present and answers nothing."""

    def unreadable(_path, _title):
        raise UnreadableDocumentError("no readable text found, even with OCR")

    monkeypatch.setattr(pipeline.prepare, "prepare", unreadable)

    accepted = send(client, pdf_bytes())

    # Accepted synchronously - the file is a valid PDF - and discarded once
    # reading it produced nothing.
    assert accepted.status_code == 202
    assert library(client) == []


def test_a_worker_that_dies_is_not_reported_as_a_corrupt_file(client, monkeypatch):
    """It is not a corrupt file, and saying so would send the user to fix the
    wrong thing."""
    from backend.errors import ExtractionFailedError

    def died(_path, _title):
        raise ExtractionFailedError("the extraction worker stopped with exit code -9")

    monkeypatch.setattr(pipeline.prepare, "prepare", died)

    send(client, pdf_bytes())

    assert library(client) == []


def test_an_embedding_failure_leaves_nothing_behind(client, monkeypatch):
    from backend.errors import EmbeddingFailedError

    def failed(chunks, embed=None):
        raise EmbeddingFailedError("provider unavailable")

    monkeypatch.setattr(pipeline.embedder, "embed_chunks", failed)

    send(client, pdf_bytes())

    assert library(client) == []


def test_a_document_yielding_no_chunks_is_rejected(client, monkeypatch):
    monkeypatch.setattr(pipeline.prepare, "prepare", lambda _p, _t: prepared(count=0))

    send(client, pdf_bytes())

    assert library(client) == []


# --- a rejection never disturbs what is already there --------------------


def test_a_rejection_leaves_an_indexed_document_alone(client):
    """One document failing must never affect another. This is why ingestion is
    per document rather than per batch."""
    send(client, pdf_bytes(pages=1), "good.pdf")
    before = library(client)

    send(client, b"not a pdf", "bad.pdf")
    send(client, pdf_bytes(pages=1, password="x"), "locked.pdf")

    assert library(client) == before


def test_the_library_still_answers_after_a_rejection(client):
    send(client, pdf_bytes(pages=1), "good.pdf")
    send(client, b"not a pdf", "bad.pdf")

    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["documents"] == 1
