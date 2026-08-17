"""Tests for the HTTP layer.

A real app against real stores in a temporary directory. The model calls and the
extraction worker are replaced, so nothing here costs money or spawns Docling,
but SQLite and Milvus are the actual ones - a rejection has to be shown coming
back as the right status with the right code, and a mock would agree with
anything.

The rejection codes are the point. The UI switches on `code` and never on
`message`, so these tests assert on the code and the status, never on wording.
"""

import json

import pymupdf
import pytest
from fastapi.testclient import TestClient

from backend.api import routes_chat
from backend.errors import StoreMismatchError
from backend.ingestion import embedder, pipeline
from backend.ingestion.chunk import Chunk
from backend.ingestion.prepare import Prepared
from backend.main import check_stores_agree
from backend.retrieval import analyzer, condenser, generator
from backend.storage import conversations, registry, vector_store
from config import settings


def pdf_bytes(pages: int = 2, password: str | None = None) -> bytes:
    document = pymupdf.open()
    for number in range(pages):
        document.new_page().insert_text((72, 100), f"Page {number + 1} of the document.")
    if password:
        data = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw=password, owner_pw=password
        )
    else:
        data = document.tobytes()
    document.close()
    return data


def prepared(count: int = 3) -> Prepared:
    return Prepared(
        chunks=[
            Chunk(
                index=index,
                text=f"Body of chunk {index}.",
                page=index + 1,
                section_path="1. Leave",
                element_type="text",
                token_count=8,
                bboxes=[(1.0, 2.0, 3.0, 4.0)],
                context_header=f"[Doc > 1. Leave > p.{index + 1}]",
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
    """A running app with isolated stores and no network anywhere."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "lens.db")
    monkeypatch.setattr(settings, "MILVUS_PATH", tmp_path / "chunks.db")

    # Extraction runs in-process and returns a fixed result: the worker itself is
    # tested elsewhere, and spawning it here would make every API test slow.
    monkeypatch.setattr(pipeline.prepare, "prepare", lambda _path, _title: prepared())
    monkeypatch.setattr(
        pipeline.embedder,
        "embed_chunks",
        lambda chunks, embed=None: [[0.01] * settings.EMBEDDING_DIMENSIONS for _ in chunks],
    )
    # The query side embeds too. Without this a chat turn makes a real call and
    # fails on the fake key, three times over, after the stream has opened.
    monkeypatch.setattr(
        embedder, "embed_query", lambda question, embed=None: [0.01] * settings.EMBEDDING_DIMENSIONS
    )

    from backend.main import create_app

    with TestClient(create_app()) as running:
        yield running


def upload(client, data: bytes | None = None, name: str = "handbook.pdf"):
    return client.post(
        "/documents",
        files={"file": (name, data if data is not None else pdf_bytes(), "application/pdf")},
    )


# --- health --------------------------------------------------------------


def test_health_reports_a_working_backend(client):
    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["embed_model_matches"]
    assert body["gate_threshold"] == settings.GATE_THRESHOLD


# --- upload --------------------------------------------------------------


def test_a_valid_upload_is_accepted_and_indexed(client):
    response = upload(client)

    assert response.status_code == 202
    body = response.json()
    assert body["page_count"] == 2
    # TestClient runs background tasks before returning, so indexing is done.
    assert client.get(f"/documents/{body['doc_id']}/status").json()["stage"] == "ready"


def test_the_library_lists_an_indexed_document(client):
    upload(client)
    library = client.get("/documents").json()

    assert len(library) == 1
    assert library[0]["chunk_count"] == 3


def test_status_reports_progress(client):
    doc_id = upload(client).json()["doc_id"]
    body = client.get(f"/documents/{doc_id}/status").json()

    assert body["finished"]
    assert body["progress"] == 1.0


@pytest.mark.parametrize(
    ("data", "status", "code"),
    [
        (b"not a pdf at all", 415, "corrupt_file"),
        (pdf_bytes(password="secret"), 415, "encrypted_pdf"),
    ],
)
def test_a_bad_file_is_rejected_immediately_with_its_own_code(client, data, status, code):
    """Synchronously, while the user is still looking at the dialog - not minutes
    later through a background job."""
    response = upload(client, data)

    assert response.status_code == status
    assert response.json()["code"] == code


def test_a_file_over_the_page_limit_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "MAX_PAGES", 1)
    response = upload(client, pdf_bytes(pages=3))

    assert response.status_code == 413
    assert response.json()["code"] == "too_many_pages"


def test_the_same_file_twice_is_a_conflict(client):
    data = pdf_bytes()
    upload(client, data)
    response = upload(client, data)

    assert response.status_code == 409
    assert response.json()["code"] == "duplicate_document"


def test_a_rejected_upload_leaves_the_library_empty(client):
    upload(client, b"not a pdf")

    assert client.get("/documents").json() == []


def test_a_missing_document_is_a_404(client):
    response = client.get("/documents/nope/status")

    assert response.status_code == 404
    assert response.json()["code"] == "document_not_found"


def test_a_deleted_document_leaves_the_library(client):
    doc_id = upload(client).json()["doc_id"]

    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.get("/documents").json() == []


# --- conversations -------------------------------------------------------


def test_a_new_chat_searches_the_whole_library(client):
    body = client.post("/conversations", json={}).json()

    assert body["scope_mode"] == conversations.SCOPE_LIBRARY
    assert body["scope_doc_ids"] is None


def test_a_chat_can_be_scoped_to_chosen_documents(client):
    doc_id = upload(client).json()["doc_id"]
    body = client.post(
        "/conversations",
        json={"scope_mode": conversations.SCOPE_SUBSET, "scope_doc_ids": [doc_id]},
    ).json()

    assert body["scope_doc_ids"] == [doc_id]


def test_an_empty_selection_is_refused(client):
    """Every question against it would be refused, which reads as broken rather
    than as the setting the user chose."""
    response = client.post(
        "/conversations", json={"scope_mode": conversations.SCOPE_SUBSET, "scope_doc_ids": []}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "empty_scope"


def test_a_chat_can_be_renamed(client):
    conv_id = client.post("/conversations", json={}).json()["conv_id"]
    body = client.patch(f"/conversations/{conv_id}", json={"title": "Leave policy"}).json()

    assert body["title"] == "Leave policy"
    assert not body["title_is_auto"]


def test_a_missing_chat_is_a_404(client):
    response = client.get("/conversations/nope")

    assert response.status_code == 404
    assert response.json()["code"] == "conversation_not_found"


def test_deleting_a_chat_leaves_documents_alone(client):
    upload(client)
    conv_id = client.post("/conversations", json={}).json()["conv_id"]

    client.delete(f"/conversations/{conv_id}")

    assert client.get("/conversations").json() == []
    assert len(client.get("/documents").json()) == 1


# --- chat ----------------------------------------------------------------
# The model calls are replaced with fixed replies, so these test the pipeline
# wiring and the event order, never what a model would say.


def sse(response):
    """Server-sent events as a list of (event name, payload)."""
    events = []
    name = None
    for line in response.text.splitlines():
        if line.startswith("event:"):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("data:") and name:
            events.append((name, json.loads(line.split(":", 1)[1].strip())))
    return events


@pytest.fixture
def chatting(client, monkeypatch):
    """A client whose model calls are fixed, with one document indexed."""
    upload(client)

    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda message, history=(), chat=None: analyzer.Analysis(
            intent=conversations.INTENT_QUESTION,
            standalone=message,
            original=message,
            was_rewritten=False,
        ),
    )
    monkeypatch.setattr(
        condenser,
        "condense",
        lambda message, chat=None: condenser.Condensed(
            text=message, original=message, was_condensed=False
        ),
    )
    return client


def say(client, conv_id, message):
    return client.post(f"/conversations/{conv_id}/messages", json={"message": message})


def test_a_greeting_is_answered_without_searching(chatting, monkeypatch):
    """Otherwise "hi" is searched, matches nothing, and comes back as a refusal -
    correct by the rules and absurd to a person."""
    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda m, history=(), chat=None: analyzer.Analysis(
            intent=conversations.INTENT_GREETING, standalone=m, original=m, was_rewritten=False
        ),
    )
    conv_id = chatting.post("/conversations", json={}).json()["conv_id"]

    events = sse(say(chatting, conv_id, "hi"))
    names = [name for name, _payload in events]
    done = next(payload for name, payload in events if name == "done")

    assert "token" not in names
    assert not done["abstained"]
    assert done["diagnostics"]["retrieved"] == 0


def test_a_question_about_the_app_is_answered_from_the_library(chatting, monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda m, history=(), chat=None: analyzer.Analysis(
            intent=conversations.INTENT_META, standalone=m, original=m, was_rewritten=False
        ),
    )
    conv_id = chatting.post("/conversations", json={}).json()["conv_id"]

    done = next(p for n, p in sse(say(chatting, conv_id, "what do you have?")) if n == "done")

    assert "handbook.pdf" in done["answer"]
    assert done["diagnostics"]["retrieved"] == 0


def test_an_answer_streams_text_then_citations_then_done(chatting, monkeypatch):
    """Citations come after the text because they cannot be validated until the
    model has stopped citing."""
    monkeypatch.setattr(
        routes_chat.generator,
        "stream",
        lambda q, hits, names, chat=None: iter(
            [
                "Leave accrues ",
                "monthly [1].",
                generator.Answer(
                    text="Leave accrues monthly [1].",
                    citations=routes_chat.generator.citations.validate(
                        "Leave accrues monthly [1].", hits, names
                    ).citations,
                    abstained=False,
                    prompt_passages=len(hits),
                ),
            ]
        ),
    )
    conv_id = chatting.post("/conversations", json={}).json()["conv_id"]

    events = sse(say(chatting, conv_id, "how much leave?"))
    names = [name for name, _payload in events]

    assert names == ["token", "token", "citations", "done"]
    done = dict(events)["done"]
    assert done["answer"] == "Leave accrues monthly [1]."
    assert done["citations"][0]["page"] == 1


def test_a_refusal_sends_no_text_at_all(chatting, monkeypatch):
    """The UI renders it as its own calm state rather than as a streamed answer."""
    monkeypatch.setattr(
        routes_chat.generator,
        "stream",
        lambda q, hits, names, chat=None: iter(
            [
                generator.Answer(
                    text="", citations=[], abstained=True, reason=generator.REASON_NOT_IN_DOCUMENTS
                )
            ]
        ),
    )
    conv_id = chatting.post("/conversations", json={}).json()["conv_id"]

    events = sse(say(chatting, conv_id, "who won the tender?"))
    done = dict(events)["done"]

    assert "token" not in [name for name, _p in events]
    assert done["abstained"]
    assert done["reason"] == generator.REASON_NOT_IN_DOCUMENTS


def test_every_turn_reports_diagnostics(chatting, monkeypatch):
    monkeypatch.setattr(
        routes_chat.generator,
        "stream",
        lambda q, hits, names, chat=None: iter(
            [
                generator.Answer(
                    text="", citations=[], abstained=True, reason=generator.REASON_NOT_IN_DOCUMENTS
                )
            ]
        ),
    )
    conv_id = chatting.post("/conversations", json={}).json()["conv_id"]

    done = next(p for n, p in sse(say(chatting, conv_id, "a question")) if n == "done")

    assert done["diagnostics"]["gate_threshold"] == settings.GATE_THRESHOLD
    assert done["diagnostics"]["retrieved"] > 0
    assert done["diagnostics"]["latency_ms"] >= 0


def test_the_turn_is_stored_and_reopens_with_its_citations(chatting, monkeypatch):
    monkeypatch.setattr(
        routes_chat.generator,
        "stream",
        lambda q, hits, names, chat=None: iter(
            [
                "Leave accrues monthly [1].",
                generator.Answer(
                    text="Leave accrues monthly [1].",
                    citations=routes_chat.generator.citations.validate(
                        "Leave accrues monthly [1].", hits, names
                    ).citations,
                    abstained=False,
                ),
            ]
        ),
    )
    conv_id = chatting.post("/conversations", json={}).json()["conv_id"]
    say(chatting, conv_id, "how much leave?")

    reopened = chatting.get(f"/conversations/{conv_id}").json()

    assert [m["role"] for m in reopened["messages"]] == ["user", "assistant"]
    assert reopened["messages"][1]["citations"][0]["display_name"] == "handbook.pdf"


def test_a_chat_is_named_after_its_first_real_question(chatting, monkeypatch):
    monkeypatch.setattr(
        routes_chat.generator,
        "stream",
        lambda q, hits, names, chat=None: iter(
            [
                generator.Answer(
                    text="", citations=[], abstained=True, reason=generator.REASON_NOT_IN_DOCUMENTS
                )
            ]
        ),
    )
    conv_id = chatting.post("/conversations", json={}).json()["conv_id"]
    say(chatting, conv_id, "How much annual leave do I get?")

    assert chatting.get(f"/conversations/{conv_id}").json()["title"] == (
        "How much annual leave do I get?"
    )


def test_a_greeting_never_names_a_chat(chatting, monkeypatch):
    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda m, history=(), chat=None: analyzer.Analysis(
            intent=conversations.INTENT_GREETING, standalone=m, original=m, was_rewritten=False
        ),
    )
    conv_id = chatting.post("/conversations", json={}).json()["conv_id"]
    say(chatting, conv_id, "hi")

    assert chatting.get(f"/conversations/{conv_id}").json()["title"] is None


def test_a_message_to_a_missing_chat_is_a_404(chatting):
    response = say(chatting, "nope", "hello")

    assert response.status_code == 404
    assert response.json()["code"] == "conversation_not_found"


def test_a_greeting_is_not_shown_as_a_refusal_when_the_chat_is_reopened(chatting, monkeypatch):
    """A greeting has no citations, and reading that as a refusal made the app
    look as though it had failed to answer "hello"."""
    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda m, history=(), chat=None: analyzer.Analysis(
            intent=conversations.INTENT_GREETING, standalone=m, original=m, was_rewritten=False
        ),
    )
    conv_id = chatting.post("/conversations", json={}).json()["conv_id"]
    say(chatting, conv_id, "hi")

    reopened = chatting.get(f"/conversations/{conv_id}").json()
    assistant = reopened["messages"][1]

    assert not assistant["abstained"]
    assert assistant["content"]


def test_a_refusal_is_shown_as_a_refusal_when_the_chat_is_reopened(chatting, monkeypatch):
    monkeypatch.setattr(
        routes_chat.generator,
        "stream",
        lambda q, hits, names, chat=None: iter(
            [
                generator.Answer(
                    text="", citations=[], abstained=True, reason=generator.REASON_NOT_IN_DOCUMENTS
                )
            ]
        ),
    )
    conv_id = chatting.post("/conversations", json={}).json()["conv_id"]
    say(chatting, conv_id, "who won the tender?")

    reopened = chatting.get(f"/conversations/{conv_id}").json()

    assert reopened["messages"][1]["abstained"]


# --- the two stores must agree -------------------------------------------
# They are separate files and nothing keeps them in step. A registry listing
# documents whose chunks are gone answers every question with "not found in your
# documents", for a library that plainly contains documents, and says nothing
# about why.
#
# Built here from their own connections rather than the running app's: the app
# opens SQLite on its own thread, and a connection cannot be used from another.


def indexed_library(tmp_path, chunk_count: int = 5):
    """A registry holding one ready document, and an empty vector store."""
    db = registry.connect(tmp_path / "agree.db")
    document = registry.register(
        db,
        original_filename="handbook.pdf",
        content_hash="deadbeef",
        size_bytes=1024,
        file_path=str(tmp_path / "handbook.pdf"),
    )
    registry.mark_ready(
        db,
        document.doc_id,
        page_count=3,
        chunk_count=chunk_count,
        table_count=0,
        image_count=0,
        chars_per_page=1200,
        ocr_applied=False,
    )
    return db


def test_startup_refuses_when_the_registry_expects_chunks_the_store_does_not_have(
    tmp_path, monkeypatch
):
    """The state left behind by deleting one store's file and not the other."""
    db = indexed_library(tmp_path)
    monkeypatch.setattr(vector_store, "count", lambda _store: 0)

    with pytest.raises(StoreMismatchError):
        check_stores_agree(db, object())

    db.close()


def test_an_empty_library_with_an_empty_store_is_fine(tmp_path, monkeypatch):
    """Nothing has been indexed yet, so there is nothing to disagree about."""
    db = registry.connect(tmp_path / "empty.db")
    monkeypatch.setattr(vector_store, "count", lambda _store: 0)

    check_stores_agree(db, object())

    db.close()


def test_a_healthy_library_starts(tmp_path, monkeypatch):
    """The exact counts are deliberately not compared: a soft-deleted document
    keeps its chunks and a reingest upserts, so both drift legitimately."""
    db = indexed_library(tmp_path, chunk_count=5)
    monkeypatch.setattr(vector_store, "count", lambda _store: 3)

    check_stores_agree(db, object())

    db.close()


# --- the cited page as an image ------------------------------------------
# The feature the whole design exists for: stop taking the answer's word for it
# and look at the page.


def test_a_page_comes_back_as_a_png(client):
    doc_id = upload(client).json()["doc_id"]

    response = client.get(f"/documents/{doc_id}/pages/1")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_page_beyond_the_document_is_a_404(client):
    doc_id = upload(client).json()["doc_id"]

    response = client.get(f"/documents/{doc_id}/pages/99")

    assert response.status_code == 404
    assert response.json()["code"] == "page_not_found"


def test_a_page_of_a_missing_document_is_a_404(client):
    response = client.get("/documents/nope/pages/1")

    assert response.status_code == 404
    assert response.json()["code"] == "document_not_found"


def test_a_removed_document_still_renders_its_pages(client):
    """Its file and row are kept precisely so answers given before it was removed
    stay checkable. Refusing here would break the citations the soft delete was
    designed to protect."""
    doc_id = upload(client).json()["doc_id"]
    client.delete(f"/documents/{doc_id}")

    response = client.get(f"/documents/{doc_id}/pages/1")

    assert response.status_code == 200
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_an_unknown_chunk_id_still_renders_the_page(client):
    """Without coordinates the page renders plain. The page is still the source,
    so showing it beats refusing."""
    doc_id = upload(client).json()["doc_id"]

    response = client.get(f"/documents/{doc_id}/pages/1", params={"chunk_id": "nope:99"})

    assert response.status_code == 200


def test_the_page_image_is_cacheable(client):
    """The bytes are a pure function of the file, the page and the box, and none
    of the three changes."""
    doc_id = upload(client).json()["doc_id"]

    response = client.get(f"/documents/{doc_id}/pages/1")

    assert "max-age" in response.headers.get("cache-control", "")
