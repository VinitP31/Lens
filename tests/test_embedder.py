"""Tests for embedding.

No call is ever made. Every test injects a stand-in embed function, so the suite
is free, offline and deterministic. What is under test is the batching, the
ordering, the retry policy and the validation - not the provider.
"""

import pytest

from backend.errors import EmbeddingFailedError, MissingApiKeyError
from backend.ingestion import embedder
from backend.ingestion.chunker import Chunk
from config import settings


def fake_embed(texts):
    """One vector per text, first component set from the text's length.

    Distinct per text and cheap to assert on, so a misordered result is visible.
    """
    return [[float(len(text))] + [0.0] * (settings.EMBEDDING_DIMENSIONS - 1) for text in texts]


def chunk(index: int, text: str, header: str = "[Handbook > p.1]") -> Chunk:
    return Chunk(
        index=index,
        text=text,
        page=1,
        section_path="1. Leave",
        element_type="text",
        token_count=len(text.split()),
        bboxes=[(1.0, 2.0, 3.0, 4.0)],
        context_header=header,
    )


# --- What gets embedded --------------------------------------------------


def test_the_context_header_is_embedded_with_the_body():
    """The header is why a question using a heading's words matches a chunk whose
    body uses different words. Embedding the body alone throws that away."""
    seen = []

    def record(texts):
        seen.extend(texts)
        return fake_embed(texts)

    embedder.embed_chunks([chunk(0, "Leave accrues monthly.")], embed=record)

    assert seen == ["[Handbook > p.1]\nLeave accrues monthly."]


def test_a_chunk_with_no_header_still_embeds_its_body():
    seen = []

    def record(texts):
        seen.extend(texts)
        return fake_embed(texts)

    embedder.embed_chunks([chunk(0, "Bare body.", header="")], embed=record)

    assert seen == ["Bare body."]


# --- Order ---------------------------------------------------------------


def test_vectors_come_back_in_the_order_the_texts_went_in():
    """A misalignment here attaches every chunk's text to another chunk's
    meaning, and produces no error anywhere."""
    texts = ["a", "bb", "ccc", "dddd"]

    vectors = embedder.embed_texts(texts, embed=fake_embed)

    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0]


def test_order_survives_being_split_across_batches(monkeypatch):
    monkeypatch.setattr(settings, "EMBED_BATCH_SIZE", 2)
    texts = ["a", "bb", "ccc", "dddd", "eeeee"]

    vectors = embedder.embed_texts(texts, embed=fake_embed)

    assert [vector[0] for vector in vectors] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_the_real_client_reorders_a_response_by_its_reported_index(monkeypatch):
    """The provider is not required to answer in request order; each result
    carries its own index. Trusting arrival order would pair every chunk with
    another chunk's vector and raise nothing.

    This drives the real client function, with the network replaced rather than
    the logic under test.
    """

    class Item:
        def __init__(self, index, embedding):
            self.index = index
            self.embedding = embedding

    class Response:
        # Deliberately backwards: index 1 arrives before index 0.
        data = [
            Item(1, [2.0] + [0.0] * (settings.EMBEDDING_DIMENSIONS - 1)),
            Item(0, [1.0] + [0.0] * (settings.EMBEDDING_DIMENSIONS - 1)),
        ]

    class Embeddings:
        def create(self, model, input):
            return Response()

    class FakeClient:
        def __init__(self, api_key):
            self.embeddings = Embeddings()

    monkeypatch.setenv(embedder.API_KEY_VARIABLE, "test-key")
    monkeypatch.setattr("openai.OpenAI", FakeClient)

    embed = embedder.openai_embedder()
    vectors = embed(["first", "second"])

    assert [vector[0] for vector in vectors] == [1.0, 2.0]


# --- Batching ------------------------------------------------------------


def test_texts_are_sent_in_batches_rather_than_one_at_a_time(monkeypatch):
    """A 144-chunk document should cost a handful of requests, not 144."""
    monkeypatch.setattr(settings, "EMBED_BATCH_SIZE", 10)
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return fake_embed(texts)

    embedder.embed_texts(["x"] * 25, embed=counting)

    assert calls == [10, 10, 5]


def test_a_single_batch_is_one_call():
    calls = []

    def counting(texts):
        calls.append(len(texts))
        return fake_embed(texts)

    embedder.embed_texts(["x"] * 5, embed=counting)

    assert calls == [5]


def test_embedding_nothing_makes_no_call():
    def explode(texts):
        raise AssertionError("should not have been called")

    assert embedder.embed_texts([], embed=explode) == []


# --- Validation ----------------------------------------------------------


def test_a_short_batch_is_refused():
    """Fewer vectors than texts would misalign every chunk after the gap."""

    def drops_one(texts):
        return fake_embed(texts)[:-1]

    with pytest.raises(EmbeddingFailedError) as raised:
        embedder.embed_texts(["a", "bb"], embed=drops_one)

    assert raised.value.code == "embedding_failed"


def test_a_wrong_width_vector_is_refused():
    """A vector of the wrong width cannot be compared with anything already
    stored, and Milvus would reject it later with less context."""

    def too_narrow(texts):
        return [[0.1, 0.2] for _ in texts]

    with pytest.raises(EmbeddingFailedError) as raised:
        embedder.embed_texts(["a"], embed=too_narrow)

    assert "dimensions" in raised.value.detail


# --- Retries -------------------------------------------------------------


def test_a_failure_that_retrying_cannot_fix_is_not_retried(monkeypatch):
    """A rejected key fails identically every time. Retrying only delays it."""
    monkeypatch.setattr(embedder, "_retryable", lambda error: False)
    attempts = []

    def always_fails(texts):
        attempts.append(1)
        raise ValueError("bad request")

    with pytest.raises(EmbeddingFailedError):
        embedder.embed_texts(["a"], embed=always_fails)

    assert len(attempts) == 1


def test_a_transient_failure_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(embedder, "_retryable", lambda error: True)
    monkeypatch.setattr(settings, "EMBED_RETRY_BACKOFF_SECONDS", 0.0)
    attempts = []

    def flaky(texts):
        attempts.append(1)
        if len(attempts) < 2:
            raise ConnectionError("dropped")
        return fake_embed(texts)

    vectors = embedder.embed_texts(["abc"], embed=flaky)

    assert len(attempts) == 2
    assert vectors[0][0] == 3.0


def test_retries_stop_at_the_configured_limit(monkeypatch):
    monkeypatch.setattr(embedder, "_retryable", lambda error: True)
    monkeypatch.setattr(settings, "EMBED_RETRY_BACKOFF_SECONDS", 0.0)
    attempts = []

    def always_fails(texts):
        attempts.append(1)
        raise ConnectionError("dropped")

    with pytest.raises(EmbeddingFailedError):
        embedder.embed_texts(["a"], embed=always_fails)

    assert len(attempts) == settings.EMBED_MAX_RETRIES


def test_the_error_says_how_many_attempts_were_made(monkeypatch):
    monkeypatch.setattr(embedder, "_retryable", lambda error: False)

    def always_fails(texts):
        raise ConnectionError("dropped")

    with pytest.raises(EmbeddingFailedError) as raised:
        embedder.embed_texts(["a"], embed=always_fails)

    assert str(settings.EMBED_MAX_RETRIES) in raised.value.detail


# --- Queries -------------------------------------------------------------


def test_a_query_embeds_through_the_same_path_as_documents():
    """A query embedded by a different model is quietly incomparable with
    everything stored, and nothing reports it."""
    vector = embedder.embed_query("How much leave do I get?", embed=fake_embed)

    assert len(vector) == settings.EMBEDDING_DIMENSIONS


# --- Missing key ---------------------------------------------------------


def test_a_missing_key_names_the_variable_to_set(monkeypatch):
    """The one failure a user can fix themselves, so it must not surface as a
    provider authentication error."""
    monkeypatch.delenv(embedder.API_KEY_VARIABLE, raising=False)

    with pytest.raises(MissingApiKeyError) as raised:
        embedder.openai_embedder()

    assert embedder.API_KEY_VARIABLE in raised.value.detail
    assert raised.value.code == "missing_api_key"
