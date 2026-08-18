"""Turn chunks into vectors.

The only part of ingestion that leaves the machine, so it batches, retries
transient failures, and takes the embedding function as an argument.

What gets embedded is `chunk.embed_text` - the context header and the body - and
order is restored from the index the provider returns, never from arrival order:
getting that wrong attaches each chunk's text to another chunk's meaning with no
error anywhere. The query side must embed through this same module.
"""

import os
import time
from collections.abc import Callable, Iterator, Sequence

from backend.errors import EmbeddingFailedError, MissingApiKeyError
from backend.ingestion.chunk import Chunk
from config import settings

# Takes a batch of texts, returns one vector per text, in the same order.
# Injected so tests are free, offline and deterministic.
EmbedFunction = Callable[[Sequence[str]], list[list[float]]]

API_KEY_VARIABLE = "OPENAI_API_KEY"


def _batches(texts: Sequence[str], size: int) -> Iterator[tuple[int, Sequence[str]]]:
    """Slices of `size`, each with the index it starts at."""
    for start in range(0, len(texts), size):
        yield start, texts[start : start + size]


def _retryable(error: Exception) -> bool:
    """Whether trying again could plausibly succeed.

    Matched on the provider's exception classes, never on message text. A bad key
    or a malformed request fails identically every time, so retrying it only
    delays the same failure.
    """
    import openai

    transient = tuple(
        getattr(openai, name)
        for name in (
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
        )
        if hasattr(openai, name)
    )
    return isinstance(error, transient)


def openai_embedder() -> EmbedFunction:
    """The real embedding function, reading its key from the environment.

    Built lazily. Nothing imports a client at module load, so the test suite and
    every offline tool run without a key present.
    """
    key = os.environ.get(API_KEY_VARIABLE)
    if not key:
        raise MissingApiKeyError(f"set {API_KEY_VARIABLE} in .env")

    from openai import OpenAI

    client = OpenAI(api_key=key)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        response = client.embeddings.create(model=settings.EMBEDDING_MODEL, input=list(texts))
        # Sort by the index the provider reports rather than trusting arrival
        # order. Getting this wrong pairs every chunk with the wrong vector and
        # produces no error at all.
        ordered = sorted(response.data, key=lambda item: item.index)
        return [item.embedding for item in ordered]

    return embed


def _embed_batch(texts: Sequence[str], embed: EmbedFunction) -> list[list[float]]:
    """One batch, with retries on transient failures only."""
    last: Exception | None = None

    for attempt in range(1, settings.EMBED_MAX_RETRIES + 1):
        try:
            return embed(texts)
        except Exception as error:  # noqa: BLE001 - re-raised as a typed error below
            last = error
            if attempt == settings.EMBED_MAX_RETRIES or not _retryable(error):
                break
            # Back off further each time, so a rate limit is given room to clear
            # rather than being hit again immediately.
            time.sleep(settings.EMBED_RETRY_BACKOFF_SECONDS * attempt)

    raise EmbeddingFailedError(
        f"embedding failed after {settings.EMBED_MAX_RETRIES} attempts: {last}"
    ) from last


def _check(vectors: list[list[float]], expected: int) -> None:
    """Refuse anything that would poison the index.

    A short batch or a wrong-width vector must stop ingestion here. Written into
    Milvus, the first would silently misalign every later chunk and the second
    would be uncomparable with every other vector in the collection.
    """
    if len(vectors) != expected:
        raise EmbeddingFailedError(f"asked for {expected} vectors, received {len(vectors)}")

    for position, vector in enumerate(vectors):
        if len(vector) != settings.EMBEDDING_DIMENSIONS:
            raise EmbeddingFailedError(
                f"vector {position} has {len(vector)} dimensions, "
                f"expected {settings.EMBEDDING_DIMENSIONS}"
            )


def embed_texts(texts: Sequence[str], embed: EmbedFunction | None = None) -> list[list[float]]:
    """Vectors for these texts, in the same order, batched.

    `embed` defaults to the real provider. Tests pass a stand-in.
    """
    if not texts:
        return []

    embed = embed or openai_embedder()

    vectors: list[list[float]] = []
    for _start, batch in _batches(texts, settings.EMBED_BATCH_SIZE):
        result = _embed_batch(batch, embed)
        _check(result, len(batch))
        vectors.extend(result)

    _check(vectors, len(texts))
    return vectors


def embed_chunks(chunks: Sequence[Chunk], embed: EmbedFunction | None = None) -> list[list[float]]:
    """Vectors for chunks, embedding the context header along with the body.

    `embed_text`, not `text`. The header carries the document, section and page,
    and embedding it is the cheapest retrieval improvement in the pipeline.
    """
    return embed_texts([chunk.embed_text for chunk in chunks], embed=embed)


def embed_query(question: str, embed: EmbedFunction | None = None) -> list[float]:
    """One vector for a question, through the same path as the documents.

    Deliberately in this module rather than in the retrieval code. A query
    embedded by a different model, or with a different prefix, is quietly
    incomparable with everything stored, and nothing reports it.
    """
    vectors = embed_texts([question], embed=embed)
    return vectors[0]
