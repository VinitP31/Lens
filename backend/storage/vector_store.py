"""Chunks and their vectors, in Milvus Lite.

One local file, no server. This module owns everything about how a chunk is
stored and searched, so no caller ever handles a raw Milvus result.

Two things here are load-bearing:

The chunk id is derived, never generated: `{doc_id}:{index}`. Ingesting the same
document twice therefore writes to the same primary keys and updates them in
place, instead of adding a second copy of every chunk.

Search returns an explicit `similarity`, already the right way up. With the
cosine metric Milvus reports similarity in a field it calls `distance`, and the
confidence gate compares against a threshold, so the one place that ambiguity
could invert the whole system is resolved here and nowhere else.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pymilvus import DataType, MilvusClient

from backend.errors import EmbedModelMismatchError
from backend.ingestion.chunker import Chunk
from config import settings


@dataclass(frozen=True)
class Hit:
    """One search result, with the score already the right way up."""

    chunk_id: str
    doc_id: str
    page: int
    section_path: str
    element_type: str
    text: str
    bboxes: list[tuple[float, float, float, float]]
    # Higher is closer. 1.0 is identical, 0.0 unrelated, -1.0 opposite.
    similarity: float
    # Exactly what Milvus returned, kept so both can be logged side by side.
    # The gate is the one number in Lens most likely to be silently inverted.
    raw_distance: float


def chunk_id(doc_id: str, index: int) -> str:
    """`{doc_id}:{index}`.

    Deterministic on purpose, so reingesting a document upserts onto the same
    rows rather than duplicating them.
    """
    return f"{doc_id}:{index}"


def _build_schema(client: MilvusClient):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(
        "chunk_id", DataType.VARCHAR, is_primary=True, max_length=settings.MILVUS_ID_MAX
    )
    # Indexed so a scoped search filters instead of scanning.
    schema.add_field("doc_id", DataType.VARCHAR, max_length=settings.MILVUS_ID_MAX)
    schema.add_field("page", DataType.INT32)
    schema.add_field("section_path", DataType.VARCHAR, max_length=settings.MILVUS_SECTION_PATH_MAX)
    schema.add_field("element_type", DataType.VARCHAR, max_length=32)
    schema.add_field("text", DataType.VARCHAR, max_length=settings.MILVUS_TEXT_MAX)
    # Coordinates for the citation highlight. JSON because a chunk can cover
    # several boxes, and they are never queried on - only read back.
    schema.add_field("bboxes", DataType.JSON)
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSIONS)
    return schema


def connect(path: Path | str | None = None) -> MilvusClient:
    """Open the store, creating the collection on first use.

    `path` is injectable so tests never touch a real index.
    """
    target = Path(path) if path is not None else settings.MILVUS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    client = MilvusClient(uri=str(target))
    if not client.has_collection(settings.MILVUS_COLLECTION):
        index_params = client.prepare_index_params()
        index_params.add_index(
            "embedding",
            index_type=settings.MILVUS_INDEX_TYPE,
            metric_type=settings.MILVUS_METRIC,
        )
        client.create_collection(
            settings.MILVUS_COLLECTION,
            schema=_build_schema(client),
            index_params=index_params,
        )
    return client


def dimension(client: MilvusClient) -> int | None:
    """The vector width this collection was created with."""
    described = client.describe_collection(settings.MILVUS_COLLECTION)
    for field in described.get("fields", []):
        if field.get("name") == "embedding":
            width = (field.get("params") or {}).get("dim")
            return int(width) if width else None
    return None


def assert_usable(client: MilvusClient) -> None:
    """Refuse to continue if the collection cannot hold the configured model's vectors.

    Called at startup. Width is the half of the model guard that lives here,
    because a collection's dimension is fixed at creation and a vector of the
    wrong width cannot be inserted at all.

    The model's *name* is checked separately, against the registry, because
    Milvus Lite does not persist a collection description and the name is
    already recorded per document there.
    """
    width = dimension(client)
    if width is not None and width != settings.EMBEDDING_DIMENSIONS:
        raise EmbedModelMismatchError(
            f"collection holds {width}-dimension vectors, but configured model "
            f"{settings.EMBEDDING_MODEL!r} produces {settings.EMBEDDING_DIMENSIONS}"
        )


def upsert(
    client: MilvusClient,
    doc_id: str,
    chunks: list[Chunk],
    vectors: list[list[float]],
) -> int:
    """Write a document's chunks. Returns how many rows were written.

    Upsert, not insert. The ids are derived from the document and the chunk
    position, so a second ingest of the same file replaces its rows rather than
    adding a parallel set of them.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
    if not chunks:
        return 0

    rows = [
        {
            "chunk_id": chunk_id(doc_id, chunk.index),
            "doc_id": doc_id,
            "page": chunk.page,
            "section_path": chunk.section_path[: settings.MILVUS_SECTION_PATH_MAX],
            "element_type": chunk.element_type,
            "text": chunk.text[: settings.MILVUS_TEXT_MAX],
            "bboxes": [list(box) for box in chunk.bboxes],
            "embedding": vector,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(settings.MILVUS_COLLECTION, rows)
    return len(rows)


def _scope_filter(doc_ids: list[str] | None) -> str:
    """Restrict a search to chosen documents.

    An empty selection is not the same as no selection. No selection means the
    whole library; an empty list would mean the user chose nothing, and that is
    the caller's error to catch, not something to silently widen here.
    """
    if not doc_ids:
        return ""
    quoted = ", ".join(f'"{doc_id}"' for doc_id in doc_ids)
    return f"doc_id in [{quoted}]"


def search(
    client: MilvusClient,
    vector: list[float],
    *,
    doc_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[Hit]:
    """Nearest chunks to a query vector, best first.

    `similarity` on each hit is already oriented so that higher means closer,
    whatever metric is configured. The gate must never see a raw Milvus score.
    """
    results = client.search(
        settings.MILVUS_COLLECTION,
        data=[vector],
        limit=limit or settings.RETRIEVE_CANDIDATES,
        filter=_scope_filter(doc_ids),
        output_fields=[
            "chunk_id",
            "doc_id",
            "page",
            "section_path",
            "element_type",
            "text",
            "bboxes",
        ],
    )
    if not results:
        return []

    hits: list[Hit] = []
    for raw in results[0]:
        entity = raw.get("entity", {})
        distance = float(raw["distance"])
        hits.append(
            Hit(
                chunk_id=entity["chunk_id"],
                doc_id=entity["doc_id"],
                page=int(entity["page"]),
                section_path=entity["section_path"],
                element_type=entity["element_type"],
                text=entity["text"],
                bboxes=[tuple(box) for box in _as_boxes(entity.get("bboxes"))],
                similarity=_similarity(distance),
                raw_distance=distance,
            )
        )
    return hits


def _similarity(distance: float) -> float:
    """Turn whatever Milvus returned into "higher is closer".

    With the cosine metric Milvus already reports similarity in the field it
    calls `distance`: an identical vector scores +1.0, an unrelated one 0.0, an
    opposite one -1.0. Measured against a live collection, not assumed.

    For a true distance metric such as L2, near is a small number, so the sign
    has to be turned around before anything compares it to a threshold.
    """
    if settings.MILVUS_METRIC in ("COSINE", "IP"):
        return distance
    return -distance


def _as_boxes(value) -> list:
    """Boxes come back as JSON, which may already be decoded."""
    if not value:
        return []
    return json.loads(value) if isinstance(value, str) else value


def delete_document(client: MilvusClient, doc_id: str) -> None:
    """Remove every chunk of one document.

    Used when a half-finished ingest is rolled back, and when a document is
    removed for good. A partially indexed document is worse than an absent one,
    because it answers some questions and silently skips the rest.
    """
    client.delete(settings.MILVUS_COLLECTION, filter=_scope_filter([doc_id]))


def count(client: MilvusClient, doc_id: str | None = None) -> int:
    """How many chunks are stored, in total or for one document."""
    rows = client.query(
        settings.MILVUS_COLLECTION,
        filter=_scope_filter([doc_id] if doc_id else None),
        output_fields=["count(*)"],
    )
    return int(rows[0]["count(*)"]) if rows else 0


def chunk_ids(client: MilvusClient, doc_id: str) -> list[str]:
    """Every stored chunk id for one document, in order. For verifying a write."""
    rows = client.query(
        settings.MILVUS_COLLECTION,
        filter=_scope_filter([doc_id]),
        output_fields=["chunk_id"],
        limit=16384,
    )
    return sorted((row["chunk_id"] for row in rows), key=lambda value: int(value.split(":")[-1]))
