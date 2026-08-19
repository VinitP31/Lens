"""Find the chunks most likely to answer a question.

Retrieval only - no gate, no generation - which is what lets it be measured alone.

Scope is resolved against the registry, not taken on trust: the vector store has no
idea a document was deleted. Twelve are fetched to pass five on, because overlap
makes neighbours near-duplicates, and those duplicates are then collapsed.
"""

from dataclasses import dataclass

from backend.ingestion import embedder
from backend.storage import registry, vector_store
from backend.storage.vector_store import Hit
from config import settings


@dataclass(frozen=True)
class Retrieved:
    """What retrieval produced, and enough of the working to explain it.

    The counts and the top score are carried deliberately: every answer reports
    diagnostics, and the gate threshold is calibrated from these scores, so they
    must be observable rather than reconstructed later.
    """

    hits: list[Hit]
    candidates_fetched: int
    duplicates_removed: int
    scope: list[str] | None

    @property
    def top_similarity(self) -> float | None:
        """The best score, or None when nothing was found at all.

        None is not zero. An empty library and a terrible match are different
        situations, and the gate treats them the same way but the diagnostics
        should not pretend they are identical.
        """
        return self.hits[0].similarity if self.hits else None


def _live_scope(connection, doc_ids: list[str] | None) -> tuple[list[str] | None, bool]:
    """Turn a requested scope into one that excludes deleted documents.

    Whole-library is `None`, meaning no filter, which is cheaper than listing every
    id. But a deleted document's vectors are still in the collection, so once
    anything has been deleted it must become an explicit list of the live ones.
    """
    live = [document.doc_id for document in registry.list_documents(connection, ready_only=True)]

    if doc_ids is None:
        deleted_exist = len(registry.list_documents(connection)) != len(live) or _has_deleted(
            connection
        )
        return (live if deleted_exist else None), bool(live)

    # An explicit selection is narrowed to what is actually searchable. A
    # selection of nothing but deleted documents searches nothing, rather than
    # silently widening to the whole library.
    permitted = [doc_id for doc_id in doc_ids if doc_id in set(live)]
    return permitted, bool(permitted)


def _has_deleted(connection) -> bool:
    row = connection.execute(
        "SELECT 1 FROM documents WHERE deleted_at IS NOT NULL LIMIT 1"
    ).fetchone()
    return row is not None


def _longest_shared_run(first: str, second: str) -> int:
    """Length, in words, of the longest run of consecutive words both texts share.

    This is what the overlap window produces: whole sentences copied from one chunk
    into the next. Counting shared vocabulary instead would collapse two short chunks
    that use ordinary words to state different facts.
    """
    left = first.split()
    right = second.split()
    if not left or not right:
        return 0

    # Longest common substring over words, one row at a time: only the previous
    # row is ever needed, so this stays linear in memory.
    previous = [0] * (len(right) + 1)
    best = 0
    for word in left:
        current = [0] * (len(right) + 1)
        for index, other_word in enumerate(right, start=1):
            if word == other_word:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def _deduplicate(hits: list[Hit]) -> tuple[list[Hit], int]:
    """Collapse chunks that are the same passage twice, keeping the better score.

    Same document and same page only: two documents stating the same rule are two
    sources, while two neighbours sharing an overlap window are one counted twice.
    Hits arrive best-first, so the survivor is the higher-scoring one.
    """
    kept: list[Hit] = []
    removed = 0

    for hit in hits:
        duplicate = any(
            existing.doc_id == hit.doc_id
            and existing.page == hit.page
            and _longest_shared_run(existing.text, hit.text) >= settings.DEDUPE_MIN_SHARED_WORDS
            for existing in kept
        )
        if duplicate:
            removed += 1
        else:
            kept.append(hit)

    return kept, removed


def retrieve(
    connection,
    store,
    question: str,
    *,
    doc_ids: list[str] | None = None,
    embed=None,
    limit: int | None = None,
) -> Retrieved:
    """The chunks most likely to answer `question`.

    `doc_ids` of None means the whole library; an empty list means the caller selected
    nothing, and returns nothing rather than searching everything.

    `embed` is injectable, so tests and the evaluation supply their own.
    """
    if doc_ids is not None and not doc_ids:
        return Retrieved(hits=[], candidates_fetched=0, duplicates_removed=0, scope=[])

    scope, searchable = _live_scope(connection, doc_ids)
    if not searchable:
        return Retrieved(hits=[], candidates_fetched=0, duplicates_removed=0, scope=scope)

    query_vector = embedder.embed_query(question, embed=embed)
    candidates = vector_store.search(
        store,
        query_vector,
        doc_ids=scope,
        limit=settings.RETRIEVE_CANDIDATES,
    )

    deduped, removed = _deduplicate(candidates)
    wanted = limit or settings.CONTEXT_CHUNKS

    return Retrieved(
        hits=deduped[:wanted],
        candidates_fetched=len(candidates),
        duplicates_removed=removed,
        scope=scope,
    )
