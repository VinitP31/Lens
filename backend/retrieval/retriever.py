"""Find the chunks most likely to answer a question.

Retrieval only. No gate, no generation, no judgement about whether the chunks are
good enough - those are separate stages, and keeping them apart is what lets this
one be measured on its own.

Three things happen here that are not just "search":

Scope is resolved against the registry, not taken on trust. The vector store has
no idea a document was deleted; that lives in SQLite. Searching without asking
means a soft-deleted document keeps answering questions.

Twelve are fetched to pass five on. Overlap deliberately makes neighbouring
chunks near-duplicates, so asking for exactly five spends slots on repeats of one
passage.

Near-duplicates are collapsed. Two chunks sharing an overlap window are the same
answer twice, and the second one has displaced a different passage that might
have completed it.
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

    Returns the scope to search and whether searching is worthwhile at all.

    Whole-library is expressed as `None`, meaning no filter, which is cheaper
    than listing every id. But a deleted document's vectors are still in the
    collection, so whole-library has to become an explicit list of the live ones
    as soon as anything has been deleted - otherwise a removed document goes on
    answering questions.
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

    This models what the overlap window actually does: it copies whole sentences
    from the end of one chunk to the start of the next, so a duplicate pair
    contains one long identical run rather than merely resembling each other.

    Counting shared vocabulary instead would collapse two short chunks that use
    the same ordinary words to state completely different facts.
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

    Only chunks from the same document and the same page are compared. Two
    different documents stating the same rule are genuinely two sources and both
    deserve to be cited; two neighbours inside one page sharing an overlap window
    are one source counted twice.

    Hits arrive best-first, so the survivor is always the higher-scoring one.
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

    `doc_ids` of None means the whole library. An empty list means the caller
    selected nothing, which returns nothing rather than quietly searching
    everything.

    `embed` is injectable so the evaluation script and the tests can supply their
    own, and so a query is never embedded by a different model than the documents.
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
