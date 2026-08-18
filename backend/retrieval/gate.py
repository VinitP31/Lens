"""The confidence gate: decide whether to answer at all, before any LLM call.

It compares `Hit.similarity`, which the vector store has already turned the right
way up, never a raw Milvus score.

It does not catch everything. Measured, it cannot stop an on-topic question whose
answer is absent - similarity measures what a passage is about, not what it
contains. The prompt refuses those.
"""

from dataclasses import dataclass

from backend.retrieval.retriever import Retrieved
from config import settings

# Why a question was refused. Stable strings: the UI maps them to wording, and
# the trace log counts them, so neither may depend on a sentence.
REASON_NO_DOCUMENTS = "no_documents"
REASON_NO_MATCHES = "no_matches"
REASON_BELOW_THRESHOLD = "below_threshold"


@dataclass(frozen=True)
class Decision:
    """Whether to answer, and the numbers behind it.

    The score and threshold travel with the decision so every answer can report
    its own margin, and so a suspicious refusal can be explained without
    re-running the query.
    """

    passed: bool
    reason: str | None
    top_similarity: float | None
    threshold: float

    @property
    def margin(self) -> float | None:
        """How far above the threshold the best chunk was. Negative when refused."""
        if self.top_similarity is None:
            return None
        return self.top_similarity - self.threshold


def evaluate(retrieved: Retrieved, threshold: float | None = None) -> Decision:
    """Decide whether the retrieved chunks are worth answering from.

    `threshold` is injectable so the calibration script can sweep values without
    editing settings, and so a test can state the number it depends on.
    """
    limit = settings.GATE_THRESHOLD if threshold is None else threshold

    # Nothing searchable at all: an empty library, or a scope that resolved to
    # nothing. Distinguished from a poor match because the honest message is
    # different - one asks the user to upload or widen, the other says the
    # documents do not cover it.
    if retrieved.scope is not None and not retrieved.scope:
        return Decision(
            passed=False,
            reason=REASON_NO_DOCUMENTS,
            top_similarity=None,
            threshold=limit,
        )

    if not retrieved.hits:
        return Decision(
            passed=False,
            reason=REASON_NO_MATCHES,
            top_similarity=None,
            threshold=limit,
        )

    top = retrieved.hits[0].similarity
    if top < limit:
        return Decision(
            passed=False,
            reason=REASON_BELOW_THRESHOLD,
            top_similarity=top,
            threshold=limit,
        )

    return Decision(passed=True, reason=None, top_similarity=top, threshold=limit)
