"""One line of JSON per query, and one per indexed document.

The reason this exists rather than being deferred: "why did it say that?" should
take seconds, not an afternoon of reproducing the question and hoping it behaves
the same way. Everything needed to answer it is written once, at the moment it is
known, and never reconstructed.

Two fields are here specifically because getting them confused is the most
expensive mistake available in this system. Milvus names its score `distance` for
every metric, but under cosine that field holds a *similarity* - identical is
+1.0, unrelated is 0.0. Reading it the wrong way up builds something that answers
confidently on nonsense and refuses real questions, and it looks exactly like a
prompt bug. So both numbers are logged side by side from the first query, and a
glance at one line settles it.

JSONL rather than a database: it appends without locking, survives a crash
mid-write with the loss of at most one line, and is readable with the tools
already on the machine.

Nothing here may raise. A trace is a diagnostic; a diagnostic that can break the
answer it was describing is worse than no diagnostic.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from config import settings

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass
class QueryTrace:
    """Everything about one turn, in the order it happened.

    Built up as the turn progresses and written once at the end, so a turn that
    failed halfway still records how far it got.
    """

    conv_id: str
    message: str

    at: str = field(default_factory=_now)
    # What was actually searched with, when it differs from what was typed. Both
    # are kept: a surprising answer is usually explained by the difference.
    condensed: str | None = None
    rewritten: str | None = None
    intent: str | None = None
    analysis_degraded: bool = False

    scope_mode: str | None = None
    scope_doc_ids: list[str] | None = None

    # Every candidate, with both numbers. See the module docstring.
    retrieved: list[dict] = field(default_factory=list)
    candidates_fetched: int = 0
    duplicates_removed: int = 0

    gate_passed: bool | None = None
    gate_reason: str | None = None
    gate_threshold: float = settings.GATE_THRESHOLD
    top_similarity: float | None = None

    # What the model was actually shown, so the prompt can be rebuilt exactly.
    sent_chunk_ids: list[str] = field(default_factory=list)

    cited: list[int] = field(default_factory=list)
    kept_citations: list[int] = field(default_factory=list)
    discarded_citations: list[int] = field(default_factory=list)
    abstained: bool | None = None
    abstain_reason: str | None = None
    partly_absent: bool = False

    answer_chars: int = 0
    stage_ms: dict[str, int] = field(default_factory=dict)
    total_ms: int = 0
    error: str | None = None

    def record_retrieval(self, retrieved) -> None:
        """Store the candidates with both scores, and the working behind them."""
        self.candidates_fetched = retrieved.candidates_fetched
        self.duplicates_removed = retrieved.duplicates_removed
        self.retrieved = [
            {
                "chunk_id": hit.chunk_id,
                "doc_id": hit.doc_id,
                "page": hit.page,
                # Both, deliberately. One is what Milvus returned and the other
                # is what the gate compares; a line showing them together is the
                # fastest way to settle an argument about which is which.
                "similarity": round(hit.similarity, 4),
                "raw_distance": round(hit.raw_distance, 4),
            }
            for hit in retrieved.hits
        ]
        self.sent_chunk_ids = [hit.chunk_id for hit in retrieved.hits]

    def record_gate(self, decision) -> None:
        self.gate_passed = decision.passed
        self.gate_reason = decision.reason
        self.gate_threshold = decision.threshold
        self.top_similarity = decision.top_similarity

    def record_answer(self, answer) -> None:
        self.abstained = answer.abstained
        self.abstain_reason = answer.reason
        self.partly_absent = answer.partly_absent
        self.answer_chars = len(answer.text)
        self.kept_citations = [citation.number for citation in answer.citations]
        self.discarded_citations = list(answer.fabricated)
        self.cited = sorted(set(self.kept_citations) | set(self.discarded_citations))


@dataclass
class DocumentTrace:
    """What one ingestion learned and how long each stage took."""

    doc_id: str
    display_name: str

    at: str = field(default_factory=_now)
    page_count: int = 0
    chars_per_page: int = 0
    ocr_applied: bool = False
    heading_count: int = 0
    table_count: int = 0
    image_count: int = 0
    chunk_count: int = 0
    stage_ms: dict[str, int] = field(default_factory=dict)
    total_ms: int = 0
    failed_at: str | None = None
    error: str | None = None


def _write(path: Path, record: dict) -> None:
    """Append one line. Never raises.

    A trace is a diagnostic. One that can break the answer it was describing is
    worse than no trace at all, so a failure here is logged and swallowed.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
    except Exception:  # noqa: BLE001 - a trace must never break a request
        log.exception("could not write trace to %s", path)


def write_query(trace: QueryTrace) -> None:
    _write(settings.QUERY_TRACE_PATH, asdict(trace))


def write_document(trace: DocumentTrace) -> None:
    _write(settings.DOCUMENT_TRACE_PATH, asdict(trace))


def read_queries(limit: int | None = None) -> list[dict]:
    """The most recent query traces, newest last. For reading a log by hand.

    A malformed line is skipped rather than raising: the file is appended to by a
    live process, and the last line can be half-written when it is read.
    """
    path = settings.QUERY_TRACE_PATH
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return records[-limit:] if limit else records
