"""Tests for the trace log.

Two things are protected. Both scores are written for every retrieved chunk, because
confusing a Milvus distance with a similarity is the most expensive mistake available
here. And a trace never breaks the thing it describes, so an unwritable path is
swallowed.
"""

import json

from backend.logging import trace
from backend.retrieval.gate import Decision
from backend.retrieval.retriever import Retrieved
from backend.storage.vector_store import Hit
from config import settings


def hit(chunk_id: str = "doc1:4", similarity: float = 0.62) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        doc_id="doc1",
        page=17,
        section_path="4.2 Leave",
        element_type="text",
        text="Leave accrues monthly.",
        bboxes=[(1.0, 2.0, 3.0, 4.0)],
        similarity=similarity,
        raw_distance=similarity,
    )


def written(tmp_path, monkeypatch) -> list[dict]:
    return [
        json.loads(line)
        for line in (tmp_path / "queries.jsonl").read_text().splitlines()
        if line.strip()
    ]


def use_tmp(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "QUERY_TRACE_PATH", tmp_path / "queries.jsonl")
    monkeypatch.setattr(settings, "DOCUMENT_TRACE_PATH", tmp_path / "documents.jsonl")


# --- what a query trace holds --------------------------------------------


def test_a_query_is_written_as_one_line(tmp_path, monkeypatch):
    use_tmp(tmp_path, monkeypatch)
    trace.write_query(trace.QueryTrace(conv_id="c1", message="how much leave?"))
    trace.write_query(trace.QueryTrace(conv_id="c1", message="and part-time?"))

    lines = written(tmp_path, monkeypatch)
    assert len(lines) == 2
    assert lines[0]["message"] == "how much leave?"


def test_both_scores_are_recorded_for_every_chunk(tmp_path, monkeypatch):
    """Every retrieved chunk records both its similarity and its raw score. Milvus names its
    score `distance` while cosine makes it a similarity, and reading it the wrong
    way up builds a system that answers on nonsense and refuses real questions."""
    use_tmp(tmp_path, monkeypatch)
    record = trace.QueryTrace(conv_id="c1", message="a question")
    record.record_retrieval(
        Retrieved(hits=[hit()], candidates_fetched=12, duplicates_removed=1, scope=None)
    )
    trace.write_query(record)

    chunk = written(tmp_path, monkeypatch)[0]["retrieved"][0]
    assert "similarity" in chunk
    assert "raw_distance" in chunk


def test_the_chunks_sent_to_the_model_are_recorded(tmp_path, monkeypatch):
    """So the prompt can be rebuilt exactly rather than approximately."""
    use_tmp(tmp_path, monkeypatch)
    record = trace.QueryTrace(conv_id="c1", message="a question")
    record.record_retrieval(
        Retrieved(
            hits=[hit("doc1:4"), hit("doc2:9")],
            candidates_fetched=12,
            duplicates_removed=0,
            scope=None,
        )
    )
    trace.write_query(record)

    assert written(tmp_path, monkeypatch)[0]["sent_chunk_ids"] == ["doc1:4", "doc2:9"]


def test_a_refusal_records_the_numbers_behind_it(tmp_path, monkeypatch):
    """A surprising refusal has to be explainable without asking again."""
    use_tmp(tmp_path, monkeypatch)
    record = trace.QueryTrace(conv_id="c1", message="a question")
    record.record_gate(
        Decision(passed=False, reason="below_threshold", top_similarity=0.31, threshold=0.45)
    )
    trace.write_query(record)

    line = written(tmp_path, monkeypatch)[0]
    assert line["gate_passed"] is False
    assert line["gate_reason"] == "below_threshold"
    assert line["top_similarity"] == 0.31
    assert line["gate_threshold"] == 0.45


def test_discarded_citations_are_recorded(tmp_path, monkeypatch):
    """This is the number that says whether the model has started inventing
    sources, so it has to be visible without re-running anything."""
    use_tmp(tmp_path, monkeypatch)

    class FakeAnswer:
        text = "Stated in [1]."
        abstained = False
        reason = None
        partly_absent = False
        fabricated = [7]

        class _Citation:
            number = 1

        citations = [_Citation()]

    record = trace.QueryTrace(conv_id="c1", message="a question")
    record.record_answer(FakeAnswer())
    trace.write_query(record)

    line = written(tmp_path, monkeypatch)[0]
    assert line["kept_citations"] == [1]
    assert line["discarded_citations"] == [7]
    assert line["cited"] == [1, 7]


def test_what_was_actually_searched_is_recorded(tmp_path, monkeypatch):
    """A surprising answer is usually explained by the difference between what
    was typed and what was searched."""
    use_tmp(tmp_path, monkeypatch)
    trace.write_query(
        trace.QueryTrace(
            conv_id="c1",
            message="and for part-time?",
            rewritten="What leave do part-time staff get?",
        )
    )

    assert written(tmp_path, monkeypatch)[0]["rewritten"] == ("What leave do part-time staff get?")


# --- documents -----------------------------------------------------------


def test_a_document_trace_records_the_stages(tmp_path, monkeypatch):
    use_tmp(tmp_path, monkeypatch)
    record = trace.DocumentTrace(doc_id="d1", display_name="Handbook.pdf")
    record.stage_ms = {"extract": 3000, "embed": 400, "index": 60}
    record.chunk_count = 109
    trace.write_document(record)

    line = json.loads((tmp_path / "documents.jsonl").read_text().splitlines()[0])
    assert line["stage_ms"]["extract"] == 3000
    assert line["chunk_count"] == 109


def test_a_failed_ingest_is_recorded_with_the_stage_it_died_on(tmp_path, monkeypatch):
    """The document row is deleted by the rollback, so this line is the only
    record that it was ever attempted."""
    use_tmp(tmp_path, monkeypatch)
    record = trace.DocumentTrace(doc_id="d1", display_name="Broken.pdf")
    record.failed_at = "embedding"
    record.error = "EmbeddingFailedError: provider down"
    trace.write_document(record)

    line = json.loads((tmp_path / "documents.jsonl").read_text().splitlines()[0])
    assert line["failed_at"] == "embedding"
    assert "provider down" in line["error"]


# --- never breaking the request it describes -----------------------------


def test_an_unwritable_path_does_not_raise(tmp_path, monkeypatch):
    """A diagnostic that can fail the request it was describing is worse than no
    diagnostic at all."""
    monkeypatch.setattr(settings, "QUERY_TRACE_PATH", tmp_path / "a-file" / "queries.jsonl")
    (tmp_path / "a-file").write_text("not a directory")

    trace.write_query(trace.QueryTrace(conv_id="c1", message="a question"))


def test_reading_back_skips_a_half_written_line(tmp_path, monkeypatch):
    """The file is appended to by a live process, so the last line can be
    incomplete when it is read."""
    use_tmp(tmp_path, monkeypatch)
    trace.write_query(trace.QueryTrace(conv_id="c1", message="complete"))
    with (tmp_path / "queries.jsonl").open("a") as handle:
        handle.write('{"conv_id": "c2", "mess')

    records = trace.read_queries()
    assert len(records) == 1
    assert records[0]["message"] == "complete"


def test_reading_back_an_absent_file_is_empty(tmp_path, monkeypatch):
    use_tmp(tmp_path, monkeypatch)

    assert trace.read_queries() == []
