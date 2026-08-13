"""Measure retrieval against the question sets, and print the metrics table.

    python evaluation/run_eval.py --ingest      build the index, then measure
    python evaluation/run_eval.py               measure against an existing index

Retrieval is measured before generation exists, on purpose. If retrieval misses,
no prompt can recover the answer, so a generation metric would only tell you the
model is fluent - not that the system is right.

A hit means the expected document *and* the expected page appear among the chunks
that would be passed to the model. Nothing here judges wording; the golden set
records where each answer lives, and that is what gets checked.

The out-of-scope set is not scored here. It has no correct page by definition.
What it produces is a score distribution, and the separation between the two
distributions is what the gate threshold is calibrated from at the next stage.
"""

import argparse
import csv
import hashlib
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from backend.errors import DuplicateDocumentError  # noqa: E402
from backend.retrieval import retriever  # noqa: E402
from backend.storage import registry, vector_store  # noqa: E402

# The extractor is imported inside ingest_corpus rather than here. It pulls in
# Docling and PyTorch, whose threading runtime conflicts with Milvus Lite's on
# macOS and aborts the process. A measuring run needs neither, and the query path
# in the application does not import them either.
from config import settings  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_SET = EVAL_DIR / "golden_set.csv"
OUT_OF_SCOPE = EVAL_DIR / "out_of_scope.csv"


@dataclass
class Result:
    question: str
    expected_doc: str
    expected_page: int
    hit_at_k: bool
    hit_doc_only: bool
    rank: int | None
    top_similarity: float | None
    pages_returned: list[int]
    docs_returned: list[str]


def ingest_corpus(db, store, sample_dir: Path) -> None:
    """Index every PDF in `sample_dir`, skipping ones already present.

    Uses the same modules as the application, in the same order, so the numbers
    below describe the real system rather than a test harness.
    """
    from backend.ingestion import chunker, embedder, extractor

    for pdf in sorted(sample_dir.glob("*.pdf")):
        digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
        if registry.find_by_hash(db, digest):
            print(f"  already indexed  {pdf.name}")
            continue

        started = time.perf_counter()
        try:
            document = registry.register(
                db,
                original_filename=pdf.name,
                content_hash=digest,
                size_bytes=pdf.stat().st_size,
                file_path=str(pdf),
            )
        except DuplicateDocumentError:
            continue

        try:
            registry.set_status(db, document.doc_id, registry.STATUS_EXTRACTING)
            extracted = extractor.extract(pdf)

            registry.set_status(db, document.doc_id, registry.STATUS_CHUNKING)
            chunks = chunker.chunk(extracted, title=document.display_name)

            registry.set_status(db, document.doc_id, registry.STATUS_EMBEDDING)
            vectors = embedder.embed_chunks(chunks)

            registry.set_status(db, document.doc_id, registry.STATUS_INDEXING)
            written = vector_store.upsert(store, document.doc_id, chunks, vectors)

            registry.mark_ready(
                db,
                document.doc_id,
                page_count=extracted.page_count,
                chunk_count=written,
                table_count=extracted.table_count,
                image_count=extracted.picture_count,
                chars_per_page=extracted.chars_per_page,
                ocr_applied=extracted.needs_ocr,
            )
        except Exception:
            # A half-indexed document answers some questions and silently skips
            # the rest of its own content, so it is removed rather than kept.
            vector_store.delete_document(store, document.doc_id)
            registry.discard(db, document.doc_id)
            raise

        print(
            f"  indexed          {pdf.name}  "
            f"{extracted.page_count} pages, {written} chunks, "
            f"{time.perf_counter() - started:.1f}s"
        )


def _document_ids_by_name(db) -> dict[str, str]:
    return {document.display_name: document.doc_id for document in registry.list_documents(db)}


def measure_golden(db, store) -> list[Result]:
    by_name = _document_ids_by_name(db)
    results: list[Result] = []

    with GOLDEN_SET.open() as handle:
        for row in csv.DictReader(handle):
            expected_doc = row["expected_doc"]
            expected_page = int(row["expected_page"])
            retrieved = retriever.retrieve(db, store, row["question"])

            expected_id = by_name.get(expected_doc)
            pages = [hit.page for hit in retrieved.hits]
            docs = [hit.doc_id for hit in retrieved.hits]

            rank = None
            for position, hit in enumerate(retrieved.hits, start=1):
                if hit.doc_id == expected_id and hit.page == expected_page:
                    rank = position
                    break

            results.append(
                Result(
                    question=row["question"],
                    expected_doc=expected_doc,
                    expected_page=expected_page,
                    hit_at_k=rank is not None,
                    hit_doc_only=expected_id in docs,
                    rank=rank,
                    top_similarity=retrieved.top_similarity,
                    pages_returned=pages,
                    docs_returned=docs,
                )
            )
    return results


def measure_out_of_scope(db, store) -> list[tuple[str, float | None]]:
    """Top similarity for each question the corpus cannot answer.

    Not scored. These produce the distribution the gate threshold is calibrated
    against, and every one of them should score below every golden question.
    """
    scores: list[tuple[str, float | None]] = []
    with OUT_OF_SCOPE.open() as handle:
        for row in csv.DictReader(handle):
            retrieved = retriever.retrieve(db, store, row["question"])
            scores.append((row["question"], retrieved.top_similarity))
    return scores


def report(results: list[Result], out_of_scope: list[tuple[str, float | None]]) -> bool:
    total = len(results)
    hits = sum(1 for result in results if result.hit_at_k)
    top_one = sum(1 for result in results if result.rank == 1)
    doc_only = sum(1 for result in results if result.hit_doc_only)

    rule = "=" * 74
    print(f"\n{rule}")
    print(f"RETRIEVAL, {total} golden questions, top {settings.CONTEXT_CHUNKS}")
    print(rule)
    print(f"  correct document and page   {hits}/{total}  ({hits / total:.0%})")
    print(f"  correct at rank 1           {top_one}/{total}  ({top_one / total:.0%})")
    print(f"  correct document, any page  {doc_only}/{total}  ({doc_only / total:.0%})")

    ranks = [result.rank for result in results if result.rank]
    if ranks:
        print(f"  mean rank when found        {sum(ranks) / len(ranks):.2f}")

    print("\n  per document")
    by_doc: dict[str, list[Result]] = {}
    for result in results:
        by_doc.setdefault(result.expected_doc, []).append(result)
    for name, group in sorted(by_doc.items()):
        found = sum(1 for result in group if result.hit_at_k)
        print(f"    {found}/{len(group)}  {name[:56]}")

    misses = [result for result in results if not result.hit_at_k]
    if misses:
        thin = "-" * 74
        print(f"\n{thin}")
        print(f"MISSES, {len(misses)} of {total} - every one needs an explanation")
        print(thin)
        for result in misses:
            print(f"\n  {result.question}")
            print(f"    expected  {result.expected_doc[:48]} p{result.expected_page}")
            print(f"    returned  pages {result.pages_returned}")
            print(
                f"    document found at all: {result.hit_doc_only}   "
                f"top similarity: {result.top_similarity:+.3f}"
                if result.top_similarity is not None
                else "    nothing returned"
            )

    scored = [score for _question, score in out_of_scope if score is not None]
    golden_scores = [
        result.top_similarity for result in results if result.top_similarity is not None
    ]
    if scored and golden_scores:
        print(
            f"\n{'=' * 74}\nSCORE DISTRIBUTIONS, for gate calibration at the next stage\n{'=' * 74}"
        )
        print(
            f"  golden        min {min(golden_scores):+.3f}   "
            f"mean {sum(golden_scores) / len(golden_scores):+.3f}   max {max(golden_scores):+.3f}"
        )
        print(
            f"  out of scope  min {min(scored):+.3f}   "
            f"mean {sum(scored) / len(scored):+.3f}   max {max(scored):+.3f}"
        )
        separation = min(golden_scores) - max(scored)
        print(f"\n  gap between the lowest good answer and the highest bad one: {separation:+.3f}")
        if separation > 0:
            print("  the two sets do not overlap, so a threshold exists that separates them")
        else:
            print("  the sets OVERLAP, so no single threshold separates them cleanly")
            print("  the calibration stage has to choose which error to prefer")

        print("\n  lowest-scoring golden questions, the ones a threshold would refuse")
        for result in sorted(
            (item for item in results if item.top_similarity is not None),
            key=lambda item: item.top_similarity,
        )[:5]:
            print(f"    {result.top_similarity:+.3f}  {result.question[:62]}")

        print("\n  highest-scoring out-of-scope questions, the ones that will fool the gate")
        for question, score in sorted(
            (pair for pair in out_of_scope if pair[1] is not None),
            key=lambda pair: pair[1],
            reverse=True,
        )[:5]:
            print(f"    {score:+.3f}  {question[:62]}")

        _threshold_table(golden_scores, scored)

    return not misses


def _threshold_table(golden: list[float], out_of_scope: list[float]) -> None:
    """What each candidate threshold would cost, in both directions.

    Both error rates are printed because they trade off against each other and
    the choice between them is a judgement about this system's purpose, not
    something a formula settles. A grounded-answers tool should prefer refusing a
    question it could have answered over answering one it could not.
    """
    print(f"\n{'-' * 74}")
    print("WHAT EACH THRESHOLD WOULD COST")
    print(f"{'-' * 74}")
    print(f"  {'threshold':>10}  {'refuses a real answer':>22}  {'answers an absent one':>22}")

    candidates = [round(value / 100, 2) for value in range(40, 80, 2)]
    for threshold in candidates:
        refused = sum(1 for score in golden if score < threshold)
        answered = sum(1 for score in out_of_scope if score >= threshold)
        marker = ""
        if refused == 0:
            marker = "  <- no real answer lost"
        print(
            f"  {threshold:>10.2f}  {refused:>13}/{len(golden):<8}"
            f"  {answered:>13}/{len(out_of_scope):<8}{marker}"
        )

    print("\n  Note: neither column can reach zero while the distributions overlap.")
    print("  Out-of-scope questions that survive the gate are the prompt's abstention")
    print("  rule to catch, which is where LENS.md already assigns them.")

    configured = settings.GATE_THRESHOLD
    refused = sum(1 for score in golden if score < configured)
    answered = sum(1 for score in out_of_scope if score >= configured)
    print(f"\n{'-' * 74}")
    print(f"THE CONFIGURED THRESHOLD: {configured}")
    print(f"{'-' * 74}")
    print(f"  real answers wrongly refused    {refused}/{len(golden)}")
    print(f"  unanswerable questions stopped  {len(out_of_scope) - answered}/{len(out_of_scope)}")
    print(f"  unanswerable questions passed   {answered}/{len(out_of_scope)}  -> the prompt's job")
    if refused:
        print("\n  WARNING: the gate is refusing questions the corpus answers.")
        print("  Lower the threshold, or accept and record why.")


def main() -> int:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    logging.disable(logging.WARNING)

    parser = argparse.ArgumentParser(description="Measure retrieval.")
    parser.add_argument("--ingest", action="store_true", help="index the corpus first")
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    parser.add_argument("--db", type=Path, default=None, help="registry path")
    parser.add_argument("--store", type=Path, default=None, help="vector store path")
    args = parser.parse_args()

    db = registry.connect(args.db)
    store = vector_store.connect(args.store)
    registry.assert_embed_model(db)
    vector_store.assert_usable(store)

    if args.ingest:
        print("ingesting")
        ingest_corpus(db, store, args.samples)

    documents = registry.list_documents(db, ready_only=True)
    if not documents:
        print("nothing indexed. Run again with --ingest")
        return 1
    print(f"\nlibrary: {len(documents)} documents, {vector_store.count(store)} chunks")

    results = measure_golden(db, store)
    out_of_scope = measure_out_of_scope(db, store)
    clean = report(results, out_of_scope)

    db.close()
    store.close()
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
