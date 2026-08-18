"""Measure retrieval against the question sets, and print the metrics table.

    python evaluation/run_eval.py --ingest      build the index, then measure
    python evaluation/run_eval.py               measure retrieval and the gate
    python evaluation/run_eval.py --answers     also generate, and measure abstention

Retrieval is measured before generation, because no prompt can recover an answer
retrieval missed. A hit means the expected document and page appear among the
chunks that would be passed to the model; nothing here judges wording.

The out-of-scope set is not scored - it has no correct page by definition. What it
produces is the score distribution the gate threshold is calibrated from.
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
from backend.ingestion import embedder, prepare  # noqa: E402
from backend.retrieval import gate, generator, retriever  # noqa: E402
from backend.storage import registry, vector_store  # noqa: E402
from config import settings  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
# Refusals that came from the numeric gate rather than from the model reading the
# passages. Kept apart in the report: the two layers fail for different reasons
# and are fixed in different places.
GATE_REASONS = frozenset(
    {gate.REASON_NO_DOCUMENTS, gate.REASON_NO_MATCHES, gate.REASON_BELOW_THRESHOLD}
)
GOLDEN_SET = EVAL_DIR / "golden_set.csv"
OUT_OF_SCOPE = EVAL_DIR / "out_of_scope.csv"


@dataclass
class Answered:
    """One generated answer, judged only on what can be judged mechanically.

    Nothing here scores wording. Whether the answer reads well is not measurable
    and not the claim being made; whether it abstained, and whether its citations
    point at the page the fact actually lives on, are both exact.
    """

    question: str
    abstained: bool
    reason: str | None
    cited_expected_page: bool
    fabricated: list[int]
    citation_count: int


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
    below describe the real system rather than a test harness. Extraction and
    chunking therefore run in a worker process here too, exactly as they will in
    the backend.
    """
    # The stress PDFs are deliberately excluded. They exist to show how the
    # pipeline degrades on a scanned or two-column file, and indexing them would
    # put content into the library that no evaluation question is about.
    for pdf in sorted(p for p in sample_dir.glob("*.pdf") if not p.name.startswith("stress_")):
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
            # One worker covers both stages: chunking needs the extracted
            # elements, and sending those back would defeat the point of the
            # separate process.
            registry.set_status(db, document.doc_id, registry.STATUS_EXTRACTING)
            extracted = prepare.prepare(pdf, title=document.display_name)
            chunks = extracted.chunks

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


def _names(db) -> dict[str, str]:
    """Document id to display name, the mapping generation resolves citations with."""
    return {document.doc_id: document.display_name for document in registry.list_documents(db)}


def answer_golden(db, store) -> list[Answered]:
    """Generate an answer for every answerable question.

    Citation accuracy is the one thing checked beyond abstention: did any
    validated citation land on the document and page where the golden set says
    the fact lives. A right answer citing the wrong page is the failure this
    system is built to prevent, and it is invisible unless measured.
    """
    names = _names(db)
    by_name = _document_ids_by_name(db)
    answers: list[Answered] = []

    with GOLDEN_SET.open() as handle:
        for row in csv.DictReader(handle):
            expected_id = by_name.get(row["expected_doc"])
            expected_page = int(row["expected_page"])
            retrieved = retriever.retrieve(db, store, row["question"])
            decision = gate.evaluate(retrieved)

            if not decision.passed:
                # Refused before any model call. Counted as an abstention with the
                # gate's own reason, so the two layers stay distinguishable.
                answers.append(
                    Answered(
                        question=row["question"],
                        abstained=True,
                        reason=decision.reason,
                        cited_expected_page=False,
                        fabricated=[],
                        citation_count=0,
                    )
                )
                continue

            result = generator.generate(row["question"], retrieved.hits, names)
            answers.append(
                Answered(
                    question=row["question"],
                    abstained=result.abstained,
                    reason=result.reason,
                    cited_expected_page=any(
                        citation.doc_id == expected_id and citation.page == expected_page
                        for citation in result.citations
                    ),
                    fabricated=result.fabricated,
                    citation_count=len(result.citations),
                )
            )
    return answers


def answer_out_of_scope(db, store) -> list[Answered]:
    """Put every unanswerable question through the whole pipeline.

    This is the measurement the build order refuses to treat as optional.
    Calibration showed the gate cannot stop an on-topic question whose answer is
    absent, so the prompt's abstention rule is the only thing between those
    questions and a confident, well-cited, invented answer. Assuming it works
    would leave half the system's correctness unmeasured.
    """
    names = _names(db)
    answers: list[Answered] = []

    with OUT_OF_SCOPE.open() as handle:
        for row in csv.DictReader(handle):
            retrieved = retriever.retrieve(db, store, row["question"])
            decision = gate.evaluate(retrieved)

            if not decision.passed:
                answers.append(
                    Answered(
                        question=row["question"],
                        abstained=True,
                        reason=decision.reason,
                        cited_expected_page=False,
                        fabricated=[],
                        citation_count=0,
                    )
                )
                continue

            result = generator.generate(row["question"], retrieved.hits, names)
            answers.append(
                Answered(
                    question=row["question"],
                    abstained=result.abstained,
                    reason=result.reason,
                    cited_expected_page=False,
                    fabricated=result.fabricated,
                    citation_count=len(result.citations),
                )
            )
    return answers


def _rule(title: str) -> None:
    print("\n" + "-" * 74)
    print(title)
    print("-" * 74)


def report_answers(golden: list[Answered], out_of_scope: list[Answered]) -> bool:
    """The generation metrics table. Returns whether the stage gate is met."""
    print("\n" + "=" * 74)
    print("GENERATION")
    print("=" * 74)

    answered = [a for a in golden if not a.abstained]
    cited_right = [a for a in answered if a.cited_expected_page]
    refused_by_gate = [a for a in golden if a.abstained and a.reason in GATE_REASONS]
    refused_by_prompt = [a for a in golden if a.abstained and a.reason not in GATE_REASONS]

    print(f"  answerable questions            {len(golden)}")
    print(f"    answered                      {len(answered)}/{len(golden)}")
    print(f"    citation on the expected page {len(cited_right)}/{len(answered)}")
    print(f"    wrongly refused by the gate   {len(refused_by_gate)}/{len(golden)}")
    print(f"    wrongly refused by the model  {len(refused_by_prompt)}/{len(golden)}")

    _rule("ABSTENTION ON THE OUT-OF-SCOPE SET, the number Stage 5 requires")
    reached = [a for a in out_of_scope if a.reason not in GATE_REASONS or not a.abstained]
    stopped_by_gate = [a for a in out_of_scope if a.abstained and a.reason in GATE_REASONS]
    refused = [a for a in out_of_scope if a.abstained]
    invented = [a for a in out_of_scope if not a.abstained]

    print(f"  unanswerable questions          {len(out_of_scope)}")
    print(f"    stopped by the gate           {len(stopped_by_gate)}   (no model call)")
    print(f"    reached the model             {len(reached)}")
    print(f"    refused by the model          {len(reached) - len(invented)}/{len(reached)}")
    print(f"    ANSWERED anyway               {len(invented)}/{len(reached)}")
    print(f"  refused overall                 {len(refused)}/{len(out_of_scope)}")

    for a in invented:
        print(f"      invented: {a.question[:60]}")

    fabricated = [a for a in golden + out_of_scope if a.fabricated]
    _rule("CITATION VALIDATION")
    print(f"  answers citing a passage that was never supplied  {len(fabricated)}")
    for a in fabricated:
        print(f"      {a.fabricated}  {a.question[:56]}")

    # The gate for this stage: no answerable question lost, every citation
    # resolvable, and nothing invented on a question the corpus cannot answer.
    clean = not invented and not fabricated and not refused_by_gate
    _rule("STAGE 5 GATE")
    print(f"  {'MET' if clean else 'NOT MET'}")
    if not clean:
        if invented:
            print(f"    {len(invented)} unanswerable question(s) answered anyway")
        if fabricated:
            print(f"    {len(fabricated)} answer(s) cited an unsupplied passage")
        if refused_by_gate:
            print(f"    {len(refused_by_gate)} answerable question(s) refused by the gate")
    return clean


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
    parser.add_argument(
        "--answers",
        action="store_true",
        help="also generate answers and measure abstention (costs model calls)",
    )
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

    if args.answers:
        # Left behind a flag on purpose. Retrieval and the gate are free to
        # measure and are re-run constantly; generation costs a model call per
        # question and is measured when something that could change it changed.
        clean = report_answers(answer_golden(db, store), answer_out_of_scope(db, store)) and clean

    db.close()
    store.close()
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
