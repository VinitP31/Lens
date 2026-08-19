"""Run every check in one command.

    python scripts/check.py              lint, format, tests        (fast)
    python scripts/check.py --corpus     also audit the real PDFs   (slow)

The corpus audit exists because a green suite cannot see these problems: every test
uses a generated fixture PDF, while these invariants are checked against the real
documents. Each has caught a real defect at least once.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402

PASS = "  ok  "
FAIL = " FAIL "


def _log_failure(label: str, output: str) -> Path | None:
    """Keep the whole output of a failed check, and say where it went.

    The tail on screen was not enough once: a test failed, would not reproduce, and
    the lines explaining it had scrolled off. Never raises - losing the result of a
    check to a logging error would be the worse failure.
    """
    try:
        settings.CHECK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings.CHECK_LOG_PATH.write_text(f"=== {label} ===\n{output}\n")
        return settings.CHECK_LOG_PATH
    except OSError:
        return None


def _run(label: str, command: list[str]) -> bool:
    """Run a command, print one line, keep its output only if it failed."""
    result = subprocess.run(command, capture_output=True, text=True)
    ok = result.returncode == 0
    print(f"[{PASS if ok else FAIL}] {label}")
    if not ok:
        output = (result.stdout + result.stderr).strip()
        print("\n".join(f"         {line}" for line in output.splitlines()[-25:]))
        written = _log_failure(label, output)
        if written:
            print(f"         full output: {written}")
    return ok


def fast_checks() -> bool:
    python = sys.executable
    return all(
        [
            _run("ruff check", [python, "-m", "ruff", "check", "."]),
            _run("ruff format", [python, "-m", "ruff", "format", "--check", "."]),
            _run("pytest", [python, "-m", "pytest", "tests/", "-q"]),
        ]
    )


def corpus_audit(sample_dir: Path) -> bool:
    """Check the invariants that only real documents can break."""
    logging.disable(logging.WARNING)
    # Imported here rather than at module scope: this pulls in Docling, which the
    # fast checks must not pay for.
    from backend.ingestion import chunker, extractor

    # stress_* files are never indexed and would fail these invariants by design.
    pdfs = sorted(p for p in sample_dir.glob("*.pdf") if not p.name.startswith("stress_"))
    if not pdfs:
        print(f"[{FAIL}] corpus audit: no PDFs in {sample_dir}")
        return False

    def flat(text: str) -> str:
        return " ".join(text.split())

    print(f"\n{'document':<34}{'pages':>6}{'chunks':>7}{'tokens med/max':>16}")
    failures: list[str] = []

    for path in pdfs:
        document = extractor.extract(path)
        chunks = chunker.chunk(document, title=path.stem)
        name = path.stem[:32]

        sizes = sorted(chunk.token_count for chunk in chunks)
        print(
            f"{name:<34}{document.page_count:>6}{len(chunks):>7}"
            f"{f'{sizes[len(sizes) // 2]}/{sizes[-1]}':>16}"
        )

        def note(problem: str, detail: str, document_name: str = name) -> None:
            failures.append(f"{document_name}: {problem} ({detail})")

        # Text must not be lost. Two exceptions are deliberate: a contents page, and
        # a heading, which lives in the section path rather than in a chunk body.
        bodies = " || ".join(flat(chunk.text) for chunk in chunks)
        paths = " || ".join(flat(chunk.section_path) for chunk in chunks)
        lost = [
            element
            for element in document.elements
            if flat(element.text)
            and element.page not in document.contents_pages
            and flat(element.text) not in bodies
            and flat(element.text) not in paths
        ]
        if lost:
            note("text lost", f"{len(lost)} elements, first on p{lost[0].page}")

        # A chunk cites one page, so its text must be on that page.
        on_page: dict[int, str] = {}
        for element in document.elements:
            on_page[element.page] = on_page.get(element.page, "") + " " + flat(element.text)
        mixed = [
            c
            for c in chunks
            if flat(c.text)[:40] and flat(c.text)[:40] not in on_page.get(c.page, "")
        ]
        if mixed:
            note("text filed under the wrong page", f"{len(mixed)} chunks, first p{mixed[0].page}")

        # Citations need coordinates.
        if boxless := [c for c in chunks if not c.bboxes]:
            note("no bounding boxes", f"{len(boxless)} chunks")

        # Sizes must respect the settings, tables excepted by design.
        prose = [c for c in chunks if c.element_type == extractor.TYPE_TEXT]
        if over := [c for c in prose if c.token_count > settings.CHUNK_MAX_TOKENS]:
            note(
                "prose over the ceiling",
                f"{len(over)} chunks, largest {max(c.token_count for c in over)}",
            )
        if huge := [c for c in chunks if c.token_count > settings.EMBED_MAX_INPUT_TOKENS]:
            note("chunk too large to embed", f"{len(huge)} chunks")

        # Nothing empty, and every id sequential from zero.
        if [c for c in chunks if not c.text.strip()]:
            note("empty chunk", "would embed to noise")
        if [c.index for c in chunks] != list(range(len(chunks))):
            note("chunk indexes not sequential", "breaks the deterministic chunk id")

        # Contents pages are deliberately excluded from the index.
        if leaked := [c for c in chunks if c.page in document.contents_pages]:
            note("contents page indexed", f"pages {sorted({c.page for c in leaked})}")

    print()
    if failures:
        print(f"[{FAIL}] corpus audit")
        for failure in failures:
            print(f"         {failure}")
        return False
    print(f"[{PASS}] corpus audit: {len(pdfs)} documents, every invariant holds")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run every check.")
    parser.add_argument(
        "--corpus",
        action="store_true",
        help="also re-extract the sample PDFs and audit them (slow)",
    )
    parser.add_argument("--samples", type=Path, default=Path("samples"))
    args = parser.parse_args()

    ok = fast_checks()
    if args.corpus:
        ok = corpus_audit(args.samples) and ok

    print("\nall checks passed" if ok else "\nsomething failed, see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
