"""The extraction worker, run as its own program.

    python -m backend.ingestion.worker <pdf> <title> <output>

Reads a PDF, chunks it, and writes the result to `output` as a pickle. Docling is
imported here and nowhere else in a process that also holds the vector store,
because the two cannot coexist - see `prepare` for that story.

Why a separate program rather than a multiprocessing child: a child started from
inside Python inherits the parent's open file descriptors, and the backend's
parent holds a live gRPC connection to the vector store. Measured on this machine,
that child dies with SIGTRAP and an empty stderr - no traceback, nothing to read,
just a document that never indexes. A subprocess launched with its descriptors
closed shares nothing and cannot be poisoned by whatever the parent happens to
have open.

Output goes to a file rather than to stdout on purpose. Docling writes progress and
warnings to both streams, so anything sharing them would arrive mixed with model
chatter.
"""

import pickle
import sys
import time
from pathlib import Path


def main(argv: list[str]) -> int:
    """Extract and chunk one document. Returns a process exit code.

    Any failure is written to the output file rather than raised, so the parent
    reports the real reason instead of only an exit code.
    """
    if len(argv) != 3:
        print("usage: worker <pdf> <title> <output>", file=sys.stderr)
        return 2

    pdf_path, title, output = argv
    started = time.perf_counter()

    try:
        # Imported here, not at module scope: the import itself is the expensive,
        # incompatible thing this program exists to isolate.
        from backend.ingestion import chunker, extractor
        from backend.ingestion.prepare import Prepared

        extracted = extractor.extract(Path(pdf_path))
        prepared = Prepared(
            chunks=chunker.chunk(extracted, title=title),
            page_count=extracted.page_count,
            table_count=extracted.table_count,
            picture_count=extracted.picture_count,
            chars_per_page=extracted.chars_per_page,
            needs_ocr=extracted.needs_ocr,
            seconds=time.perf_counter() - started,
        )
        Path(output).write_bytes(pickle.dumps(("ok", prepared)))
        return 0
    except Exception as error:  # noqa: BLE001 - handed to the parent to decide
        try:
            Path(output).write_bytes(pickle.dumps(("error", error)))
        except Exception:  # noqa: BLE001 - nothing left to do but exit non-zero
            print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
