"""Extraction and chunking, run in a process of their own.

The reason this module exists is a hard incompatibility, not a preference.

Docling brings its own copy of the OpenMP runtime, inside PyTorch. Milvus Lite
brings a second copy, inside FAISS. That runtime refuses to initialise twice in
one process, and when both do it the process dies outright - `OMP: Error #15`
and an abort, or a bare segmentation fault, with no Python traceback to read.

It only bites once the vector index has actually been loaded, which is why an
empty library ingests happily and a library with documents already in it does
not. In other words the failure is invisible on a first run and certain on every
run after that. Adding a second document is the ordinary case, so this is not an
edge to note and move past.

There is a documented environment variable that suppresses the abort. Its own
documentation says it may silently produce incorrect results, which is worse
than a crash: a crash is visible. Making the two copies into one by hand is a
per-machine change that would not survive a fresh install by anyone else.

So the two never meet. Docling is imported only in the child process, and the
parent keeps the vector store. `Prepared` and `Chunk` are plain dataclasses over
built-in types, so they cross the process boundary without either side needing
to import the other's libraries.

This also makes the separation the backend rules already ask for literal rather
than aspirational: ingestion is a batch job, the query path is interactive, and
now they cannot even share an address space by accident.

One thing a caller has to know. "spawn" re-imports the module that started the
process, so any script that calls this must keep its work behind
`if __name__ == "__main__":`. Without that guard the child re-runs the script
from the top - opening the vector store a second time, ingesting again - and dies
in a way that surfaces only as a worker exit code. Running under a server does
not have this problem, because the main module is the server's own.
"""

import multiprocessing
import queue
import time
from dataclasses import dataclass
from pathlib import Path

from backend.errors import ExtractionFailedError
from backend.ingestion.chunk import Chunk
from config import settings


@dataclass(frozen=True)
class Prepared:
    """Everything the parent needs from a document, and nothing that cannot cross.

    Deliberately not `ExtractedDocument`: that carries `Element`, which is
    defined beside the Docling imports, so unpickling one would load Docling into
    the very process this module exists to keep it out of.
    """

    chunks: list[Chunk]
    page_count: int
    table_count: int
    picture_count: int
    chars_per_page: int
    needs_ocr: bool
    seconds: float


def _child(pdf_path: Path, title: str, mailbox) -> None:
    """Extract and chunk one document, in the child process.

    Imports happen here rather than at module scope so that a parent importing
    this module does not pull Docling in with it.

    Any failure is sent back rather than raised, so the parent reports the real
    reason instead of a dead worker.
    """
    try:
        from backend.ingestion import chunker, extractor

        extracted = extractor.extract(pdf_path)
        mailbox.put(
            (
                "ok",
                Prepared(
                    chunks=chunker.chunk(extracted, title=title),
                    page_count=extracted.page_count,
                    table_count=extracted.table_count,
                    picture_count=extracted.picture_count,
                    chars_per_page=extracted.chars_per_page,
                    needs_ocr=extracted.needs_ocr,
                    seconds=extracted.seconds,
                ),
            )
        )
    except Exception as exc:  # noqa: BLE001 - the parent decides what to do
        mailbox.put(("error", exc))


def prepare(pdf_path: Path, title: str) -> Prepared:
    """Turn a PDF into chunks without loading Docling into this process.

    Raises:
        ExtractionFailedError: the worker died, timed out, or returned nothing.
        Anything the extraction or chunking stage raised, re-raised unchanged so
            that a caller still sees its own typed errors.
    """
    # "spawn" rather than "fork": a forked child inherits the parent's loaded
    # libraries, including the vector index, which is exactly the pairing that
    # cannot coexist. It is the default on macOS and is set explicitly so the
    # guarantee does not depend on the platform.
    context = multiprocessing.get_context("spawn")
    mailbox = context.Queue(maxsize=1)
    worker = context.Process(target=_child, args=(pdf_path, title, mailbox))
    worker.start()

    deadline = time.monotonic() + settings.EXTRACT_TIMEOUT_SECONDS
    try:
        while True:
            try:
                outcome, payload = mailbox.get(timeout=settings.EXTRACT_POLL_SECONDS)
                break
            except queue.Empty:
                if not worker.is_alive():
                    # The worker may have finished and exited between the last
                    # poll and this check, leaving its result in the pipe. One
                    # more read tells a completed run apart from a dead one.
                    try:
                        outcome, payload = mailbox.get(timeout=settings.EXTRACT_POLL_SECONDS)
                        break
                    except queue.Empty:
                        raise ExtractionFailedError(
                            f"the extraction worker stopped with exit code {worker.exitcode}"
                        ) from None
                if time.monotonic() >= deadline:
                    raise ExtractionFailedError(
                        f"extraction exceeded {settings.EXTRACT_TIMEOUT_SECONDS:.0f}s"
                    ) from None
    finally:
        # A timed-out or half-dead worker holds a Docling model and a few hundred
        # megabytes. Nothing here may leave one running.
        if worker.is_alive():
            worker.terminate()
        worker.join(settings.EXTRACT_SHUTDOWN_SECONDS)
        if worker.is_alive():
            worker.kill()
            worker.join()
        mailbox.close()

    if outcome == "error":
        raise payload
    return payload
