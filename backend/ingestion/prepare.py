"""Extraction and chunking, run in a program of their own.

The reason this module exists is a hard incompatibility, not a preference.

Docling brings its own copy of the OpenMP runtime, inside PyTorch. Milvus Lite
brings a second copy, inside FAISS. That runtime refuses to initialise twice in
one process, and when both do it the process dies outright - `OMP: Error #15` and
an abort, or a bare segmentation fault, with no Python traceback to read.

It only bites once the vector index has actually been loaded, which is why an
empty library ingests happily and a library with documents already in it does
not. In other words the failure is invisible on a first run and certain on every
run after that. Adding a second document is the ordinary case, so this is not an
edge to note and move past.

There is a documented environment variable that suppresses the abort. Its own
documentation says it may silently produce incorrect results, which is worse
than a crash: a crash is visible. Making the two copies into one by hand is a
per-machine change that would not survive a fresh install by anyone else.

**A subprocess, not a multiprocessing child.** This started as `multiprocessing`
with the "spawn" method, which is the usual answer and is wrong here. A child
started that way inherits the parent's open file descriptors, and the parent in
the real system holds a live gRPC connection to the vector store. Measured: the
child dies with SIGTRAP and an empty stderr - no traceback, no message, just a
document that never indexes. It failed three runs in four under load and none
when the machine was quiet, which is the worst kind of bug to inherit. A
subprocess launched with its descriptors closed shares nothing with the parent
and cannot be poisoned by what the parent has open.

`Prepared` and `Chunk` are plain dataclasses over built-in types, so they cross
the process boundary without either side needing the other's libraries.
"""

import logging
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from backend.errors import ExtractionFailedError
from backend.ingestion.chunk import Chunk
from config import settings

log = logging.getLogger(__name__)

# The project root, so the worker can import `backend` whatever directory the
# caller happens to be in.
ROOT = Path(__file__).resolve().parent.parent.parent


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


def prepare(pdf_path: Path, title: str) -> Prepared:
    """Turn a PDF into chunks without loading Docling into this process.

    Retried once if the worker is killed by a signal without writing anything.
    That failure is transient and has a measured cause: Milvus Lite leaks around
    ten file descriptors per connection even after being closed, and starting a
    child from a process holding hundreds of them, with live gRPC threads among
    them, occasionally kills the child outright - SIGTRAP, empty stderr, nothing
    to read.

    The real system opens one connection for its lifetime and never reaches that
    state, which is why uploads succeed there; a long-running process that opens
    many would. One retry costs nothing when nothing is wrong and turns a random
    failed upload into a slower successful one.

    A worker that failed on its own - a corrupt PDF, no text - writes the reason
    and exits non-zero. That is not retried: it would fail identically.

    Raises:
        ExtractionFailedError: the worker died twice, timed out, or wrote nothing.
        Anything the extraction or chunking stage raised, re-raised unchanged so
            that a caller still sees its own typed errors.
    """
    for attempt in (1, 2):
        try:
            return _attempt(pdf_path, title)
        except _WorkerKilled as killed:
            if attempt == 2:
                raise ExtractionFailedError(str(killed)) from killed
            log.warning("extraction worker was killed, retrying once: %s", killed)
    raise ExtractionFailedError("extraction did not run")  # pragma: no cover


class _WorkerKilled(Exception):
    """The worker died by signal without reporting anything. Worth one retry."""


def _attempt(pdf_path: Path, title: str) -> Prepared:
    """One run of the worker."""
    with tempfile.TemporaryDirectory(prefix="lens-extract-") as workspace:
        outcome_path = Path(workspace) / "prepared.pickle"

        try:
            result = subprocess.run(  # noqa: S603 - fixed argument list, no shell
                [
                    sys.executable,
                    "-m",
                    "backend.ingestion.worker",
                    str(pdf_path),
                    title,
                    str(outcome_path),
                ],
                cwd=ROOT,
                # Nothing of this process is handed to the worker. This is the
                # whole point: an inherited gRPC descriptor is what killed the
                # multiprocessing version.
                close_fds=True,
                capture_output=True,
                text=True,
                timeout=settings.EXTRACT_TIMEOUT_SECONDS,
                check=False,
                env={
                    "PYTHONPATH": str(ROOT),
                    "PATH": _path_for_worker(),
                    # Docling reads these; passing them through keeps the worker's
                    # model cache where the rest of the system expects it.
                    **{
                        name: value
                        for name, value in _inheritable_environment().items()
                        if value is not None
                    },
                },
            )
        except subprocess.TimeoutExpired as error:
            raise ExtractionFailedError(
                f"extraction exceeded {settings.EXTRACT_TIMEOUT_SECONDS:.0f}s"
            ) from error

        if not outcome_path.exists():
            # No result written at all: the worker was killed rather than failing
            # on its own. Its stderr is the only clue, and Docling is noisy, so
            # only the tail is worth reporting.
            tail = (result.stderr or "").strip().splitlines()
            detail = tail[-1] if tail else "no output"
            message = f"the extraction worker stopped with exit code {result.returncode}: {detail}"
            # A negative code means a signal, which is the transient case.
            if result.returncode < 0:
                raise _WorkerKilled(message)
            raise ExtractionFailedError(message)

        status, payload = pickle.loads(outcome_path.read_bytes())  # noqa: S301 - our own bytes

    if status == "error":
        raise payload
    return payload


def _path_for_worker() -> str:
    """The search path the worker needs, taken from this process."""
    import os

    return os.environ.get("PATH", "/usr/bin:/bin")


def _inheritable_environment() -> dict[str, str | None]:
    """Environment the worker legitimately needs.

    Deliberately a short list rather than the whole environment. The worker does
    not answer questions and never needs an API key, so it is not given one.
    """
    import os

    return {
        "HOME": os.environ.get("HOME"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "HF_HOME": os.environ.get("HF_HOME"),
        "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
        "TORCH_HOME": os.environ.get("TORCH_HOME"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    }
