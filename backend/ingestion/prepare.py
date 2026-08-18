"""Extraction and chunking, run in a program of their own.

Docling and Milvus Lite each bundle their own copy of the OpenMP runtime, and a
process that initialises both aborts - `OMP: Error #15`, or a segfault, with no
traceback. It only bites once the vector index has been loaded, so an empty
library ingests happily and every run after that dies.

A subprocess rather than a multiprocessing child: a child inherits the parent's
descriptors, and the parent holds a live gRPC connection to the vector store.
Measured, that child dies with SIGTRAP and an empty stderr.

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

    Retried once if the worker is killed by a signal without writing anything: a
    child started from a process holding many descriptors and live gRPC threads
    occasionally dies outright. A worker that failed on its own - a corrupt PDF,
    no text - writes the reason and is not retried.

    Raises:
        ExtractionFailedError: the worker died twice, timed out, or wrote nothing.
        Whatever extraction or chunking raised, re-raised unchanged.
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
