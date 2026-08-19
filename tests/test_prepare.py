"""Tests for the extraction worker.

The point of this module is that Docling never loads into the process holding the
vector store, so the most important test here runs a fresh interpreter and looks
at what actually got imported. Asserting it from inside the suite would prove
nothing: by then another test module has already imported the extractor.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend.errors import ExtractionFailedError
from backend.ingestion import prepare
from config import settings

ROOT = Path(__file__).resolve().parent.parent


def test_the_parent_process_never_loads_docling():
    """Importing the worker must not pull Docling into this process.

    Docling and Milvus Lite each bundle a copy of the OpenMP runtime, and a process
    that initialises both aborts. This is the reason the worker is a separate program.
    """
    probe = (
        "import sys;"
        "from backend.ingestion import prepare;"
        "from backend.storage import vector_store;"
        "print([m for m in ('torch', 'docling', 'transformers') if m in sys.modules])"
    )
    # An explicit path rather than relying on the working directory, and the
    # error surfaced rather than swallowed: `check=True` raises a
    # CalledProcessError that hides the child's stderr, which is the only thing
    # that explains why the probe failed.
    #
    # gRPC's fork handling is switched off for the child. By the time this test
    # runs, other tests have opened the vector store, so the pytest process holds
    # live gRPC threads - and a fork from it runs gRPC's atfork handlers in the
    # child, which intermittently killed the probe before it reached the exec.
    # Measured: the child died with "FD from fork parent still in poll list" and a
    # non-zero exit, roughly one run in six, with nothing wrong with the code under
    # test. The child replaces itself with a fresh interpreter immediately, so it
    # has no use for the parent's gRPC state.
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "GRPC_ENABLE_FORK_SUPPORT": "0",
    }
    result = subprocess.run(  # noqa: S603 - fixed argument list, no shell
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, f"probe failed:\n{result.stderr[-2000:]}"
    assert result.stdout.strip().endswith("[]"), result.stdout


def test_a_document_comes_back_as_chunks(simple_pdf):
    prepared = prepare.prepare(simple_pdf, title="Simple")

    assert prepared.chunks
    assert prepared.page_count == 3
    assert all(chunk.text for chunk in prepared.chunks)


def test_the_counts_needed_to_mark_a_document_ready_are_present(simple_pdf):
    """The registry records these at the end of ingestion. A worker that returned
    chunks but no counts would leave every document reporting zero pages."""
    prepared = prepare.prepare(simple_pdf, title="Simple")

    assert prepared.chars_per_page > 0
    assert prepared.seconds > 0
    assert prepared.needs_ocr is False


def test_chunks_survive_the_process_boundary_intact(simple_pdf):
    """Chunks are pickled between processes, so anything not built from plain
    types would arrive damaged or not at all."""
    prepared = prepare.prepare(simple_pdf, title="Simple")
    chunk = prepared.chunks[0]

    assert isinstance(chunk.page, int)
    assert isinstance(chunk.bboxes, list)
    assert chunk.context_header.startswith("[Simple")
    assert chunk.embed_text.endswith(chunk.text)


def test_a_failure_in_the_worker_keeps_its_own_type(corrupt_pdf):
    """The caller has to see the real reason. A worker that died silently would
    be indistinguishable from a file that cannot be read."""
    with pytest.raises(ExtractionFailedError):
        prepare.prepare(corrupt_pdf, title="Corrupt")


def test_a_missing_file_is_reported_rather_than_hanging(tmp_path):
    with pytest.raises(ExtractionFailedError):
        prepare.prepare(tmp_path / "absent.pdf", title="Absent")


def test_a_worker_that_overruns_is_stopped(monkeypatch, simple_pdf):
    """A hung worker holds a Docling model and several hundred megabytes.

    The outcome is asserted, not the wording: at this timeout the parent stops the
    child during its startup, so two correct reports race - the deadline passing, and
    the child found already dead.
    """
    monkeypatch.setattr(settings, "EXTRACT_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(ExtractionFailedError):
        prepare.prepare(simple_pdf, title="Simple")


def test_nothing_on_the_query_side_can_reach_docling():
    """The same rule as the probe above, read from the imports rather than run.

    Cannot be upset by whatever else is running, and names the file that broke it.
    """
    import ast

    forbidden = {"docling", "docling_core", "torch", "transformers"}
    # Everything the process that answers questions imports, directly or not.
    entry_points = [
        ROOT / "backend" / "main.py",
        *(ROOT / "backend" / "api").glob("*.py"),
        *(ROOT / "backend" / "retrieval").glob("*.py"),
        *(ROOT / "backend" / "storage").glob("*.py"),
        *(ROOT / "backend" / "rendering").glob("*.py"),
        ROOT / "backend" / "ingestion" / "pipeline.py",
        ROOT / "backend" / "ingestion" / "prepare.py",
        ROOT / "backend" / "ingestion" / "chunk.py",
        ROOT / "backend" / "ingestion" / "embedder.py",
        ROOT / "backend" / "ingestion" / "validator.py",
    ]

    def imports_of(path):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    yield alias.name, node.lineno
            elif isinstance(node, ast.ImportFrom) and node.module:
                yield node.module, node.lineno

    seen: set[Path] = set()
    queue = list(entry_points)
    offences: list[str] = []

    while queue:
        path = queue.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)

        for module, line in imports_of(path):
            root = module.split(".")[0]
            if root in forbidden:
                offences.append(f"{path.relative_to(ROOT)}:{line} imports {module}")
            if root in ("backend", "config"):
                # Follow it: the rule is about the whole graph, not one file.
                target = ROOT / Path(*module.split(".")).with_suffix(".py")
                queue.append(target)

    assert not offences, "the query side reaches Docling:\n" + "\n".join(offences)
