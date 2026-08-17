"""Tests for the reset script.

The script deletes everything the app has produced, so what is tested here is
mostly what it refuses to do: nothing without `--yes`, and nothing at all while a
backend is serving. A reset that half-worked is worse than one that did not run.

Every path is redirected into a temporary directory, so no test can reach the real
`data/`.
"""

import pytest

from config import settings
from scripts import reset_store


@pytest.fixture
def state(tmp_path, monkeypatch):
    """A data directory holding one of everything the script removes."""
    monkeypatch.setattr(settings, "DATA_DIR", tmp_path)
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "lens.db")
    monkeypatch.setattr(settings, "VECTOR_DIR", tmp_path / "vectors")
    monkeypatch.setattr(settings, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(settings, "PROFILE_DIR", tmp_path / "profiles")
    monkeypatch.setattr(settings, "TRACE_DIR", tmp_path / "traces")
    # Nothing is serving in a test run, and a real request would make the suite
    # depend on whether a developer happens to have the backend open.
    monkeypatch.setattr(reset_store, "backend_is_running", lambda: False)

    settings.ensure_dirs()
    (tmp_path / "lens.db").write_bytes(b"registry")
    (tmp_path / "lens.db-wal").write_bytes(b"pending writes")
    (tmp_path / "vectors" / "chunks.db").write_bytes(b"vectors")
    (tmp_path / "uploads" / "abc123.pdf").write_bytes(b"%PDF-1.7")
    (tmp_path / "traces" / "queries.jsonl").write_text('{"q": 1}\n')
    return tmp_path


def run(*arguments: str) -> int:
    import sys

    original = sys.argv
    sys.argv = ["reset_store.py", *arguments]
    try:
        return reset_store.main()
    finally:
        sys.argv = original


def test_nothing_is_removed_without_the_flag(state):
    """The default has to be safe. Someone reading the script name and running it
    to see what it does must not lose their library by doing so."""
    assert run() == 0

    assert (state / "lens.db").exists()
    assert (state / "uploads" / "abc123.pdf").exists()


def test_the_flag_removes_every_kind_of_state(state):
    assert run("--yes") == 0

    assert not (state / "lens.db").exists()
    assert not (state / "lens.db-wal").exists()
    assert not (state / "vectors" / "chunks.db").exists()
    assert not (state / "uploads" / "abc123.pdf").exists()
    assert not (state / "traces" / "queries.jsonl").exists()


def test_the_directories_come_back_empty(state):
    """The next start expects them to exist. Leaving them missing makes a clean
    reset look like an install that failed halfway."""
    run("--yes")

    for directory in (settings.UPLOAD_DIR, settings.VECTOR_DIR, settings.TRACE_DIR):
        assert directory.is_dir()
        assert not list(directory.iterdir())


def test_a_running_backend_stops_the_whole_reset(state, monkeypatch):
    """Not a partial delete. The vector store is one file held open by one
    process, so removing it underneath a live server leaves the backend answering
    from a store that is gone."""
    monkeypatch.setattr(reset_store, "backend_is_running", lambda: True)

    assert run("--yes") == 1
    assert (state / "lens.db").exists()
    assert (state / "vectors" / "chunks.db").exists()


def test_an_already_empty_directory_is_not_an_error(state):
    run("--yes")

    assert run("--yes") == 0


def test_what_would_be_removed_is_listed_before_it_is(state, capsys):
    """A user is about to lose the original PDFs a citation renders from. Naming
    them first is the difference between a reset and an accident."""
    run()

    printed = capsys.readouterr().out
    assert "uploaded PDFs" in printed
    assert "traces" in printed
    assert "--yes" in printed
