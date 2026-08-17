"""Tests for the check runner's failure log.

The console shows only the tail of a failed check, and that was enough until a test
failed once, refused to reproduce, and the lines explaining it had already scrolled
away. These tests are about the part that stops that happening twice: the whole
output surviving on disk, and a logging problem never being allowed to hide the
result of the check itself.
"""

import sys

import pytest

from config import settings
from scripts import check


@pytest.fixture
def log_path(tmp_path, monkeypatch):
    path = tmp_path / "last-check.log"
    monkeypatch.setattr(settings, "CHECK_LOG_PATH", path)
    return path


def test_a_failing_check_writes_its_whole_output(log_path, capsys):
    """The tail on screen is a summary. The file is the evidence."""
    marker = "the line that explains everything"
    command = [sys.executable, "-c", f"import sys; print({marker!r}); sys.exit(1)"]

    assert check._run("probe", command) is False

    written = log_path.read_text()
    assert marker in written
    assert "probe" in written
    # And the console says where to look, or the file might as well not exist.
    assert str(log_path) in capsys.readouterr().out


def test_a_passing_check_writes_nothing(log_path):
    """A log left over from a green run would be read as a failure next time."""
    assert check._run("probe", [sys.executable, "-c", "print('fine')"]) is True

    assert not log_path.exists()


def test_the_log_holds_output_the_console_cut_off(log_path):
    """The reason this exists. The console keeps 25 lines; the first line of a long
    failure is exactly the one that names the cause."""
    script = "import sys\nfor n in range(200): print(f'line {n}')\nsys.exit(1)"

    check._run("probe", [sys.executable, "-c", script])

    written = log_path.read_text()
    assert "line 0" in written
    assert "line 199" in written


def test_a_log_that_cannot_be_written_does_not_hide_the_failure(tmp_path, monkeypatch):
    """A check that failed has produced a result. Losing it to a logging error
    would be the worse failure of the two."""
    monkeypatch.setattr(settings, "CHECK_LOG_PATH", tmp_path / "missing" / "x" / "log")
    monkeypatch.setattr(
        check.Path, "mkdir", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("read-only"))
    )

    assert check._run("probe", [sys.executable, "-c", "import sys; sys.exit(1)"]) is False


def test_each_run_replaces_the_last(log_path):
    """The interesting failure is the one that just happened."""
    check._run("first", [sys.executable, "-c", "import sys; print('old'); sys.exit(1)"])
    check._run("second", [sys.executable, "-c", "import sys; print('new'); sys.exit(1)"])

    written = log_path.read_text()
    assert "new" in written
    assert "old" not in written
