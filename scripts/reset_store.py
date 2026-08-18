"""Wipe all runtime state, so the app starts as though it had never been used.

    python scripts/reset_store.py            say what would be removed
    python scripts/reset_store.py --yes      remove it

Deleting `data/` by hand does the same thing. This says what it is about to destroy
first - the uploaded PDFs are what citations render from - and refuses to run while
the backend is up, since removing the vector store underneath it leaves a server
answering from a file that no longer exists.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings  # noqa: E402


def _size(path: Path) -> int:
    """Bytes under a path, whether it is a file or a directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _count(path: Path) -> int:
    """How many files this holds. A directory reports its contents, a file itself."""
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file())


def targets() -> list[tuple[str, Path]]:
    """What gets removed, in the order it is reported.

    The SQLite write-ahead files are listed separately because they exist only
    while the database has been written to, and leaving one behind next to a
    deleted database is what turns a clean reset into a confusing half-state.
    """
    return [
        ("document registry", settings.DB_PATH),
        ("registry write-ahead log", settings.DB_PATH.with_suffix(".db-wal")),
        ("registry shared memory", settings.DB_PATH.with_suffix(".db-shm")),
        ("vector store", settings.VECTOR_DIR),
        ("uploaded PDFs", settings.UPLOAD_DIR),
        ("extraction profiles", settings.PROFILE_DIR),
        ("query and document traces", settings.TRACE_DIR),
    ]


def backend_is_running() -> bool:
    """Whether something is already serving on the configured address.

    Checked with the standard library rather than httpx: this script must work
    even in a half-installed environment, since needing a reset and having a
    broken install often arrive together.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{settings.API_BASE_URL}/health", timeout=2):  # noqa: S310
            return True
    except (urllib.error.URLError, OSError):
        return False


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Wipe all runtime state.")
    parser.add_argument("--yes", action="store_true", help="actually remove it")
    args = parser.parse_args()

    present = [(label, path) for label, path in targets() if path.exists()]

    if not present:
        print("nothing to remove: no runtime state exists")
        return 0

    print(f"under {settings.DATA_DIR}")
    for label, path in present:
        print(f"  {label:<26} {path.name:<12} {_count(path):>4} files {_size(path) / 1e6:>7.1f} MB")

    if not args.yes:
        print("\nnothing removed. Run again with --yes to remove all of the above")
        return 0

    # Checked here rather than at the top, so listing what exists always works.
    if backend_is_running():
        print(
            f"\nrefusing to remove anything: something is serving on {settings.API_BASE_URL}.\n"
            "Stop the backend first - the vector store is one file held open by one process."
        )
        return 1

    for _label, path in present:
        remove(path)

    # Recreated immediately, so the next start finds the directories it expects
    # rather than creating them and looking, for one run, like a fresh install
    # that failed halfway.
    settings.ensure_dirs()
    print(f"\nremoved {len(present)} item(s). The next start begins with an empty library")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
