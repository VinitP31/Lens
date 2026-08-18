"""Keeping the original PDF on disk.

Citations render the source page as an image, so the file is the evidence the
product rests on. If it is gone, every past citation stops being checkable.

Files are named by content hash, never by the uploaded name: an uploaded name can
collide, hold a path separator, or be a traversal attempt. The display name lives
in the registry.
"""

from pathlib import Path

from config import settings


def path_for(content_hash: str) -> Path:
    """Where this document's bytes live.

    Derived from the hash rather than stored, so a moved data directory does not
    invalidate every row in the registry.
    """
    return settings.UPLOAD_DIR / f"{content_hash}.pdf"


def save(data: bytes, content_hash: str) -> Path:
    """Write the upload and return its path.

    Written to a temporary name and then moved into place. A crash halfway
    through a direct write would leave a truncated file under the name of a
    document the registry believes is complete, and the rename is what makes the
    file appear whole or not at all.
    """
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    destination = path_for(content_hash)

    # Already present, byte-identical by definition of the name. Rewriting it
    # would risk truncating a file other answers are citing right now.
    if destination.exists():
        return destination

    staging = destination.with_suffix(".part")
    staging.write_bytes(data)
    staging.replace(destination)
    return destination


def delete(content_hash: str) -> None:
    """Remove a document's bytes. Safe to call when they are already gone.

    Only used when an ingest is rolled back. A soft-deleted document keeps its
    file, because answers given before the deletion still cite its pages.
    """
    path_for(content_hash).unlink(missing_ok=True)
