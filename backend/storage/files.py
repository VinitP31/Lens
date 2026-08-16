"""Keeping the original PDF on disk.

Citations render the source page as an image, so the original file is not a
convenience - it is the evidence the whole product rests on. If it is gone, every
citation in every past answer stops being checkable.

Files are named by their content hash, never by the name they were uploaded
under. An uploaded name is chosen by whoever uploaded it: it can collide, contain
a path separator, or be a traversal attempt. A hash is fixed-length, unique to
the bytes, and cannot escape the directory it is written to.

The display name a user sees lives in the registry, where a duplicate can be
given a counter suffix without touching what is on disk.
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
