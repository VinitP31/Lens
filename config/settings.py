"""Every tunable value for Lens. No literals anywhere else in the codebase.

This file grows one build stage at a time. A value appears here when the code
that reads it is written, so there is never a setting nobody uses.

Secrets are never stored here. They come from the environment.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
# Everything is relative to the project root, so the app behaves the same
# whatever directory it is launched from.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DIR = DATA_DIR / "vectors"
PROFILE_DIR = DATA_DIR / "profiles"

# --- Extraction ----------------------------------------------------------
# Docling's standard PDF pipeline. AUTO picks the best available accelerator,
# which is Apple MPS on this machine and CPU elsewhere.

DOCLING_ACCELERATOR = "AUTO"

# Table structure recognition is what turns a table into real rows and columns
# instead of a flat run of text. Cell matching aligns recognised cells with the
# PDF's own text, so table content stays exact rather than being re-read.
DOCLING_DETECT_TABLES = True
DOCLING_MATCH_TABLE_CELLS = True

# --- OCR -----------------------------------------------------------------
# OCR is conditional, never on by default: it roughly triples ingestion time,
# and its output is a guess where a real text layer is exact.

# Below this many characters per page, averaged across the whole document,
# the PDF is treated as scanned and OCR runs. Averaged document-wide on
# purpose — a good report with a few full-page charts must not be flagged.
OCR_TRIGGER_CHARS_PER_PAGE = 150

# Still below this after OCR means the file is genuinely unreadable, and is
# rejected rather than indexed as near-empty chunks.
MIN_CHARS_PER_PAGE = 150


def ensure_dirs() -> None:
    """Create the runtime directories. Safe to call repeatedly."""
    for directory in (DATA_DIR, UPLOAD_DIR, VECTOR_DIR, PROFILE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
