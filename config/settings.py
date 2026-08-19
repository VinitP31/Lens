"""Every tunable value for Lens. No literals anywhere else in the codebase.

Secrets are never stored here. They come from the environment. The reasoning
behind the measured values is in docs/LENS.md.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
# Relative to the project root, so the app behaves the same whatever directory
# it is launched from.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DIR = DATA_DIR / "vectors"
PROFILE_DIR = DATA_DIR / "profiles"
TRACE_DIR = DATA_DIR / "traces"

# One JSON line per query and per indexed document.
QUERY_TRACE_PATH = TRACE_DIR / "queries.jsonl"
DOCUMENT_TRACE_PATH = TRACE_DIR / "documents.jsonl"

# Full output of the last failed check. The console shows only the tail, which is
# not enough for a failure that will not reproduce.
CHECK_LOG_PATH = DATA_DIR / "last-check.log"

# Documents, conversations and messages.
DB_PATH = DATA_DIR / "lens.db"

# --- Upload limits -------------------------------------------------------
# Checked before any expensive work. Extraction cost scales with pages, and the
# whole file is held in memory while it is hashed.

MAX_PAGES = 50
MAX_FILE_MB = 25
MAX_FILE_BYTES = MAX_FILE_MB * 1024 * 1024
# Streamlit's uploader advertises its own limit under the drop zone, read from
# .streamlit/config.toml at startup rather than from here. The two are kept equal
# by a test: a widget offering more than the backend accepts wastes a long upload.


# --- Extraction ----------------------------------------------------------

DOCLING_ACCELERATOR = "AUTO"  # Apple MPS here, CPU elsewhere

# What turns a table into real rows and columns rather than a flat run of text.
# Cell matching keeps the text exact instead of re-reading it.
DOCLING_DETECT_TABLES = True
DOCLING_MATCH_TABLE_CELLS = True

# --- Heading levels ------------------------------------------------------
# Docling calls every heading level 1, so depth is recovered from the height of
# the heading text against the document's most common heading height. Deltas in
# PDF points. A document whose headings are all one size gets a flat path.

HEADING_LEVEL_1_DELTA = 4.0  # a part or chapter title
HEADING_LEVEL_2_DELTA = 1.5  # a section
HEADING_LEVEL_3_DELTA = -1.0  # a subsection
# Anything smaller becomes level 4.

# A multi-line block such as a postal address can be classified as a heading, and
# its box is then several lines tall, which reads as a huge font. Titles are
# short; blocks of text are not.
HEADING_MAX_PARENT_CHARS = 60

HEADING_MAX_DEPTH = 3
HEADING_MAX_CHARS = 120

# A heading this long ending in a full stop or colon is a lead-in sentence, and is
# kept as body text. Short labels like "PERFORMANCE BONDS:" stay headings.
HEADING_SENTENCE_CHARS = 45
HEADING_SENTENCE_WORDS = 5

# "PERFORMANCE BONDS:" introduces a paragraph; "Contract Requirements" names a
# section. Without this they are siblings and each erases the last.
HEADING_DEMOTE_TRAILING_COLON = True

# Some documents open a section with its title inside the first paragraph:
# "BEFORE DAY ONE - Ensure everything is in place...". On one sample that left 75
# of 337 elements filed under the wrong heading.
RUN_IN_HEADING_MIN_BODY_CHARS = 40
RUN_IN_HEADING_LEVEL = 2

# The same in lower case with a colon: "Section D: Secure Hosting Facility:...".
# At ordinary depth, so these are siblings of a heading rather than children.
RUN_IN_LABEL_LEVEL = 3
RUN_IN_LABEL_MAX_WORDS = 4

# A running title repeated across a document is furniture, not a heading.
REPEATED_HEADING_PAGE_RATIO = 0.25

# Repetition alone is not enough: a role label recurs under every phase of a guide
# and is a real heading each time. Furniture sits at the same height on every page
# - measured, 0.0pt of variation against more than 360pt for a real heading.
REPEATED_HEADING_POSITION_SPREAD = 6.0

# --- Glyph repair --------------------------------------------------------
# Layout analysis occasionally mis-decodes a symbol: "≥ 4 hours" arrived as
# "‡ 4 hours", which changes what the document requires. PyMuPDF reads the same
# glyph correctly, so it is asked; if both readers agree, nothing is changed.

# Words after the suspect character used to locate the passage. The match must be
# unique or the repair is skipped.
GLYPH_REPAIR_CONTEXT_WORDS = 5

# --- Contents pages ------------------------------------------------------
# Recognised by what a contents page is - a list of the document's own headings -
# so a page of prose can never qualify.

CONTENTS_MAX_PAGE_RATIO = 0.15  # contents sit at the front
CONTENTS_MIN_ENTRIES = 4
CONTENTS_HEADING_MATCH_RATIO = 0.6

# --- Reading order -------------------------------------------------------
# Docling sometimes emits elements out of visual position, separating a value from
# its label, so pages are re-ordered by position - but only when that is safe. A
# grid and a two-column article both have two columns; what separates them is that
# a grid's cells line up in rows and an article's paragraphs do not.

COLUMN_GAP_RATIO = 0.15

# An element taller than this holds more than one line, which is what a paragraph
# does and a grid cell does not.
MULTILINE_HEIGHT_POINTS = 18.0
PROSE_COLUMN_RATIO = 0.5

# Article columns are equal width by design; a label-and-value grid is lopsided.
# Measured: 0.21 on a real calendar page against 0.99 for an article.
COLUMN_WIDTH_SIMILARITY = 0.7

ROW_BAND = 2.0  # tops this close count as the same row when sorting

# --- Label and value pairing ---------------------------------------------
# "Fees" and "30 pts" arrive as two elements and nothing records that they belong
# together, so several such pairs read as a list of words and numbers. Both sides
# are held to a strict shape: merging two elements that were never a pair invents
# a fact the document does not state.

LABEL_MAX_WORDS = 4

# Between the nearest edges, so it covers a value beside its label as well as
# beneath it. About one line of text.
LABEL_VALUE_MAX_GAP_POINTS = 20.0

# --- Extraction worker ---------------------------------------------------
# Extraction runs in its own process because Docling and Milvus Lite each bundle a
# copy of the OpenMP runtime and a process that initialises both dies.

# Generous: extraction measured about 3s a page against a 50 page limit. This
# catches a worker that has hung, it does not hurry a slow one.
EXTRACT_TIMEOUT_SECONDS = 600.0

# --- Embedding model -----------------------------------------------------
# Named here because chunk size must be measured with the tokenizer this model
# actually uses.

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# The model's hard input ceiling. A chunk over this cannot be indexed at all,
# which is why tables - otherwise never split - are split at this limit.
EMBED_MAX_INPUT_TOKENS = 8191

# A 144-chunk document becomes two requests instead of 144. Kept well under the
# provider's per-request budget so a batch of large tables cannot overflow it.
EMBED_BATCH_SIZE = 64

EMBED_MAX_RETRIES = 3
EMBED_RETRY_BACKOFF_SECONDS = 2.0

# --- Vector store --------------------------------------------------------
# Milvus Lite: a local file, no server to run.

MILVUS_PATH = VECTOR_DIR / "chunks.db"
MILVUS_COLLECTION = "chunks"

# Read this before touching the gate. With COSINE, Milvus puts the cosine
# SIMILARITY in the field it calls `distance`: identical +1.0, unrelated 0.0,
# opposite -1.0. Measured, not assumed. Subtracting it from 1 inverts the gate.
MILVUS_METRIC = "COSINE"
MILVUS_INDEX_TYPE = "HNSW"

MILVUS_SECTION_PATH_MAX = 1024
MILVUS_TEXT_MAX = 65535
MILVUS_ID_MAX = 80

# --- Chunking ------------------------------------------------------------
# In tokens, never characters: the same passage differs three or four times over
# in tokens depending on whether it is prose, numbers or codes.

CHUNK_TARGET_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 75  # 15% of target, so a straddling answer stays whole

# Below this a chunk has lost its subject: "employees get 18 days" does not say of
# what. A short trailing chunk is merged back rather than stored alone.
CHUNK_MIN_TOKENS = 120

CHUNK_MAX_TOKENS = 800  # hard ceiling, splits even inside a section

# The header is embedded with the chunk, so a question phrased in the heading's
# words matches even when the body uses different ones.
CONTEXT_SEPARATOR = " > "
CONTEXT_PAGE_PREFIX = "p."

# --- Retrieval -----------------------------------------------------------
# Over-fetch then narrow: overlap makes neighbouring chunks near-duplicates, so
# asking for exactly what will be used spends slots on repeats.

RETRIEVE_CANDIDATES = 12
CONTEXT_CHUNKS = 5

# --- The confidence gate -------------------------------------------------
# Compared against `Hit.similarity`, never a raw Milvus score.
#
# Measured on 24 answerable and 17 unanswerable questions:
#
#   answerable    lowest +0.496   mean +0.650
#   unanswerable  lowest +0.392   mean +0.550
#
# The two sets overlap, so no value separates them. What each costs:
#
#   0.45  refuses 0 of 24 real answers, lets 13 of 17 unanswerable through
#   0.60  refuses 7 of 24,              lets  7 of 17 through
#   0.74  refuses 20 of 24,             lets  0 of 17 through
#
# Set below the lowest real answer observed, because the two errors are not
# symmetrical: an unanswerable question that gets through still meets the prompt's
# abstention rule, while a real answer wrongly refused meets nothing.
#
# Raising it is the right response to off-topic questions being answered. It is
# the wrong response to on-topic questions whose answer is absent.
GATE_THRESHOLD = 0.45

# Two chunks are the same passage when one repeats a run of the other's words,
# which is what the 75-token overlap window produces. Measured as the longest
# shared run, not a proportion: short chunks about different facts share most of
# their ordinary vocabulary.
DEDUPE_MIN_SHARED_WORDS = 12

# --- OCR -----------------------------------------------------------------
# Conditional, never on by default: it roughly triples ingestion time and its
# output is a guess where a real text layer is exact.

# Averaged across the whole document, so a good report with a few full-page charts
# is not mistaken for a scan.
OCR_TRIGGER_CHARS_PER_PAGE = 150

# Still below this after OCR means the file is unreadable, and it is rejected
# rather than indexed as near-empty chunks.
MIN_CHARS_PER_PAGE = 150

# RapidOCR needs no system package, so a clean clone works without brew or apt.
OCR_ENGINE = "rapidocr"
OCR_LANGUAGES = ("english",)

# --- Page rendering ------------------------------------------------------
# An image rather than an embedded PDF viewer: browsers treat a page anchor in an
# iframe inconsistently, and a viewer cannot draw a box over the cited text.

RENDER_DPI = 150  # roughly 1275x1650 for US Letter

# Translucent, so the words underneath stay readable.
HIGHLIGHT_FILL = (1.0, 0.85, 0.25)
HIGHLIGHT_OPACITY = 0.32
HIGHLIGHT_BORDER = (0.85, 0.6, 0.05)
HIGHLIGHT_BORDER_WIDTH = 0.8

# A box tight around a line reads as an underline rather than a highlight.
HIGHLIGHT_PADDING = 1.5

# Height of the thread pane and the page pane beside it. One value, so the two
# line up. Both scroll inside themselves and the window does not scroll at all.
PANEL_HEIGHT = 720

# --- Frontend to backend -------------------------------------------------
# The UI is a separate process and reaches the backend over HTTP only.

API_BASE_URL = "http://127.0.0.1:8000"

API_TIMEOUT_SECONDS = 120.0  # an answer waits on a model
API_UPLOAD_TIMEOUT_SECONDS = 180.0  # uploading validates the whole file first
STATUS_POLL_SECONDS = 2.0

# --- Conversations -------------------------------------------------------
# Chat state lives in SQLite because Streamlit wipes session state on refresh.

# Turns used to rewrite a follow-up into a standalone question. Bounded: turns
# from far enough back describe a different subject.
HISTORY_WINDOW_TURNS = 6

TITLE_MAX_CHARS = 60  # cut at a word boundary

# --- Utility model -------------------------------------------------------
# Deciding what kind of message something is, rewriting a follow-up, shortening a
# long message. None of it produces an answer.

MODEL_UTILITY = "gpt-4o-mini"

# Above this a message is reduced to the question inside it before being embedded:
# a long message spreads its meaning over far more text than any chunk holds, so
# it matches everything weakly.
CONDENSE_CHAR_THRESHOLD = 1500

CONDENSE_MAX_OUTPUT_TOKENS = 200
ANALYZE_MAX_OUTPUT_TOKENS = 300

# --- Generation ----------------------------------------------------------

MODEL_ANSWER = "gpt-4o-mini"

# Retrieval is deterministic, so the same question retrieves the same passages.
# Zero gets as close to a deterministic answer as hosted inference allows: the
# same substance and the same citations, not the same bytes.
TEMPERATURE = 0.0

ANSWER_MAX_OUTPUT_TOKENS = 800
ANSWER_MAX_RETRIES = 3
ANSWER_RETRY_BACKOFF_SECONDS = 2.0

# What the model must reply when the passages do not hold the answer. Deliberately
# not a sentence: matching on prose would read "I could not find" as a real
# answer, and this is the second of the two refusal layers.
ABSTENTION_MARKER = "NOT_IN_DOCUMENTS"

# How much of a cited passage is stored for display. The whole chunk is a wall of
# text in a chat reply; the page view is where the full passage is read.
CITATION_SNIPPET_CHARS = 300

# A table gets more room and is cut only between rows: half a row states a label
# with no value, and a reader takes what they see as what the document says.
CITATION_TABLE_SNIPPET_CHARS = 1200

# Marks a table that had rows left over, on its own line.
SNIPPET_TRUNCATED_MARK = "…"


def ensure_dirs() -> None:
    """Create the runtime directories. Safe to call repeatedly."""
    for directory in (DATA_DIR, UPLOAD_DIR, VECTOR_DIR, PROFILE_DIR, TRACE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
