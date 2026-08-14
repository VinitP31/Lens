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

# Application state: the document registry, conversations and messages. One
# local file, so a restart loses nothing and there is no server to run.
DB_PATH = DATA_DIR / "lens.db"

# --- Extraction ----------------------------------------------------------
# Docling's standard PDF pipeline. AUTO picks the best available accelerator,
# which is Apple MPS on this machine and CPU elsewhere.

DOCLING_ACCELERATOR = "AUTO"

# Table structure recognition is what turns a table into real rows and columns
# instead of a flat run of text. Cell matching aligns recognised cells with the
# PDF's own text, so table content stays exact rather than being re-read.
DOCLING_DETECT_TABLES = True
DOCLING_MATCH_TABLE_CELLS = True

# --- Heading levels ------------------------------------------------------
# Docling assigns level 1 to every heading it finds, in every document tested,
# so heading depth has to be recovered from typography instead. Levels are
# derived from the height of the heading text relative to the most common
# heading height in that document, which is generic: no document knows it is
# being measured, and a document whose headings are all one size degrades to a
# flat path rather than failing.
#
# Deltas are in PDF points, measured against the document's modal height.
HEADING_LEVEL_1_DELTA = 4.0  # much larger than usual: a part or chapter title
HEADING_LEVEL_2_DELTA = 1.5  # clearly larger: a section
HEADING_LEVEL_3_DELTA = -1.0  # about usual: a subsection
# Anything smaller becomes level 4.

# A heading only becomes a parent if its text is short. Layout models sometimes
# classify a multi-line block such as a postal address as a heading, and its
# bounding box is then several lines tall, which would read as a very large font
# and promote it above every real section. Chapter and section titles are short;
# blocks of text are not.
HEADING_MAX_PARENT_CHARS = 60

# Deepest section path kept. Beyond this the extra ancestors add words to the
# embedded text without adding meaning.
HEADING_MAX_DEPTH = 3

# Layout models sometimes classify a lead-in sentence as a heading, for example
# "The vendor must clearly identify all third-party components and describe:"
# above a list. Treating that as a heading loses it as content and invents a
# section that does not exist.
#
# A heading this long that ends in a full stop or a colon is a sentence, and is
# kept as body text instead. Short labels such as "PERFORMANCE BONDS:" are real
# headings and stay below the limit.
HEADING_SENTENCE_CHARS = 45

# A label reads as a few words: "Evaluation Factors:", "Mail sealed proposals
# to:". Once it reaches this many words it is a sentence introducing what comes
# next, whatever its length in characters.
HEADING_SENTENCE_WORDS = 5

# A short label ending in a colon sits beneath the heading above it, even when
# both are set in the same size. "Contract Requirements" names a section;
# "PERFORMANCE BONDS:" introduces the paragraph under it. Without this they are
# siblings, and each one erases the last, so the parent disappears.
HEADING_DEMOTE_TRAILING_COLON = True

# Some documents set a section title as the opening words of its own paragraph:
# "BEFORE DAY ONE - Ensure everything is in place before the first day...".
# Layout analysis correctly calls that one block of text, so the title never
# becomes a heading and every following page inherits whatever heading came
# before it. On one sample that mislabelled 75 of 337 elements.
#
# Such an opening is recognised so it can label the text beneath it. The
# paragraph itself is never altered, so nothing can be lost by this.
RUN_IN_HEADING_MIN_BODY_CHARS = 40  # prose must follow, not just a few words
RUN_IN_HEADING_LEVEL = 2  # a title set in capitals introduces a whole section

# The same thing happens in lower case with a colon: "Section D: Secure Hosting
# Facility Profile: Details of...". Where one item of such a list happens to be
# typeset as a real heading, it otherwise becomes the parent of its own siblings.
#
# These sit at ordinary heading depth, so they are siblings of a heading rather
# than children of one.
RUN_IN_LABEL_LEVEL = 3

# A label is a few words. Beyond this it is an instruction - "Establish preferred
# method of communication:" - and making it a heading would file the items that
# follow underneath it.
RUN_IN_LABEL_MAX_WORDS = 4

# No heading is longer than this. Anything longer is a paragraph.
HEADING_MAX_CHARS = 120

# A running title repeated across a document is furniture, even when the layout
# model calls it a heading. Left in, it becomes the parent of unrelated sections.
# A real heading does not recur on this share of the pages.
REPEATED_HEADING_PAGE_RATIO = 0.25

# Repetition alone does not make a heading furniture. A role label such as
# "Onboarding Partner" recurs under every phase of a guide and is a real heading
# each time. What gives furniture away is that it is printed at the same height
# on every page: measured across one sample, a running title varied by 0.0pt
# while a recurring role heading varied by more than 360pt.
REPEATED_HEADING_POSITION_SPREAD = 6.0

# --- Glyph repair --------------------------------------------------------
# Layout analysis occasionally mis-decodes a symbol: a threshold written "≥ 4
# hours" arrived as "‡ 4 hours", which turns "at least four hours" into "four
# hours" and changes what the document requires.
#
# PyMuPDF reads the same glyph correctly, so where a character looks like a
# decoding artifact the second reader is asked what is actually there. Nothing
# is assumed about what the character should be: if both readers agree, the text
# is left alone. That keeps a document that legitimately uses a dagger intact.

# Words after the suspect character used to locate the passage in the page text.
# The match must be unique, or the repair is skipped.
GLYPH_REPAIR_CONTEXT_WORDS = 5

# --- Contents pages ------------------------------------------------------
# Docling labels most contents pages itself. When it does not, the page is
# recognised by what a contents page is: a list of the document's own headings.
# Matching against headings the document really has means an ordinary page of
# prose can never qualify, and no typographic guessing is involved.

# Only pages this far into the document may be considered. Contents sit at the
# front; a page listing section names in the middle is a summary, not an index.
CONTENTS_MAX_PAGE_RATIO = 0.15

# The page needs at least this many entries before the shape means anything.
CONTENTS_MIN_ENTRIES = 4

# Share of the page's text that must repeat one of the document's own headings.
CONTENTS_HEADING_MATCH_RATIO = 0.6

# --- Reading order -------------------------------------------------------
# Docling occasionally emits elements out of visual position, which separates a
# value from its label. Pages are therefore re-ordered by position, but only
# when doing so is safe.
#
# The unsafe case is a genuine two-column article, where reading row by row
# would interleave the two columns into nonsense. A grid and an article both
# have two columns of text; what separates them is that a grid's cells line up
# in rows, and an article's paragraphs do not.

# Horizontal gap, as a fraction of the page's content width, that separates one
# column of text from another.
COLUMN_GAP_RATIO = 0.15

# What separates a grid from an article is the shape of its content, not its
# alignment: a grid cell holds one line, an article paragraph holds several.
# Baselines often line up across the columns of an article too, so alignment
# alone cannot tell them apart.
#
# An element taller than this holds more than one line of text.
MULTILINE_HEIGHT_POINTS = 18.0

# Fraction of a page's elements that must be multi-line blocks before the page
# is read as columns of prose rather than as a grid.
PROSE_COLUMN_RATIO = 0.5

# Two columns of an article are set to the same width by design. A grid pairing
# a short label with a long value is lopsided: on a real calendar page the ratio
# measured 0.21, against 0.99 for a two-column article.
#
# Either signal is enough to call a page prose, because the two mistakes are not
# equal: reading a grid in the original order merely leaves it as it was, while
# reordering an article interleaves its columns into nonsense.
COLUMN_WIDTH_SIMILARITY = 0.7

# Tops within this many points count as the same row when sorting, so that two
# cells set fractionally apart still order left to right.
ROW_BAND = 2.0

# --- Label and value pairing ---------------------------------------------
# A document often states a fact as a short label beside or above a bare value:
# "Fees" then "30 pts". Layout analysis emits each as its own element, and once
# they are two elements nothing records that they belong together. Several such
# pairs in a row read as an undifferentiated list of words and numbers, and a
# model asked which number belongs to which label can only guess.
#
# The two are rejoined into a single element, "Fees: 30 pts", so the pairing
# survives chunking, embedding and the prompt.
#
# Both sides are held to a strict shape, because merging two elements that were
# never a pair invents a fact the document does not state.

# The label may be at most this many words. A label names a thing; anything
# longer is a sentence, and a sentence followed by a number is not a pair.
LABEL_MAX_WORDS = 4

# The gap between the label and its value, in points. Measured between the
# nearest edges, so it covers a value set beside its label as well as beneath
# it. Roughly one line of ordinary text - further apart than that and the two
# are separate items on the page.
LABEL_VALUE_MAX_GAP_POINTS = 20.0

# --- Embedding model -----------------------------------------------------
# Named here because the tokenizer used to measure chunk size must be the one
# the embedding model actually uses. Measuring with a different tokenizer means
# chunks are the wrong size in the only unit that matters.

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Hard input ceiling of the embedding model. Text beyond this is not truncated
# by us and not accepted by the model, so a chunk over this limit cannot be
# indexed at all. Tables are otherwise never split, but a table nobody can embed
# is worse than a table split at its row boundaries.
EMBED_MAX_INPUT_TOKENS = 8191

# Chunks sent per request. Batching is the difference between one call and one
# call per chunk: a 144-chunk document becomes two requests instead of 144.
# Kept well under the provider's per-request token budget so a document of large
# table chunks cannot overflow one batch.
EMBED_BATCH_SIZE = 64

# Transient failures - a dropped connection, a rate limit - are retried. A
# genuine error such as a bad key is not, because retrying it just delays the
# same failure three times over.
EMBED_MAX_RETRIES = 3
EMBED_RETRY_BACKOFF_SECONDS = 2.0

# --- Vector store --------------------------------------------------------
# Milvus Lite: a local file, no server to run.

MILVUS_PATH = VECTOR_DIR / "chunks.db"
MILVUS_COLLECTION = "chunks"

# Cosine, because the embeddings are already normalised, so only direction
# carries meaning and magnitude is noise.
#
# Read this before touching the gate: with the COSINE metric Milvus puts the
# cosine SIMILARITY in the field it calls `distance`. Higher is closer, and an
# identical vector scores +1.0, an unrelated one 0.0, an opposite one -1.0.
# Measured, not assumed. Subtracting it from 1 would inverse the gate.
MILVUS_METRIC = "COSINE"
MILVUS_INDEX_TYPE = "HNSW"

# Field widths. A section path can be deep, and a chunk can be a whole table.
MILVUS_SECTION_PATH_MAX = 1024
MILVUS_TEXT_MAX = 65535
MILVUS_ID_MAX = 80

# --- Chunking ------------------------------------------------------------
# Sizes are in tokens, never characters. The same passage can differ three or
# four times over in tokens depending on whether it is prose, a table of
# numbers, or a run of codes, so a character limit produces chunks of wildly
# different real sizes.

# Where a chunk is aimed. Large enough to keep its subject, small enough that
# the embedding describes one topic rather than the average of several.
CHUNK_TARGET_TOKENS = 500

# Text repeated from the end of the previous chunk. An answer that straddles a
# boundary is then whole in at least one chunk. 15% of target.
CHUNK_OVERLAP_TOKENS = 75

# Below this a chunk has lost its subject: "employees get 18 days" does not say
# of what, or for whom. A short trailing chunk is merged back into the one
# before it instead of being stored on its own.
CHUNK_MIN_TOKENS = 120

# Hard ceiling. A single element longer than this is split even though that
# means splitting inside a section.
CHUNK_MAX_TOKENS = 800

# Separator between the parts of a chunk's context header, and the header's
# page marker. The header is embedded with the chunk, so a question phrased in
# the heading's words matches even when the body uses different words.
CONTEXT_SEPARATOR = " > "
CONTEXT_PAGE_PREFIX = "p."

# --- Retrieval -----------------------------------------------------------
# Over-fetch, then narrow. Overlap deliberately makes neighbouring chunks
# near-duplicates, so asking for exactly what will be used spends slots on
# repeats of the same passage.

RETRIEVE_CANDIDATES = 12
CONTEXT_CHUNKS = 5

# --- The confidence gate -------------------------------------------------
# Compared against `Hit.similarity`, which is already oriented so that higher
# means closer. Never against a raw Milvus score.
#
# MEASURED, not chosen by taste. From 24 answerable and 17 unanswerable
# questions run against the real index:
#
#   answerable    lowest +0.496   mean +0.650   highest +0.787
#   unanswerable  lowest +0.392   mean +0.550   highest +0.723
#
# The two sets overlap, so no value separates them. What each costs:
#
#   0.45  refuses 0 of 24 real answers, lets 13 of 17 unanswerable through
#   0.48  refuses 0 of 24,              lets 12 of 17 through
#   0.60  refuses 7 of 24,              lets  7 of 17 through
#   0.74  refuses 20 of 24,             lets  0 of 17 through
#
# Set at 0.45, below the lowest real answer observed, for two reasons.
#
# First, the errors are not symmetrical in this design. An unanswerable question
# that gets past the gate meets a second line of defence - the prompt's
# abstention rule, which reads the chunks and can see the fact is absent. A real
# answer wrongly refused meets nothing; the user is simply told the documents do
# not cover something they do. Only one of the two errors is recoverable.
#
# Second, 24 questions is a small sample and 0.496 is its floor, not the
# population's. A threshold pressed against the lowest observed value would
# refuse legitimate questions that happen to fall below it. This leaves 0.046 of
# margin at the cost of one extra unanswerable question reaching the prompt.
#
# Raising it is the right response to off-topic questions being answered. It is
# the wrong response to on-topic questions with absent answers, which no
# threshold separates - those are the prompt's to refuse.
GATE_THRESHOLD = 0.45

# Two chunks are the same passage twice when one literally repeats a run of the
# other's words - which is exactly what the overlap window produces, since it
# copies whole sentences from one chunk into the next.
#
# Measured as the longest run of consecutive shared words, not as a proportion of
# vocabulary. A proportion collapses short chunks that merely share ordinary
# words: "The minimum password length is twelve characters" and "The session idle
# timeout is twenty minutes" have most of their vocabulary in common and are
# entirely different facts.
#
# The overlap window is 75 tokens, roughly 55 words, so a run this long is well
# inside a real overlap while being far too long to occur by coincidence.
DEDUPE_MIN_SHARED_WORDS = 12

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
