-- Lens application state. See LENS.md section 7.

CREATE TABLE IF NOT EXISTS documents (
    doc_id            TEXT PRIMARY KEY,
    display_name      TEXT    NOT NULL,
    original_filename TEXT    NOT NULL,
    content_hash      TEXT    NOT NULL UNIQUE,
    page_count        INTEGER,
    size_bytes        INTEGER NOT NULL,
    status            TEXT    NOT NULL,
    failure_reason    TEXT,
    image_count       INTEGER DEFAULT 0,
    table_count       INTEGER DEFAULT 0,
    chunk_count       INTEGER DEFAULT 0,
    chars_per_page    INTEGER,
    ocr_applied       INTEGER DEFAULT 0,
    embed_model       TEXT,
    visibility        TEXT    NOT NULL DEFAULT 'all',
    file_path         TEXT    NOT NULL,
    uploaded_at       TEXT    NOT NULL,
    deleted_at        TEXT
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id      TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    stage       TEXT NOT NULL,
    progress    REAL DEFAULT 0.0,
    message     TEXT,
    started_at  TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    conv_id       TEXT PRIMARY KEY,
    title         TEXT,
    title_is_auto INTEGER NOT NULL DEFAULT 1,
    scope_mode    TEXT    NOT NULL DEFAULT 'library',
    scope_doc_ids TEXT,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    msg_id         TEXT PRIMARY KEY,
    conv_id        TEXT NOT NULL REFERENCES conversations(conv_id),
    role           TEXT NOT NULL,
    content        TEXT NOT NULL,
    citations      TEXT,
    scope_snapshot TEXT,
    intent         TEXT,
    gate_passed    INTEGER,
    top_score      REAL,
    latency_ms     INTEGER,
    created_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv    ON messages(conv_id, created_at);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_conv_updated     ON conversations(updated_at DESC);
