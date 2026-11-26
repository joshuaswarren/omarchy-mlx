-- mlx-omarchy community data, initial schema.
--
-- submissions: one row per content hash. summary holds the normalized
-- (minified, validated) JSON summary so cache builds never re-parse JSON.
-- Unpublished rows are incomplete uploads; no read route may serve them.
-- chunks: one row per archive chunk so no row approaches the 2 MB D1
-- limit. 768 KB chunks x 11 max = 8 MB archives stay far under it.
-- cache_meta / cache_parts: cron-built responses for the bulk read
-- routes, split into ~500 KB parts so no row approaches 2 MB.
CREATE TABLE submissions (
    content_sha256       TEXT PRIMARY KEY,
    received_at          INTEGER NOT NULL,
    updated_at           INTEGER NOT NULL,
    kind                 TEXT NOT NULL CHECK (kind IN ('quick', 'deep')),
    schema_version       INTEGER NOT NULL,
    arch                 TEXT,
    model                TEXT,
    chip                 TEXT,
    kernel               TEXT,
    mesa_driver          TEXT,
    mesa_device          TEXT,
    mlx_version          TEXT,
    mlx_device           TEXT,
    summary              TEXT NOT NULL,
    archive_total_bytes  INTEGER,
    archive_chunk_bytes  INTEGER,
    archive_chunk_count  INTEGER,
    archive_chunk_sha256 TEXT,
    pow_difficulty       INTEGER NOT NULL,
    published            INTEGER NOT NULL DEFAULT 0,
    published_at         INTEGER
);

CREATE TABLE chunks (
    content_sha256 TEXT NOT NULL,
    idx            INTEGER NOT NULL,
    chunk_sha256   TEXT NOT NULL,
    bytes          BLOB NOT NULL,
    PRIMARY KEY (content_sha256, idx)
);

CREATE TABLE cache_meta (
    cache_key    TEXT PRIMARY KEY,
    built_at     INTEGER NOT NULL,
    generated_at TEXT NOT NULL,
    count        INTEGER NOT NULL,
    parts        INTEGER NOT NULL
);

CREATE TABLE cache_parts (
    cache_key TEXT NOT NULL,
    part_idx  INTEGER NOT NULL,
    text      TEXT NOT NULL,
    PRIMARY KEY (cache_key, part_idx)
);

-- Read routes list published submissions, newest first.
CREATE INDEX idx_submissions_published
    ON submissions (published, published_at DESC);

-- GC sweep of incomplete submissions past the retention window.
CREATE INDEX idx_submissions_gc
    ON submissions (published, received_at);
