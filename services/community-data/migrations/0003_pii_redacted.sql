-- 0003: record the PII redaction outcome per submission.
--
-- Owner directive, 2026-10-03: the worker now STRIPS PII instead of
-- REJECTING. The redacted kind counts ride the response, the cleaned
-- summary goes into the existing `summary` column, and a JSON-shaped
-- `pii_redacted` column lets handleComplete decide whether to
-- sanitize the archive blob before publish (clean submissions skip
-- the rewrite; stripped ones re-store the rewritten archive or
-- withhold it on rewrite failure).
--
-- Rebuild-and-copy mirrors 0002_e2e_kind.sql so existing rows
-- survive. Default null = clean; a JSON object = the per-kind count
-- map returned in the initiate response.
CREATE TABLE submissions_new (
    content_sha256       TEXT PRIMARY KEY,
    received_at          INTEGER NOT NULL,
    updated_at           INTEGER NOT NULL,
    kind                 TEXT NOT NULL,
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
    published_at         INTEGER,
    pii_redacted         TEXT
);

INSERT INTO submissions_new
    SELECT content_sha256, received_at, updated_at, kind, schema_version,
           arch, model, chip, kernel, mesa_driver, mesa_device,
           mlx_version, mlx_device, summary, archive_total_bytes,
           archive_chunk_bytes, archive_chunk_count, archive_chunk_sha256,
           pow_difficulty, published, published_at, NULL
    FROM submissions;

DROP TABLE submissions;
ALTER TABLE submissions_new RENAME TO submissions;

CREATE INDEX idx_submissions_published
    ON submissions (published, published_at DESC);

CREATE INDEX idx_submissions_gc
    ON submissions (published, received_at);
