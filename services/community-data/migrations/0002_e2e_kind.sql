-- Widen the submissions kind CHECK to accept every collector kind.
--
-- 0001 pinned the CHECK to ('quick', 'deep'). Adding the Omarchy Mac E2E
-- collector kind ('omarchy-mac-e2e') validated at the API layer but hit
-- this table CHECK: INSERT OR IGNORE silently dropped the row and the
-- caller saw "insert succeeded but row is missing" (500).
--
-- The API validator (src/validate.ts) is the real kind gate and is
-- already kind-enum-checked, so the database drops its own copy of the
-- list instead of gaining a fourth place to edit on the next kind.
-- Rebuild-and-copy keeps every existing row, and the two indexes are
-- recreated identically.
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
    published_at         INTEGER
);

INSERT INTO submissions_new
    SELECT content_sha256, received_at, updated_at, kind, schema_version,
           arch, model, chip, kernel, mesa_driver, mesa_device,
           mlx_version, mlx_device, summary, archive_total_bytes,
           archive_chunk_bytes, archive_chunk_count, archive_chunk_sha256,
           pow_difficulty, published, published_at
    FROM submissions;

DROP TABLE submissions;
ALTER TABLE submissions_new RENAME TO submissions;

CREATE INDEX idx_submissions_published
    ON submissions (published, published_at DESC);

CREATE INDEX idx_submissions_gc
    ON submissions (published, received_at);
