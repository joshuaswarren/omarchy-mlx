import { CACHE_PART_CHARS, SCHEMA_VERSIONS } from "./caps";

export interface SubmissionRow {
  content_sha256: string;
  received_at: number;
  updated_at: number;
  kind: string;
  schema_version: number;
  arch: string | null;
  model: string | null;
  chip: string | null;
  kernel: string | null;
  mesa_driver: string | null;
  mesa_device: string | null;
  mlx_version: string | null;
  mlx_device: string | null;
  summary: string;
  archive_total_bytes: number | null;
  archive_chunk_bytes: number | null;
  archive_chunk_count: number | null;
  archive_chunk_sha256: string | null;
  pow_difficulty: number;
  published: number;
  published_at: number | null;
}

const SUBMISSION_COLS =
  "content_sha256, received_at, updated_at, kind, schema_version, arch, " +
  "model, chip, kernel, mesa_driver, mesa_device, mlx_version, mlx_device, " +
  "summary, archive_total_bytes, archive_chunk_bytes, archive_chunk_count, " +
  "archive_chunk_sha256, pow_difficulty, published, published_at";

export type PublishOutcome =
  | { ok: true; status: "stored" | "duplicate" }
  | { ok: false; code: "not_found" }
  | { ok: false; code: "incomplete"; missing: number[] }
  | { ok: false; code: "chunk_hash_mismatch"; idx: number };

export async function getSubmission(
  db: D1Database,
  sha: string,
): Promise<SubmissionRow | null> {
  const row = await db
    .prepare(`SELECT ${SUBMISSION_COLS} FROM submissions WHERE content_sha256 = ?1`)
    .bind(sha)
    .first<SubmissionRow>();
  return row ?? null;
}

export async function getPublished(
  db: D1Database,
  sha: string,
): Promise<SubmissionRow | null> {
  const row = await db
    .prepare(
      `SELECT ${SUBMISSION_COLS} FROM submissions
       WHERE content_sha256 = ?1 AND published = 1`,
    )
    .bind(sha)
    .first<SubmissionRow>();
  return row ?? null;
}

export type InitiateFields = {
  contentSha: string;
  now: number;
  kind: string;
  schemaVersion: number;
  arch: string | null;
  model: string | null;
  chip: string | null;
  kernel: string | null;
  mesaDriver: string | null;
  mesaDevice: string | null;
  mlxVersion: string | null;
  mlxDevice: string | null;
  summary: string;
  archive: {
    total_bytes: number;
    chunk_bytes: number;
    chunk_count: number;
    chunk_sha256: string[];
  } | null;
  powDifficulty: number;
};

/**
 * Content-addressed insert. Replays land on the same row, so initiation
 * is idempotent. A summary-only submission publishes immediately.
 */
export async function initiateSubmission(
  db: D1Database,
  f: InitiateFields,
): Promise<SubmissionRow> {
  await db
    .prepare(
      `INSERT OR IGNORE INTO submissions (
         content_sha256, received_at, updated_at, kind, schema_version,
         arch, model, chip, kernel, mesa_driver, mesa_device, mlx_version,
         mlx_device, summary, archive_total_bytes, archive_chunk_bytes,
         archive_chunk_count, archive_chunk_sha256, pow_difficulty,
         published, published_at
       ) VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                 ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)`,
    )
    .bind(
      f.contentSha,
      f.now,
      f.kind,
      f.schemaVersion,
      f.arch,
      f.model,
      f.chip,
      f.kernel,
      f.mesaDriver,
      f.mesaDevice,
      f.mlxVersion,
      f.mlxDevice,
      f.summary,
      f.archive?.total_bytes ?? null,
      f.archive?.chunk_bytes ?? null,
      f.archive?.chunk_count ?? null,
      f.archive ? JSON.stringify(f.archive.chunk_sha256) : null,
      f.powDifficulty,
      f.archive ? 0 : 1,
      f.archive ? null : f.now,
    )
    .run();
  const row = await getSubmission(db, f.contentSha);
  if (row === null) {
    throw new Error("initiate: insert succeeded but row is missing");
  }
  return row;
}

/** Refresh the GC clock when an incomplete upload is resumed. */
export async function touchIncomplete(
  db: D1Database,
  sha: string,
  now: number,
): Promise<void> {
  await db
    .prepare(
      `UPDATE submissions SET received_at = ?2, updated_at = ?2
       WHERE content_sha256 = ?1 AND published = 0`,
    )
    .bind(sha, now)
    .run();
}

export async function storedChunkIndexes(
  db: D1Database,
  sha: string,
): Promise<Set<number>> {
  const { results } = await db
    .prepare("SELECT idx FROM chunks WHERE content_sha256 = ?1")
    .bind(sha)
    .all<{ idx: number }>();
  const present = new Set<number>();
  for (const row of results) present.add(row.idx);
  return present;
}

export async function missingChunks(
  db: D1Database,
  sha: string,
  chunkCount: number,
): Promise<number[]> {
  const present = await storedChunkIndexes(db, sha);
  const missing: number[] = [];
  for (let idx = 0; idx < chunkCount; idx++) {
    if (!present.has(idx)) missing.push(idx);
  }
  return missing;
}

export type ChunkOutcome =
  | { ok: true; status: "stored" | "duplicate" }
  | { ok: false; code: "chunk_hash_conflict" };

export async function putChunk(
  db: D1Database,
  sha: string,
  idx: number,
  chunkHash: string,
  bytes: Uint8Array,
): Promise<ChunkOutcome> {
  const result = await db
    .prepare(
      `INSERT OR IGNORE INTO chunks (content_sha256, idx, chunk_sha256, bytes)
       VALUES (?1, ?2, ?3, ?4)`,
    )
    .bind(sha, idx, chunkHash, bytes)
    .run();
  if ((result.meta.changes ?? 0) === 1) return { ok: true, status: "stored" };
  const existing = await db
    .prepare("SELECT chunk_sha256 FROM chunks WHERE content_sha256 = ?1 AND idx = ?2")
    .bind(sha, idx)
    .first<{ chunk_sha256: string }>();
  if (existing?.chunk_sha256 === chunkHash) {
    return { ok: true, status: "duplicate" };
  }
  return { ok: false, code: "chunk_hash_conflict" };
}

/**
 * Publish only when every declared chunk is stored and each stored
 * chunk hash matches the hash the initiate call declared. Hashes were
 * computed when the bytes arrived, so this check is string comparison
 * only: no re-hashing, and well inside the 10 ms CPU budget.
 */
export async function completeSubmission(
  db: D1Database,
  sha: string,
  now: number,
): Promise<PublishOutcome> {
  const row = await getSubmission(db, sha);
  if (row === null) return { ok: false, code: "not_found" };
  if (row.published === 1) return { ok: true, status: "duplicate" };
  if (row.archive_chunk_count === null || row.archive_chunk_sha256 === null) {
    return { ok: false, code: "incomplete", missing: [] };
  }
  const declared: string[] = JSON.parse(row.archive_chunk_sha256);
  const { results } = await db
    .prepare(
      "SELECT idx, chunk_sha256 FROM chunks WHERE content_sha256 = ?1 ORDER BY idx",
    )
    .bind(sha)
    .all<{ idx: number; chunk_sha256: string }>();
  const stored = new Map<number, string>(results.map((r) => [r.idx, r.chunk_sha256]));
  const missing: number[] = [];
  for (let idx = 0; idx < declared.length; idx++) {
    if (!stored.has(idx)) missing.push(idx);
  }
  if (missing.length > 0) return { ok: false, code: "incomplete", missing };
  for (const [idx, storedHash] of stored) {
    if (declared[idx] !== storedHash) {
      return { ok: false, code: "chunk_hash_mismatch", idx };
    }
  }
  await db
    .prepare(
      `UPDATE submissions SET published = 1, published_at = ?2, updated_at = ?2
       WHERE content_sha256 = ?1`,
    )
    .bind(sha, now)
    .run();
  return { ok: true, status: "stored" };
}

export async function archiveChunks(
  db: D1Database,
  sha: string,
): Promise<ArrayBuffer[]> {
  const { results } = await db
    .prepare(
      "SELECT bytes FROM chunks WHERE content_sha256 = ?1 ORDER BY idx",
    )
    .bind(sha)
    .all<{ bytes: ArrayBuffer }>();
  return results.map((r) => r.bytes);
}

// ---------------------------------------------------------------------------
// Cron-built caches for the bulk read routes.
// ---------------------------------------------------------------------------

export interface CacheEntry {
  generated_at: string;
  count: number;
  text: string;
}

export async function getCache(
  db: D1Database,
  key: string,
): Promise<CacheEntry | null> {
  const meta = await db
    .prepare(
      "SELECT generated_at, count FROM cache_meta WHERE cache_key = ?1",
    )
    .bind(key)
    .first<{ generated_at: string; count: number }>();
  if (meta === null) return null;
  const { results } = await db
    .prepare(
      "SELECT text FROM cache_parts WHERE cache_key = ?1 ORDER BY part_idx",
    )
    .bind(key)
    .all<{ text: string }>();
  return {
    generated_at: meta.generated_at,
    count: meta.count,
    text: results.map((r) => r.text).join(""),
  };
}

async function saveCache(
  db: D1Database,
  key: string,
  entry: CacheEntry,
  now: number,
): Promise<void> {
  const parts: D1PreparedStatement[] = [
    db.prepare("DELETE FROM cache_parts WHERE cache_key = ?1").bind(key),
    db.prepare("DELETE FROM cache_meta WHERE cache_key = ?1").bind(key),
    db
      .prepare(
        `INSERT INTO cache_meta (cache_key, built_at, generated_at, count, parts)
         VALUES (?1, ?2, ?3, ?4, ?5)`,
      )
      .bind(key, now, entry.generated_at, entry.count, Math.ceil(entry.text.length / CACHE_PART_CHARS)),
  ];
  for (let i = 0; i * CACHE_PART_CHARS < entry.text.length; i++) {
    parts.push(
      db
        .prepare(
          "INSERT INTO cache_parts (cache_key, part_idx, text) VALUES (?1, ?2, ?3)",
        )
        .bind(key, i, entry.text.slice(i * CACHE_PART_CHARS, (i + 1) * CACHE_PART_CHARS)),
    );
  }
  await db.batch(parts);
}

/**
 * Strip a single top-level JSON object field from a JSON text, preserving
 * the trailing brace. Brace-aware: walks the value (object or array)
 * using depth counting so nested braces inside the field's value do
 * not break the splice. Returns the input unchanged when the field is
 * absent, which is the common case for older rows.
 *
 * Per-row detail (`ane_port_detail`) is large and bounded only by the
 * per-payload byte cap. Keeping it in every cached index row would
 * balloon /v1/results and /v1/dataset/latest.jsonl; the per-row record
 * at /v1/results/<sha> still serves it verbatim from the DB column.
 */
function stripJsonField(text: string, field: string): string {
  const needle = `"${field}"`;
  let i = text.indexOf(needle);
  if (i < 0) return text;
  // Walk back to the preceding comma or opening brace; the field must
  // be at the top level so the character right before is one of those.
  let j = i - 1;
  while (j >= 0 && /\s/.test(text[j])) j--;
  if (j < 0 || (text[j] !== "," && text[j] !== "{")) return text;
  // Find the colon after the field name, then the value start.
  let k = i + needle.length;
  while (k < text.length && /\s/.test(text[k])) k++;
  if (text[k] !== ":") return text;
  k++;
  while (k < text.length && /\s/.test(text[k])) k++;
  if (k >= text.length) return text;
  let depth = 0;
  let inString = false;
  let escape = false;
  let valueEnd = -1;
  if (text[k] === "{") {
    depth = 1;
    let m = k + 1;
    while (m < text.length && depth > 0) {
      const ch = text[m];
      if (inString) {
        if (escape) escape = false;
        else if (ch === "\\") escape = true;
        else if (ch === '"') inString = false;
      } else if (ch === '"') {
        inString = true;
      } else if (ch === "{") {
        depth++;
      } else if (ch === "}") {
        depth--;
      }
      m++;
    }
    valueEnd = m;
  } else if (text[k] === "[") {
    depth = 1;
    let m = k + 1;
    while (m < text.length && depth > 0) {
      const ch = text[m];
      if (inString) {
        if (escape) escape = false;
        else if (ch === "\\") escape = true;
        else if (ch === '"') inString = false;
      } else if (ch === '"') {
        inString = true;
      } else if (ch === "[") {
        depth++;
      } else if (ch === "]") {
        depth--;
      }
      m++;
    }
    valueEnd = m;
  } else {
    // Scalar (string/number/bool/null): read until comma or closing brace.
    let m = k;
    if (text[k] === '"') {
      inString = true;
      m = k + 1;
      while (m < text.length) {
        const ch = text[m];
        if (escape) escape = false;
        else if (ch === "\\") escape = true;
        else if (ch === '"') {
          inString = false;
          m++;
          break;
        }
        m++;
      }
    } else {
      while (m < text.length && text[m] !== "," && text[m] !== "}") m++;
    }
    valueEnd = m;
  }
  if (valueEnd < 0) return text;
  // Splice out [j+1, valueEnd) inclusive of leading whitespace so the
  // surrounding commas collapse correctly.
  return text.slice(0, j) + text.slice(valueEnd);
}

/**
 * Rebuild both cached responses from stored summaries. Summaries are
 * already normalized JSON text, so this is string concatenation: no
 * JSON parsing, safe for the cron CPU budget even with many rows.
 */
export async function rebuildCaches(db: D1Database, now: number): Promise<number> {
  const { results } = await db
    .prepare(
      `SELECT content_sha256, summary FROM submissions WHERE published = 1
       ORDER BY published_at, content_sha256`,
    )
    .all<{ content_sha256: string; summary: string }>();
  // Each cached entry is the stored summary with the content hash
  // prepended, so bulk consumers can address /v1/results/<sha> and
  // fetch the archive. String splice only: no JSON re-parsing. The
  // per-row detail (`ane_port_detail`) is stripped from the cached
  // index/dataset so /v1/results and /v1/dataset/latest.jsonl stay
  // small; the per-row route at /v1/results/<sha> serves it directly
  // from the DB column.
  const lines = results.map((r) => {
    const stripped = stripJsonField(r.summary, "ane_port_detail");
    return `{"content_sha256":"${r.content_sha256}",${stripped.slice(1)}`;
  });
  const generatedAt = new Date(now * 1000).toISOString();
  await saveCache(
    db,
    "results",
    {
      generated_at: generatedAt,
      count: lines.length,
      text: `{"generated_at":"${generatedAt}","schema_versions":${JSON.stringify(SCHEMA_VERSIONS)},"count":${lines.length},"results":[${lines.join(",")}]`,
    },
    now,
  );
  await saveCache(
    db,
    "dataset",
    {
      generated_at: generatedAt,
      count: lines.length,
      text: lines.length > 0 ? lines.join("\n") + "\n" : "",
    },
    now,
  );
  return lines.length;
}

export async function gcStale(
  db: D1Database,
  cutoff: number,
): Promise<{ submissions: number; chunks: number }> {
  // Chunks first: the subselect reads the submission rows it deletes.
  const chunks = await db
    .prepare(
      `DELETE FROM chunks WHERE content_sha256 IN (
         SELECT content_sha256 FROM submissions WHERE published = 0 AND received_at < ?1
       )`,
    )
    .bind(cutoff)
    .run();
  const subs = await db
    .prepare(
      `DELETE FROM submissions WHERE published = 0 AND received_at < ?1`,
    )
    .bind(cutoff)
    .run();
  return {
    submissions: subs.meta.changes ?? 0,
    chunks: chunks.meta.changes ?? 0,
  };
}
