import payloadSchemaJson from "../schema/payload-v1.schema.json";
import payloadE2ESchemaJson from "../schema/payload-v1-e2e.schema.json";
import {
  MAX_CHUNK_BYTES,
  MAX_PAYLOAD_BYTES,
  MIN_POW_BITS,
  expectedChunkLength,
} from "./caps";
import { sha256Hex } from "./hash";
import { coerceToSchema } from "./coerce";
import { ALIAS_HEADER, parseHostAliases, redactPiiPayload, scanPiiPayload } from "./pii";
import { sanitizeArchiveBlob } from "./archive_sanitize";
import { verifyPow } from "./pow";
import { SchemaNode, validateSchemaRoot } from "./schema";
import * as store from "./store";
import { checkInitiate, isSha256 } from "./validate";

const payloadSchema = payloadSchemaJson as SchemaNode;
const payloadE2ESchema = payloadE2ESchemaJson as SchemaNode;

// Stable identity for the bundled schema: recomputed by
// scripts/compute_schema_identity.py when the JSON schema changes. Surfaced
// via GET /v1/schema so stale deploys are caught at the wire instead of
// silently 422-ing every submission that carries a field the live worker
// does not know about. schema_versions lists every version the worker
// accepts (v1 rows stay readable; v2 adds the ANE turn-on blocks).
export const SCHEMA_IDENTITY = {
  schema_versions: [1, 2] as number[],
  fields_sha256:
    "47a5e33ad5d3356bb386f8075e59695d8e8e0901b6ec640920ea68dfdb679e4f",
  schema_sha256:
    "1e0da02331a359753d70638b5d1fb6e6183c77f6b800a742e80ed915ec87c6ea",
};

const CACHEABLE = "public, max-age=60";
const IMMUTABLE = "public, max-age=31536000, immutable";

function jsonResponse(status: number, data: unknown, headers?: Record<string, string>): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json",
      "access-control-allow-origin": "*",
      ...headers,
    },
  });
}

function errorResponse(status: number, code: string, detail?: unknown): Response {
  return jsonResponse(status, { error: code, detail: detail ?? null });
}

function notAllowed(): Response {
  return errorResponse(405, "method_not_allowed");
}

function receiptUrl(origin: string, sha: string): string {
  return `${origin}/v1/results/${sha}`;
}

type PayloadFields = {
  arch?: string | null;
  model?: string | null;
  chip?: string | null;
  kernel?: string | null;
  mesa_driver?: string | null;
  mesa_device?: string | null;
  mlx_version?: string | null;
  mlx_device?: string | null;
};

function textColumn(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

async function handleInitiate(request: Request, env: Env): Promise<Response> {
  const contentType = request.headers.get("content-type") ?? "";
  if (!contentType.includes("application/json")) {
    return errorResponse(415, "unsupported_media_type");
  }
  let body: unknown;
  try {
    body = JSON.parse(await request.text());
  } catch {
    return errorResponse(400, "bad_json");
  }
  const checked = checkInitiate(body);
  if (!checked.ok) {
    // Cap violations are 413 by name; every other shape problem is 400.
    const cap = checked.code === "archive_too_large" ||
      checked.code === "chunk_too_large" || checked.code === "too_many_chunks";
    return errorResponse(cap ? 413 : 400, checked.code);
  }
  const init = checked.value;

  const schema = init.payload.kind === "omarchy-mac-e2e" ? payloadE2ESchema : payloadSchema;
  // Tolerance for v0.7.14+ collectors (#24, #26, and the M3 422 of
  // 14:21Z): coerce hex/NUL encodings, then sanitize the diagnostic
  // blocks — anything still schema-invalid is parked under the root's
  // `unparsed` (capped) and dropped from the stored payload, so one
  // bad diagnostic value can never reject a submission. Schema identity
  // is recomputed from the canonical files; every action is logged and
  // echoed as coerced.
  const { changes: coerced, unparsed } = coerceToSchema(init.payload, schema);
  if (coerced.length > 0) {
    console.log(
      `coerced payload fields (${coerced.length}, unparked ${Object.keys(unparsed).length}):`,
      JSON.stringify(coerced),
    );
  }
  const audit = coerced.length > 0 ? { coerced } : {};

  const schemaErrors = validateSchemaRoot(init.payload, schema);
  if (schemaErrors.length > 0) {
    return errorResponse(422, "schema_invalid", {
      errors: schemaErrors.slice(0, 20),
      ...audit,
    });
  }

  const pow = await verifyPow(init.content_sha256, (body as Record<string, unknown>).pow, MIN_POW_BITS);
  if (!pow.ok) {
    const status = pow.code === "pow_missing" ? 400 : 403;
    return errorResponse(status, pow.code, { min_difficulty: MIN_POW_BITS });
  }

  const pii = scanPiiPayload(init.payload, parseHostAliases(request.headers.get(ALIAS_HEADER)));
  let piiRedacted: Record<string, number> | null = null;
  let storedPayload: unknown = init.payload;
  if (pii !== null) {
    // Owner directive 2026-10-03: STRIP PII instead of REJECTING. The
    // worker scrubs matched substrings in the payload, stores the
    // cleaned copy, and returns the redaction counts to the client.
    // Hard rejects (size, schema, schema_version, identity) are
    // unchanged above.
    try {
      const out = redactPiiPayload(init.payload, parseHostAliases(request.headers.get(ALIAS_HEADER)));
      storedPayload = out.payload;
      piiRedacted = out.kinds;
    } catch (exc) {
      // redactPiiPayload rewrites per string value, so it cannot
      // produce invalid JSON; any throw is a bug — fail closed so
      // the raw payload never lands in storage.
      return errorResponse(500, "pii_redaction_failed",
        { reason: (exc as Error).message });
    }
    console.log(`pii_redacted ${init.content_sha256} ${JSON.stringify(piiRedacted)}`);
  }

  // Sized AFTER the strip: the stored summary is what must fit.
  const summaryText = JSON.stringify(storedPayload);
  if (summaryText.length > MAX_PAYLOAD_BYTES) {
    return errorResponse(413, "payload_too_large", { limit: MAX_PAYLOAD_BYTES });
  }

  const now = Math.floor(Date.now() / 1000);
  const fields: PayloadFields = init.payload as PayloadFields;
  const existing = await store.getSubmission(env.DB, init.content_sha256);

  if (existing !== null && existing.published === 1) {
    return jsonResponse(200, {
      status: "duplicate",
      content_sha256: init.content_sha256,
      missing_chunks: [],
      receipt_url: receiptUrl(new URL(request.url).origin, init.content_sha256),
      ...(piiRedacted ? { pii_redacted: piiRedacted } : {}),
      ...audit,
    });
  }

  if (existing !== null) {
    const storedShape = {
      total_bytes: existing.archive_total_bytes,
      chunk_bytes: existing.archive_chunk_bytes,
      chunk_count: existing.archive_chunk_count,
    };
    const newShape = init.archive
      ? {
          total_bytes: init.archive.total_bytes,
          chunk_bytes: init.archive.chunk_bytes,
          chunk_count: init.archive.chunk_count,
        }
      : { total_bytes: null, chunk_bytes: null, chunk_count: null };
    if (JSON.stringify(storedShape) !== JSON.stringify(newShape)) {
      return errorResponse(409, "archive_conflict");
    }
    await store.touchIncomplete(env.DB, init.content_sha256, now);
    const missing = await store.missingChunks(
      env.DB,
      init.content_sha256,
      init.archive?.chunk_count ?? 0,
    );
    return jsonResponse(200, {
      status: "awaiting_chunks",
      content_sha256: init.content_sha256,
      missing_chunks: missing,
      receipt_url: receiptUrl(new URL(request.url).origin, init.content_sha256),
      ...(piiRedacted ? { pii_redacted: piiRedacted } : {}),
      ...audit,
    });
  }

  const row = await store.initiateSubmission(env.DB, {
    contentSha: init.content_sha256,
    now,
    kind: init.kind,
    schemaVersion: init.schema_version,
    arch: textColumn(fields.arch),
    model: textColumn(fields.model),
    chip: textColumn(fields.chip),
    kernel: textColumn(fields.kernel),
    mesaDriver: textColumn(fields.mesa_driver),
    mesaDevice: textColumn(fields.mesa_device),
    mlxVersion: textColumn(fields.mlx_version),
    mlxDevice: textColumn(fields.mlx_device),
    summary: summaryText,
    archive: init.archive,
    powDifficulty: pow.difficulty,
    piiRedacted: piiRedacted,
  });
  if (row.published === 1) {
    // A published row must be visible to readers immediately; the hourly
    // cron is only the safety net.
    await store.rebuildCaches(env.DB, now);
  }
  const missing = row.published === 1
    ? []
    : await store.missingChunks(env.DB, init.content_sha256, init.archive?.chunk_count ?? 0);
  return jsonResponse(200, {
    status: row.published === 1 ? "stored" : "awaiting_chunks",
    content_sha256: init.content_sha256,
    missing_chunks: missing,
    receipt_url: receiptUrl(new URL(request.url).origin, init.content_sha256),
    ...(piiRedacted ? { pii_redacted: piiRedacted } : {}),
    ...audit,
  });
}

async function handleChunk(
  sha: string,
  idxText: string,
  request: Request,
  env: Env,
): Promise<Response> {
  const idx = Number(idxText);
  if (!Number.isInteger(idx) || idx < 0) {
    return errorResponse(400, "chunk_index_invalid");
  }
  const row = await store.getSubmission(env.DB, sha);
  if (row === null) return errorResponse(404, "not_found");
  if (row.published === 1) return errorResponse(409, "already_published");
  if (row.archive_chunk_count === null || row.archive_chunk_bytes === null) {
    return errorResponse(409, "no_archive");
  }
  if (idx >= row.archive_chunk_count) {
    return errorResponse(400, "chunk_index_invalid");
  }
  const bytes = new Uint8Array(await request.arrayBuffer());
  if (bytes.length > MAX_CHUNK_BYTES) {
    return errorResponse(413, "chunk_too_large", { limit: MAX_CHUNK_BYTES });
  }
  const expected = expectedChunkLength(
    row.archive_total_bytes ?? 0,
    row.archive_chunk_bytes,
    idx,
  );
  if (bytes.length !== expected) {
    return errorResponse(400, "chunk_size_invalid", { expected, got: bytes.length });
  }
  const chunkHash = await sha256Hex(bytes);
  const declared: string[] = JSON.parse(row.archive_chunk_sha256 ?? "[]");
  if (declared[idx] !== undefined && declared[idx] !== chunkHash) {
    return errorResponse(422, "chunk_hash_mismatch", { idx });
  }
  const outcome = await store.putChunk(env.DB, sha, idx, chunkHash, bytes);
  if (!outcome.ok) {
    return errorResponse(409, "chunk_hash_conflict", { idx });
  }
  const missing = await store.missingChunks(env.DB, sha, row.archive_chunk_count);
  return jsonResponse(200, { status: outcome.status, idx, missing_chunks: missing });
}

async function handleComplete(sha: string, request: Request, env: Env): Promise<Response> {
  // Sanitize the archive blob BEFORE publishing, but only when the
  // initiate step recorded a PII strip — clean submissions publish
  // exactly as uploaded (idempotent re-rewrite would change chunk
  // hashes and break handleChunk's declared-vs-stored comparison).
  const row = await store.getSubmission(env.DB, sha);
  let archiveWithheld = false;
  if (row !== null && row.pii_redacted !== null) {
    const chunks = await store.archiveChunks(env.DB, sha);
    if (chunks.length > 0) {
      const totalLen = chunks.reduce((n, b) => n + b.byteLength, 0);
      const assembled = new Uint8Array(totalLen);
      let off = 0;
      for (const c of chunks) { assembled.set(new Uint8Array(c), off); off += c.byteLength; }
      const out = await sanitizeArchiveBlob(assembled);
      if (out.ok) {
        const newTotal = out.bytes.byteLength;
        const chunkBytes = row.archive_chunk_bytes ?? newTotal;
        const newCount = Math.ceil(newTotal / chunkBytes);
        const newHashes: string[] = [];
        for (let i = 0; i < newCount; i++) {
          const start = i * chunkBytes;
          const end = Math.min(newTotal, start + chunkBytes);
          newHashes.push(await sha256Hex(out.bytes.subarray(start, end)));
        }
        await store.replaceArchive(env.DB, sha,
          out.bytes.buffer.slice(out.bytes.byteOffset, out.bytes.byteOffset + out.bytes.byteLength),
          { total_bytes: newTotal, chunk_bytes: chunkBytes,
            chunk_count: newCount, chunk_sha256: newHashes });
        console.log(`pii_archive_rewritten ${sha} ${newTotal} bytes`);
      } else {
        await store.withholdArchive(env.DB, sha);
        archiveWithheld = true;
        console.log(`pii_archive_withheld ${sha} code=${out.code} reason=${out.reason}`);
      }
    }
  }
  const outcome = await store.completeSubmission(
    env.DB,
    sha,
    Math.floor(Date.now() / 1000),
  );
  if (!outcome.ok) {
    if (outcome.code === "not_found") return errorResponse(404, "not_found");
    if (outcome.code === "incomplete") {
      return errorResponse(409, "incomplete", { missing_chunks: outcome.missing });
    }
    return errorResponse(422, "chunk_hash_mismatch", { idx: outcome.idx });
  }
  // Publishing changed the result set, so the cached index and dataset
  // must be rebuilt now rather than at the next cron tick.
  await store.rebuildCaches(env.DB, Math.floor(Date.now() / 1000));
  return jsonResponse(200, {
    status: outcome.status,
    content_sha256: sha,
    receipt_url: receiptUrl(new URL(request.url).origin, sha),
    ...(archiveWithheld ? { archive_withheld_pii: true } : {}),
  });
}

async function handleProbe(sha: string, request: Request, env: Env): Promise<Response> {
  const row = await store.getPublished(env.DB, sha);
  if (row === null) return errorResponse(404, "not_found");
  return jsonResponse(200, {
    status: "duplicate",
    receipt_url: receiptUrl(new URL(request.url).origin, sha),
  });
}

async function handleResultOne(sha: string, request: Request, env: Env): Promise<Response> {
  const row = await store.getPublished(env.DB, sha);
  if (row === null) return errorResponse(404, "not_found");
  return jsonResponse(200, {
    content_sha256: row.content_sha256,
    kind: row.kind,
    schema_version: row.schema_version,
    arch: row.arch,
    model: row.model,
    chip: row.chip,
    kernel: row.kernel,
    mesa_driver: row.mesa_driver,
    mesa_device: row.mesa_device,
    mlx_version: row.mlx_version,
    mlx_device: row.mlx_device,
    summary: JSON.parse(row.summary),
    archive:
      row.archive_chunk_count === null
        ? null
        : {
            total_bytes: row.archive_total_bytes,
            chunk_bytes: row.archive_chunk_bytes,
            chunk_count: row.archive_chunk_count,
          },
    received_at: new Date(row.received_at * 1000).toISOString(),
    published_at:
      row.published_at === null
        ? null
        : new Date(row.published_at * 1000).toISOString(),
    receipt_url: receiptUrl(new URL(request.url).origin, sha),
  });
}

async function handleArchive(sha: string, env: Env): Promise<Response> {
  const row = await store.getPublished(env.DB, sha);
  if (row === null || row.archive_chunk_count === null) {
    return errorResponse(404, "not_found");
  }
  const chunks = await store.archiveChunks(env.DB, sha);
  if (chunks.length !== row.archive_chunk_count) {
    return errorResponse(500, "archive_unavailable");
  }
  const total = row.archive_total_bytes ?? 0;
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(new Uint8Array(chunk));
      }
      controller.close();
    },
  });
  return new Response(stream, {
    headers: {
      "content-type": "application/octet-stream",
      "content-length": String(total),
      "cache-control": IMMUTABLE,
      "access-control-allow-origin": "*",
    },
  });
}

async function serveCache(env: Env, key: string, contentType: string): Promise<Response> {
  let cache = await store.getCache(env.DB, key);
  if (cache === null) {
    // Self-heal before the first cron fires: build once on demand.
    await store.rebuildCaches(env.DB, Math.floor(Date.now() / 1000));
    cache = await store.getCache(env.DB, key);
  }
  if (cache === null) return errorResponse(500, "cache_unavailable");
  return new Response(cache.text, {
    headers: {
      "content-type": contentType,
      "cache-control": CACHEABLE,
      "access-control-allow-origin": "*",
      "x-generated-at": cache.generated_at,
    },
  });
}

function handleSchema(): Response {
  return jsonResponse(200, {
    schema_versions: SCHEMA_IDENTITY.schema_versions,
    fields_sha256: SCHEMA_IDENTITY.fields_sha256,
    schema_sha256: SCHEMA_IDENTITY.schema_sha256,
  }, { "cache-control": "public, max-age=300" });
}

export async function handleFetch(request: Request, env: Env): Promise<Response> {
  const { pathname } = new URL(request.url);

  if (pathname === "/v1/schema") {
    if (request.method !== "GET") return notAllowed();
    return handleSchema();
  }

  if (pathname === "/v1/submit") {
    if (request.method !== "POST") return notAllowed();
    return handleInitiate(request, env);
  }

  let match = pathname.match(/^\/v1\/submit\/([0-9a-f]{64})\/chunk\/(\d+)$/);
  if (match) {
    if (request.method !== "POST") return notAllowed();
    return handleChunk(match[1], match[2], request, env);
  }

  match = pathname.match(/^\/v1\/submit\/([0-9a-f]{64})\/complete$/);
  if (match) {
    if (request.method !== "POST") return notAllowed();
    return handleComplete(match[1], request, env);
  }

  match = pathname.match(/^\/v1\/submit\/([0-9a-f]{64})$/);
  if (match) {
    if (request.method !== "GET") return notAllowed();
    return handleProbe(match[1], request, env);
  }

  match = pathname.match(/^\/v1\/results\/([0-9a-f]{64})\/archive$/);
  if (match) {
    if (request.method !== "GET") return notAllowed();
    return handleArchive(match[1], env);
  }

  match = pathname.match(/^\/v1\/results\/([0-9a-f]{64})$/);
  if (match) {
    if (request.method !== "GET") return notAllowed();
    return handleResultOne(match[1], request, env);
  }

  if (pathname === "/v1/results") {
    if (request.method !== "GET") return notAllowed();
    return serveCache(env, "results", "application/json");
  }

  if (pathname === "/v1/dataset/latest.jsonl") {
    if (request.method !== "GET") return notAllowed();
    return serveCache(env, "dataset", "application/x-ndjson");
  }

  return errorResponse(404, "not_found");
}

