import {
  MAX_ARCHIVE_BYTES,
  MAX_CHUNKS,
  MAX_CHUNK_BYTES,
  SCHEMA_VERSIONS,
  expectedChunkCount,
} from "./caps";

const SHA256_RE = /^[0-9a-f]{64}$/;

export type InitiateArchive = {
  total_bytes: number;
  chunk_bytes: number;
  chunk_count: number;
  chunk_sha256: string[];
};

export type InitiateBody = {
  schema_version: number;
  kind: "quick" | "deep" | "omarchy-mac-e2e";
  content_sha256: string;
  payload: unknown;
  archive: InitiateArchive | null;
};

export function isSha256(value: unknown): value is string {
  return typeof value === "string" && SHA256_RE.test(value);
}

export type Check<T> = { ok: true; value: T } | { ok: false; code: string };

function isInt(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

export function checkArchive(archive: unknown): Check<InitiateArchive> {
  if (archive === null || typeof archive !== "object") {
    return { ok: false, code: "archive_invalid" };
  }
  const a = archive as Record<string, unknown>;
  if (!isInt(a.total_bytes) || a.total_bytes <= 0) {
    return { ok: false, code: "archive_invalid" };
  }
  if (!isInt(a.chunk_bytes) || a.chunk_bytes <= 0 || a.chunk_bytes > MAX_CHUNK_BYTES) {
    return { ok: false, code: "chunk_too_large" };
  }
  if (!Array.isArray(a.chunk_sha256) || !a.chunk_sha256.every(isSha256)) {
    return { ok: false, code: "archive_invalid" };
  }
  if (a.total_bytes > MAX_ARCHIVE_BYTES) {
    return { ok: false, code: "archive_too_large" };
  }
  if (!isInt(a.chunk_count) || a.chunk_count !== expectedChunkCount(a.total_bytes, a.chunk_bytes)) {
    return { ok: false, code: "chunk_count_invalid" };
  }
  if (a.chunk_count > MAX_CHUNKS || a.chunk_sha256.length !== a.chunk_count) {
    return { ok: false, code: "too_many_chunks" };
  }
  return {
    ok: true,
    value: {
      total_bytes: a.total_bytes,
      chunk_bytes: a.chunk_bytes,
      chunk_count: a.chunk_count,
      chunk_sha256: a.chunk_sha256 as string[],
    },
  };
}

export function checkInitiate(body: unknown): Check<InitiateBody> {
  if (body === null || typeof body !== "object") {
    return { ok: false, code: "bad_json" };
  }
  const b = body as Record<string, unknown>;
  if (
    typeof b.schema_version !== "number" ||
    !(SCHEMA_VERSIONS as readonly number[]).includes(b.schema_version)
  ) {
    return { ok: false, code: "schema_unsupported" };
  }
  if (b.kind !== "quick" && b.kind !== "deep" && b.kind !== "omarchy-mac-e2e") {
    return { ok: false, code: "kind_invalid" };
  }
  if (!isSha256(b.content_sha256)) {
    return { ok: false, code: "bad_content_sha256" };
  }
  if (b.payload === null || typeof b.payload !== "object") {
    return { ok: false, code: "payload_invalid" };
  }
  if (b.archive !== null) {
    const check = checkArchive(b.archive);
    if (!check.ok) return check;
  }
  return {
    ok: true,
    value: {
      schema_version: b.schema_version,
      kind: b.kind,
      content_sha256: b.content_sha256,
      payload: b.payload,
      archive: (b.archive ?? null) as InitiateArchive | null,
    },
  };
}
