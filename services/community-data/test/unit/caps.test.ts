import { describe, expect, test } from "bun:test";
import {
  MAX_CHUNK_BYTES,
  MAX_CHUNKS,
  expectedChunkCount,
  expectedChunkLength,
} from "../../src/caps";
import { checkArchive, checkInitiate } from "../../src/validate";

function archiveShape(total: number, chunkBytes: number) {
  const count = expectedChunkCount(total, chunkBytes);
  return {
    total_bytes: total,
    chunk_bytes: chunkBytes,
    chunk_count: count,
    chunk_sha256: Array.from({ length: count }, (_, i) => String(i).padStart(64, "0")),
  };
}

function validBody(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: 1,
    kind: "deep",
    content_sha256: "a".repeat(64),
    payload: { schema_version: 1, kind: "deep", generated_at: "2026-09-03T00:00:00Z" },
    archive: null,
    pow: { nonce: "1", difficulty: 18 },
    ...overrides,
  };
}

describe("chunk math", () => {
  test("8 MB in 768 KB chunks is 11 chunks, matching the contract cap", () => {
    expect(expectedChunkCount(8 * 1024 * 1024, MAX_CHUNK_BYTES)).toBe(MAX_CHUNKS);
    expect(expectedChunkLength(8 * 1024 * 1024, MAX_CHUNK_BYTES, 10)).toBe(8 * 1024 * 1024 - 10 * MAX_CHUNK_BYTES);
  });

  test("exact multiples do not create an empty trailing chunk", () => {
    expect(expectedChunkCount(2 * MAX_CHUNK_BYTES, MAX_CHUNK_BYTES)).toBe(2);
    expect(expectedChunkLength(2 * MAX_CHUNK_BYTES, MAX_CHUNK_BYTES, 1)).toBe(MAX_CHUNK_BYTES);
  });

  test("past the boundary every chunk is full length", () => {
    expect(expectedChunkLength(10, 4, 2)).toBe(2);
    expect(expectedChunkLength(10, 4, 3)).toBe(0);
  });
});

describe("cap enforcement rejects by name", () => {
  test("archive over 8 MB is archive_too_large", () => {
    const tooBig = archiveShape(8 * 1024 * 1024 + 1, MAX_CHUNK_BYTES);
    expect(checkArchive(tooBig)).toEqual({ ok: false, code: "archive_too_large" });
  });

  test("single chunk over 768 KB is chunk_too_large", () => {
    const shape = archiveShape(2 * MAX_CHUNK_BYTES, MAX_CHUNK_BYTES + 1);
    expect(checkArchive(shape)).toEqual({ ok: false, code: "chunk_too_large" });
  });

  test("more than 11 chunks is too_many_chunks", () => {
    // 8 MB in 512 KB chunks = 16 consistent chunks; the chunk-count cap
    // fires before any single cap other than total size is exceeded.
    const shape = archiveShape(8 * 1024 * 1024, 512 * 1024);
    expect(checkArchive(shape)).toEqual({ ok: false, code: "too_many_chunks" });
  });

  test("count not matching ceil(total/chunk) is chunk_count_invalid", () => {
    const shape = { ...archiveShape(1000, 100), chunk_count: 99 };
    expect(checkArchive(shape)).toEqual({ ok: false, code: "chunk_count_invalid" });
  });

  test("non-hex chunk hash list is archive_invalid", () => {
    const shape = { ...archiveShape(100, 100), chunk_sha256: ["zz"] };
    expect(checkArchive(shape)).toEqual({ ok: false, code: "archive_invalid" });
  });
});

describe("initiate body checks", () => {
  test("valid body passes with null archive", () => {
    const result = checkInitiate(validBody());
    expect(result.ok).toBe(true);
  });

  test("wrong schema_version is schema_unsupported", () => {
    expect(checkInitiate(validBody({ schema_version: 3 }))).toEqual({
      ok: false,
      code: "schema_unsupported",
    });
  });

  test("v2 bodies pass the gate like v1", () => {
    expect(checkInitiate(validBody({ schema_version: 2 })).ok).toBe(true);
  });

  test("unknown kind is kind_invalid", () => {
    expect(checkInitiate(validBody({ kind: "telemetry" }))).toEqual({
      ok: false,
      code: "kind_invalid",
    });
  });

  test("non-hex content hash is bad_content_sha256", () => {
    expect(checkInitiate(validBody({ content_sha256: "XY".repeat(32) }))).toEqual({
      ok: false,
      code: "bad_content_sha256",
    });
  });

  test("missing payload is payload_invalid", () => {
    expect(checkInitiate(validBody({ payload: "string" }))).toEqual({
      ok: false,
      code: "payload_invalid",
    });
  });

  test("oversize archive is rejected through the body path too", () => {
    const result = checkInitiate(
      validBody({ archive: archiveShape(9 * 1024 * 1024, MAX_CHUNK_BYTES) }),
    );
    expect(result).toEqual({ ok: false, code: "archive_too_large" });
  });
});
