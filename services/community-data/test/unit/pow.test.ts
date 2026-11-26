import { describe, expect, test } from "bun:test";
import { MIN_POW_BITS, expectedChunkCount } from "../../src/caps";
import { leadingZeroBits, verifyPow } from "../../src/pow";
import { MAX_CHUNK_BYTES } from "../../src/caps";

const SHA = "b".repeat(64);

// Test-side solver (sync, Bun hasher). The server only verifies.
function solvePow(contentSha: string, difficulty: number): string {
  for (let nonce = 0; ; nonce++) {
    const hasher = new Bun.CryptoHasher("sha256");
    hasher.update(`${contentSha}:${nonce}`);
    if (leadingZeroBits(hasher.digest("hex")) >= difficulty) return String(nonce);
  }
}

describe("verifyPow", () => {
  test("accepts a nonce solving the minimum difficulty", async () => {
    const nonce = solvePow(SHA, MIN_POW_BITS);
    const result = await verifyPow(SHA, { nonce, difficulty: MIN_POW_BITS }, MIN_POW_BITS);
    expect(result).toEqual({ ok: true, difficulty: MIN_POW_BITS });
  });

  test("rejects a weak nonce", async () => {
    const result = await verifyPow(SHA, { nonce: "0", difficulty: MIN_POW_BITS }, MIN_POW_BITS);
    expect(result).toEqual({ ok: false, code: "pow_invalid" });
  });

  test("rejects an absent token", async () => {
    expect(await verifyPow(SHA, undefined, MIN_POW_BITS)).toEqual({
      ok: false,
      code: "pow_missing",
    });
    expect(await verifyPow(SHA, null, MIN_POW_BITS)).toEqual({
      ok: false,
      code: "pow_missing",
    });
  });

  test("rejects difficulty below the server minimum", async () => {
    const nonce = solvePow(SHA, MIN_POW_BITS - 2);
    const result = await verifyPow(
      SHA,
      { nonce, difficulty: MIN_POW_BITS - 2 },
      MIN_POW_BITS,
    );
    expect(result).toEqual({ ok: false, code: "pow_invalid" });
  });

  test("rejects a malformed nonce", async () => {
    const result = await verifyPow(SHA, { nonce: "12ab", difficulty: MIN_POW_BITS }, MIN_POW_BITS);
    expect(result).toEqual({ ok: false, code: "pow_invalid" });
  });

  test("binds the token to the content hash", async () => {
    const nonce = solvePow(SHA, MIN_POW_BITS);
    const other = "c".repeat(64);
    const result = await verifyPow(other, { nonce, difficulty: MIN_POW_BITS }, MIN_POW_BITS);
    expect(result).toEqual({ ok: false, code: "pow_invalid" });
  });
});

describe("difficulty scale sanity", () => {
  test("18 bits needs on the order of 2^18 hashes", () => {
    // 8 MB / 768 KB = 11 chunks; the workload stays independent of size.
    expect(expectedChunkCount(8 * 1024 * 1024, MAX_CHUNK_BYTES)).toBe(11);
    expect(MIN_POW_BITS).toBe(18);
  });
});
