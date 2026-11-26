import { describe, expect, test } from "bun:test";
import { chunkBytes, sha256Hex } from "../../src/hash";
import { leadingZeroBits } from "../../src/pow";
import { MAX_CHUNK_BYTES, expectedChunkCount, expectedChunkLength } from "../../src/caps";

describe("sha256Hex", () => {
  test("matches the known empty and abc vectors", async () => {
    expect(await sha256Hex("")).toBe(
      "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    );
    expect(await sha256Hex("abc")).toBe(
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    );
  });

  test("accepts binary input", async () => {
    const bytes = new Uint8Array([0, 1, 2, 255]);
    expect(await sha256Hex(bytes)).toBe(await sha256Hex(bytes.slice()));
  });
});

describe("chunk splitting and reassembly", () => {
  test("round-trips an 8 MB archive at the wire chunk size", async () => {
    const data = new Uint8Array(8 * 1024 * 1024);
    for (let i = 0; i < data.length; i++) data[i] = i & 0xff;
    const chunks = chunkBytes(data, MAX_CHUNK_BYTES);
    expect(chunks.length).toBe(expectedChunkCount(data.length, MAX_CHUNK_BYTES));
    const reassembled = new Uint8Array(data.length);
    let off = 0;
    for (const chunk of chunks) {
      reassembled.set(chunk, off);
      off += chunk.length;
    }
    expect(await sha256Hex(reassembled)).toBe(await sha256Hex(data));
  });

  test("per-chunk lengths match the expected accounting", () => {
    const data = new Uint8Array(10);
    const chunks = chunkBytes(data, 4);
    expect(chunks.map((c) => c.length)).toEqual([4, 4, 2]);
    chunks.forEach((c, i) => {
      expect(c.length).toBe(expectedChunkLength(10, 4, i));
    });
  });
});

describe("leading zero bits", () => {
  test("counts nibble-accurate zero prefixes", () => {
    expect(leadingZeroBits("00")).toBe(8);
    expect(leadingZeroBits("0f")).toBe(4);
    expect(leadingZeroBits("1f")).toBe(3);
    expect(leadingZeroBits("7f")).toBe(1);
    expect(leadingZeroBits("ff")).toBe(0);
  });
});
