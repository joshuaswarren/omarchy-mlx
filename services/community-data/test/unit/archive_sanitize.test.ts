import { describe, expect, test } from "bun:test";
import { sanitizeArchiveBlob } from "../../src/archive_sanitize";

// Synthetic dtc text with serial-family property lines. Mirrors
// the contributor-reported shape (M1 t8103 + M2 Pro t6020 with dtc
// installed). The text is what a pre-hotfix collector would have
// shipped; the sanitizer must drop identity property lines.
const LEAK_DTC = (
  "/dts-v1/;\n" +
  "/ {\n" +
  "    compatible = \"apple,t8103\";\n" +
  "    model = \"Apple MacBook Pro (13-inch, M1, 2020)\";\n" +
  "    serial-number = <0x12345678 0x9abcdef0>;\n" +
  "    mlb-serial-number = \"F5KXY123456\";\n" +
  "    board-serial = \"C02XY9876543\";\n" +
  "    serial-index = <0x00000002>;\n" +
  "};\n"
);

// Build a minimal tar (ustar) entry: 512-byte header + body padded
// to 512, terminated by two 512-byte zero blocks.
function buildTar(entries: { name: string; body: Uint8Array }[]): Uint8Array {
  const blocks: Uint8Array[] = [];
  for (const e of entries) {
    const header = new Uint8Array(512);
    const nameBytes = new TextEncoder().encode(e.name);
    header.set(nameBytes.subarray(0, Math.min(100, nameBytes.length)), 0);
    header.set(new TextEncoder().encode("0000644 "), 100);
    const zero = new TextEncoder().encode("0000000 ");
    header.set(zero, 108);
    header.set(zero, 116);
    const sizeOct = e.body.length.toString(8).padStart(11, "0") + "\0";
    header.set(new TextEncoder().encode(sizeOct), 124);
    header.set(new TextEncoder().encode("00000000000\0"), 136);
    header[156] = 0x30;
    header.set(new TextEncoder().encode("ustar"), 257);
    header[262] = 0x00;
    header[263] = 0x30;
    header[264] = 0x30;
    let checksum = 0;
    for (let i = 0; i < 512; i++) {
      checksum += i >= 148 && i < 156 ? 0x20 : header[i];
    }
    const ck = checksum.toString(8).padStart(6, "0") + "\0 ";
    header.set(new TextEncoder().encode(ck), 148);
    blocks.push(header);
    const padded = Math.ceil(e.body.length / 512) * 512;
    const paddedBody = new Uint8Array(padded);
    paddedBody.set(e.body, 0);
    blocks.push(paddedBody);
  }
  blocks.push(new Uint8Array(512));
  blocks.push(new Uint8Array(512));
  const total = blocks.reduce((n, b) => n + b.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const b of blocks) { out.set(b, off); off += b.length; }
  return out;
}

async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Response(bytes).body!.pipeThrough(
    new CompressionStream("gzip"),
  );
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Response(bytes).body!.pipeThrough(
    new DecompressionStream("gzip"),
  );
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function findMember(tar: Uint8Array, name: string): Uint8Array | null {
  let cursor = 0;
  while (cursor + 512 <= tar.length) {
    if (tar[cursor] === 0) return null;
    const header = tar.subarray(cursor, cursor + 512);
    let nameEnd = 0;
    while (nameEnd < 100 && header[nameEnd] !== 0) nameEnd++;
    const memberName = new TextDecoder("utf-8", { fatal: false })
      .decode(header.subarray(0, nameEnd));
    let sizeStr = "";
    for (let i = 124; i < 136 && header[i] !== 0 && header[i] !== 0x20; i++) {
      sizeStr += String.fromCharCode(header[i]);
    }
    const size = Number.parseInt(sizeStr.trim(), 8) || 0;
    const dataStart = cursor + 512;
    if (memberName === name) {
      return tar.subarray(dataStart, dataStart + size);
    }
    const padded = Math.ceil(size / 512) * 512;
    cursor = dataStart + padded;
  }
  return null;
}

describe("archive blob sanitization (owner directive 2026-10-03)", () => {
  test("serial-family property lines are removed from dtc text", async () => {
    const tar = buildTar([
      { name: "ane-linux-dt.txt", body: new TextEncoder().encode(LEAK_DTC) },
    ]);
    const gz = await gzip(tar);
    const out = await sanitizeArchiveBlob(gz);
    expect(out.ok).toBe(true);
    if (!out.ok) return;
    const newTar = await gunzip(out.bytes);
    const member = findMember(newTar, "ane-linux-dt.txt");
    expect(member).not.toBeNull();
    const text = new TextDecoder().decode(member!);
    expect(text).not.toContain("serial-number");
    expect(text).not.toContain("mlb-serial-number");
    expect(text).not.toContain("board-serial");
    expect(text).not.toContain("serial-index");
    expect(text).not.toContain("0x9abcdef0");
    expect(text).not.toContain("C02XY9876543");
    // Benign structural lines survive.
    expect(text).toContain("compatible = \"apple,t8103\"");
    expect(text).toContain("model = \"Apple MacBook Pro");
  });

  test("non-text members pass through unchanged", async () => {
    // ane-adt-dump.bin is a binary blob; the sanitizer must not
    // mangle it.
    const bin = new Uint8Array(64);
    for (let i = 0; i < bin.length; i++) bin[i] = i & 0xff;
    const tar = buildTar([
      { name: "ane-adt-dump.bin", body: bin },
      { name: "ane-linux-dt.txt", body: new TextEncoder().encode(LEAK_DTC) },
    ]);
    const gz = await gzip(tar);
    const out = await sanitizeArchiveBlob(gz);
    expect(out.ok).toBe(true);
    if (!out.ok) return;
    const newTar = await gunzip(out.bytes);
    const member = findMember(newTar, "ane-adt-dump.bin");
    expect(member).not.toBeNull();
    expect(Array.from(member!)).toEqual(Array.from(bin));
  });

  test("oversize archive is refused with too_large", async () => {
    const bytes = new Uint8Array(9 * 1024 * 1024);
    const out = await sanitizeArchiveBlob(bytes);
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.code).toBe("too_large");
  });

  test("non-gzip bytes produce bad_tar", async () => {
    const bytes = new TextEncoder().encode("not a gzip stream");
    const out = await sanitizeArchiveBlob(bytes);
    expect(out.ok).toBe(false);
    if (out.ok) return;
    expect(out.code).toBe("bad_tar");
  });
});
