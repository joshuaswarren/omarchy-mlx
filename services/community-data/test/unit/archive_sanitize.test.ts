import { describe, expect, test } from "bun:test";
import { sanitizeArchiveBlob } from "../../src/archive_sanitize";
import { buildTar, findMember, gunzip, gzip } from "../fixtures/archive";

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
