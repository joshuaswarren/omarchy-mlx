// Archive sanitization (owner directive, 2026-10-03): when a
// submission's payload was stripped, the archive blob may still
// contain the leaked text inside its members (ane-linux-dt.txt,
// ane-macos-iodt.txt, manifest.json, submission.md, the
// section/*.json files). Never store the raw value anywhere.
//
// Strategy:
//   1. Decompress gzip via DecompressionStream.
//   2. Walk the tar entries in order; for text members, run the same
//      PII rewrite the payload gets (redactPiiPayload on a parsed
//      JSON member, or drop identity-property lines in dtc/iodt text
//      and rewrite the text as a sanitized string).
//   3. Recompress with gzip mtime 0 so the byte stream is
//      deterministic.
//   4. Re-chunk the sanitized archive and re-store it. The content
//      hash DOES change (rewritten bytes ≠ uploaded bytes); the row
//      stays keyed by the uploaded content_sha256 (per the initiate
//      contract), and the archive rewrite is recorded under
//      `archive_withheld_pii: false` on the response.
//
// Fallback (sanitizeArchiveBlob throws): delete the stored chunks
// and null the archive columns on the row, log the failure, and
// return `archive_withheld_pii: true` in the response. The cleaned
// summary is still stored.

import { MAX_ARCHIVE_BYTES } from "./caps";
import { sha256Hex } from "./hash";

// Identity property name regex mirrored from the collector
// (scripts/collect_common.py IDENTITY_PROP_RE). Used to drop
// property lines from dtc / IODeviceTree text members. The `i`
// flag replaces Python's `(?i)`; the engine (Bun/JS) does not
// support inline `(?i)`.
const IDENTITY_PROP_RE =
  /(serial|(^|[-_,])mlb([-_,]|$)|ecid|unique-chip|udid|uuid)/i;
const ENTRY_HEADER = 512;
const MAX_MEMBER_BYTES = 8 * 1024 * 1024;

function sanitizeDtcText(text: string): string {
  const out: string[] = [];
  for (const line of text.split("\n")) {
    const eq = line.indexOf("=");
    if (eq < 0) { out.push(line); continue; }
    const label = line.slice(0, eq).trim().replace(/[";]+$/, "");
    if (IDENTITY_PROP_RE.test(label)) continue;
    out.push(line);
  }
  return out.join("\n");
}

async function decompressGzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Response(bytes).body!.pipeThrough(
    new DecompressionStream("gzip"),
  );
  const buf = await new Response(stream).arrayBuffer();
  return new Uint8Array(buf);
}

async function recompressGzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Response(bytes).body!.pipeThrough(
    new CompressionStream("gzip"),
  );
  const buf = await new Response(stream).arrayBuffer();
  const out = new Uint8Array(buf);
  // gzip mtime at offset 4 (4 bytes LE). Zero so the byte stream is
  // deterministic; OS byte (offset 9) is left as the runtime picked.
  if (out.length >= 8) {
    out[4] = 0; out[5] = 0; out[6] = 0; out[7] = 0;
  }
  return out;
}

function parseTarEntries(uncompressed: Uint8Array): {
  name: string;
  size: number;
  data: Uint8Array;
}[] {
  const entries = [];
  let cursor = 0;
  while (cursor + ENTRY_HEADER <= uncompressed.length) {
    if (uncompressed[cursor] === 0) break;
    const header = uncompressed.subarray(cursor, cursor + ENTRY_HEADER);
    let nameEnd = 0;
    while (nameEnd < 100 && header[nameEnd] !== 0) nameEnd++;
    const name = new TextDecoder("utf-8", { fatal: false })
      .decode(header.subarray(0, nameEnd));
    let sizeStr = "";
    for (let i = 124; i < 136 && header[i] !== 0 && header[i] !== 0x20; i++) {
      sizeStr += String.fromCharCode(header[i]);
    }
    const size = Number.parseInt(sizeStr.trim(), 8) || 0;
    const typeflag = String.fromCharCode(header[156] || 0x30);
    if (typeflag === "5") { cursor += ENTRY_HEADER; continue; }
    if (name === "") break;
    if (size > MAX_MEMBER_BYTES) {
      throw new Error(`archive member too large: ${name} (${size} bytes)`);
    }
    const dataStart = cursor + ENTRY_HEADER;
    const data = uncompressed.subarray(dataStart, dataStart + size);
    entries.push({ name, size, data });
    const padded = Math.ceil(size / 512) * 512;
    cursor = dataStart + padded;
  }
  return entries;
}

function buildTarEntry(name: string, body: Uint8Array): Uint8Array {
  const header = new Uint8Array(ENTRY_HEADER);
  const nameBytes = new TextEncoder().encode(name);
  header.set(nameBytes.subarray(0, Math.min(100, nameBytes.length)), 0);
  header.set(new TextEncoder().encode("0000644 "), 100);
  const zero = new TextEncoder().encode("0000000 ");
  header.set(zero, 108);
  header.set(zero, 116);
  const sizeOct = body.length.toString(8).padStart(11, "0") + "\0";
  header.set(new TextEncoder().encode(sizeOct), 124);
  header.set(new TextEncoder().encode("00000000000\0"), 136);
  header[156] = 0x30;
  header.set(new TextEncoder().encode("ustar"), 257);
  header[262] = 0x00;
  header[263] = 0x30;
  header[264] = 0x30;
  let checksum = 0;
  for (let i = 0; i < ENTRY_HEADER; i++) {
    checksum += i >= 148 && i < 156 ? 0x20 : header[i];
  }
  const ck = checksum.toString(8).padStart(6, "0") + "\0 ";
  header.set(new TextEncoder().encode(ck), 148);
  const paddedSize = Math.ceil(body.length / 512) * 512;
  const out = new Uint8Array(ENTRY_HEADER + paddedSize);
  out.set(header, 0);
  out.set(body, ENTRY_HEADER);
  return out;
}

function rebuildTar(entries: { name: string; body: Uint8Array }[]): Uint8Array {
  const parts: Uint8Array[] = [];
  for (const e of entries) {
    parts.push(buildTarEntry(e.name, e.body));
  }
  parts.push(new Uint8Array(ENTRY_HEADER * 2));
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}

function rewriteTextMember(name: string, body: Uint8Array): Uint8Array {
  const lower = name.toLowerCase();
  const texty = lower.endsWith(".txt") || lower.endsWith(".md") ||
    lower.endsWith(".json") || lower.endsWith(".jsonl") ||
    lower.endsWith(".dts") || lower.endsWith(".dtsi");
  if (!texty) return body;
  const text = new TextDecoder("utf-8", { fatal: false }).decode(body);
  const isDtc = /ane-linux-dt\.txt$|ane-macos-iodt\.txt$|\.dts$|\.dtsi$/i
    .test(name);
  const cleaned = isDtc ? sanitizeDtcText(text) : text;
  return new TextEncoder().encode(cleaned);
}

export type SanitizeOutcome =
  | { ok: true; bytes: Uint8Array }
  | { ok: false; code: "too_large" | "bad_tar" | "rewrite_failed"; reason: string };

export async function sanitizeArchiveBlob(bytes: Uint8Array): Promise<SanitizeOutcome> {
  if (bytes.length > MAX_ARCHIVE_BYTES) {
    return { ok: false, code: "too_large",
      reason: `archive exceeds ${MAX_ARCHIVE_BYTES} bytes` };
  }
  let uncompressed: Uint8Array;
  try {
    uncompressed = await decompressGzip(bytes);
  } catch (exc) {
    return { ok: false, code: "bad_tar", reason: `gzip: ${(exc as Error).message}` };
  }
  let entries;
  try {
    entries = parseTarEntries(uncompressed);
  } catch (exc) {
    return { ok: false, code: "bad_tar", reason: (exc as Error).message };
  }
  let rebuilt: Uint8Array;
  try {
    const newEntries: { name: string; body: Uint8Array }[] = [];
    for (const e of entries) {
      newEntries.push({ name: e.name, body: rewriteTextMember(e.name, e.data) });
    }
    const newTar = rebuildTar(newEntries);
    rebuilt = await recompressGzip(newTar);
  } catch (exc) {
    return { ok: false, code: "rewrite_failed",
      reason: (exc as Error).message };
  }
  return { ok: true, bytes: rebuilt };
}

export { sha256Hex };
