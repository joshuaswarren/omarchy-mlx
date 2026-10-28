// Shared archive fixtures for the sanitizer unit test and the
// end-to-end smoke: a minimal ustar builder, gzip helpers, and a
// probe archive whose members carry a synthetic identity line and a
// deterministic binary control blob.

export function buildTar(entries: { name: string; body: Uint8Array }[]): Uint8Array {
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

export async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Response(bytes).body!.pipeThrough(
    new CompressionStream("gzip"),
  );
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new Response(bytes).body!.pipeThrough(
    new DecompressionStream("gzip"),
  );
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export function findMember(tar: Uint8Array, name: string): Uint8Array | null {
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

// Deterministic pseudo-random control bytes (xorshift32, seed
// recorded by the caller). Incompressible, so the fixture's chunk
// count stays stable across gzip runs.
export function controlBytes(seed: number, length = 4096): Uint8Array {
  let s = seed >>> 0 || 1;
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    out[i] = s & 0xff;
  }
  return out;
}

export const PROBE_DTS = (
  "/dts-v1/;\n" +
  "/ {\n" +
  "    compatible = \"test,fixture\";\n" +
  "    serial-number = \"TEST-SERIAL-0001\";\n" +
  "};\n"
);

export async function buildProbeArchive(seed: number): Promise<Uint8Array> {
  return gzip(buildTar([
    { name: "probe.dts", body: new TextEncoder().encode(PROBE_DTS) },
    { name: "control.bin", body: controlBytes(seed) },
  ]));
}
