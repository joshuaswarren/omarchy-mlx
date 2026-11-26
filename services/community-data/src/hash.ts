export async function sha256Hex(
  data: ArrayBuffer | Uint8Array | string,
): Promise<string> {
  const bytes =
    typeof data === "string" ? new TextEncoder().encode(data) : data;
  const digest = await crypto.subtle.digest(
    "SHA-256",
    bytes as ArrayBuffer,
  );
  return [...new Uint8Array(digest)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/** Split into chunks of exactly chunkBytes (last may be shorter). */
export function chunkBytes(
  data: Uint8Array,
  chunkBytesSize: number,
): Uint8Array[] {
  const out: Uint8Array[] = [];
  for (let off = 0; off < data.length; off += chunkBytesSize) {
    out.push(data.subarray(off, Math.min(off + chunkBytesSize, data.length)));
  }
  return out;
}
