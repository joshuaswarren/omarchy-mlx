import { sha256Hex } from "./hash";

export type PowCheck =
  | { ok: true; difficulty: number }
  | { ok: false; code: "pow_missing" | "pow_invalid" };

/**
 * Proof of work binds a nonce to the archive content hash:
 * sha256("<content_sha256>:<nonce>") must have at least `difficulty`
 * leading zero bits. A token is therefore useless for any other
 * submission, and replays of the same token dedupe into the same
 * content-addressed record.
 */
export async function verifyPow(
  contentSha256: string,
  pow: unknown,
  minBits: number,
): Promise<PowCheck> {
  if (pow === null || typeof pow !== "object") {
    return { ok: false, code: "pow_missing" };
  }
  const { nonce, difficulty } = pow as Record<string, unknown>;
  if (typeof nonce !== "string" || !/^[0-9]{1,32}$/.test(nonce)) {
    return { ok: false, code: "pow_invalid" };
  }
  if (
    typeof difficulty !== "number" ||
    !Number.isInteger(difficulty) ||
    difficulty < minBits ||
    difficulty > 64
  ) {
    return { ok: false, code: "pow_invalid" };
  }
  const digest = await sha256Hex(`${contentSha256}:${nonce}`);
  return leadingZeroBits(digest) >= difficulty
    ? { ok: true, difficulty }
    : { ok: false, code: "pow_invalid" };
}

export function leadingZeroBits(hex: string): number {
  let bits = 0;
  for (const ch of hex) {
    const nibble = parseInt(ch, 16);
    if (nibble === 0) {
      bits += 4;
      continue;
    }
    if (nibble < 2) bits += 3;
    else if (nibble < 4) bits += 2;
    else if (nibble < 8) bits += 1;
    break;
  }
  return bits;
}
