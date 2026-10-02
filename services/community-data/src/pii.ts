// Server-side PII rejection scan: defense in depth behind the
// collector's local redaction. Scans the JSON-serialized summary text;
// on any hit the submission is REFUSED, never stored. Patterns mirror
// the collector's Redactor so a string the collector would have
// redacted is also refused server-side if redaction was bypassed.
//
// Matches are reported as kind counts only; matched text is never
// echoed back, so the error response cannot leak the PII itself.

export type PiiKinds = Record<string, number>;

const PII_RE = new RegExp(
  [
    // MAC address (colon-separated).
    "(?<mac>\\b[0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}\\b)",
    // IPv6: link-local / ULA prefixes, then the full 8-group form.
    "(?<ipv6>\\b(?:fe80|fd[0-9a-f]{2}|fc[0-9a-f]{2})(?::[0-9a-fA-F]{0,4}){1,7}(?:%\\w+)?\\b|\\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\\b)",
    // IPv4, optional CIDR suffix.
    "(?<ipv4>\\b\\d{1,3}(?:\\.\\d{1,3}){3}(?:/\\d{1,3})?\\b)",
    // UUID.
    "(?<uuid>\\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\\b)",
    // Serial numbers in JSON/text shapes; skips bare null/true/false.
    "(?<serial>\\bserial(?:[_-]?number)?\\b[\"'\\s:=]{1,4}(?!null\\b|true\\b|false\\b)[^\"',\\s}{]{2,})",
    // Home paths: POSIX and Windows (backslashes arrive JSON-escaped).
    "(?<home_path>/(?:home|Users)/[A-Za-z0-9._-]{1,64}|[A-Za-z]:\\\\+Users\\\\+[A-Za-z0-9._-]{1,64})",
    // Credential shapes: token prefixes, AWS keys, Slack tokens, JWTs,
    // bearer headers, and key=value assignments with quoted secrets.
    "(?<credential>\\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\\b|\\bgithub_pat_[A-Za-z0-9_]{20,}\\b|\\bAKIA[0-9A-Z]{16}\\b|\\bxox[baprs]-[A-Za-z0-9-]{10,}\\b|\\bsk-[A-Za-z0-9_-]{20,}\\b|\\beyJ[A-Za-z0-9_-]{10,}\\.[A-Za-z0-9_-]{10,}\\.[A-Za-z0-9_-]{5,}\\b|\\bBearer\\s+[A-Za-z0-9._~+/=-]{8,}|(?:password|passwd|secret|api_?key|private_?key|auth_?token)\\w*\\\\*[\"']?\\s*[:=]\\s*\\\\*[\"'][^\"']{4,}[\"'])",
    // Hostname shapes: mDNS and common private suffixes.
    "(?<hostname>\\.(?:local|lan|home|internal)\\b)",
  ].join("|"),
  "i",
);
const MAX_REPORTED_HITS = 50;
// Observed collector firmware values only. Shape-based exceptions can
// disguise an address as version components, even with a large build ID.
const IBOOT_VERSIONS = new Set([
  "iboot-10151.140.19.700.2",
  "iboot-20712.1.2.0.0",
]);

// Header carrying the collector redactor's derived short host names
// (X-MLX-Host-Aliases). Request-only: scanned against the summary,
// never stored, logged, or echoed, so the alias list itself stays out
// of the public record.
export const ALIAS_HEADER = "X-MLX-Host-Aliases";
const MAX_ALIASES = 16;
const ALIAS_RE = /^[a-z0-9][a-z0-9._-]{3,63}$/;

export function parseHostAliases(raw: string | null | undefined): string[] {
  if (!raw) return [];
  const seen = new Set<string>();
  for (const piece of raw.split(",")) {
    const alias = piece.trim().toLowerCase();
    if (ALIAS_RE.test(alias)) seen.add(alias);
    if (seen.size >= MAX_ALIASES) break;
  }
  return [...seen];
}

export function scanPii(text: string, hostAliases: string[] = []): PiiKinds | null {
  const aliases = parseHostAliases(
    Array.isArray(hostAliases) ? hostAliases.join(",") : null,
  ).slice(0, MAX_ALIASES);
  let source = PII_RE.source;
  if (aliases.length > 0) {
    // Mirror of the collector's derived-alias rule: an alias may sit
    // inside a longer token (`/etc/systemd/system/<alias>-ane.service`),
    // so both neighbors must be non-alphanumeric; hex runs and longer
    // words never match. Longest first so a shorter prefix cannot eat
    // a longer alias at the same position.
    const alternation = [...aliases]
      .sort((a, b) => b.length - a.length)
      .map((a) => a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|");
    source += `|(?<hostname_alias>(?<![A-Za-z0-9])(?:${alternation})(?![A-Za-z0-9]))`;
  }
  const re = new RegExp(source, PII_RE.flags + "g");
  const kinds: PiiKinds = {};
  let match: RegExpExecArray | null;
  let hits = 0;
  while ((match = re.exec(text)) !== null) {
    const groups = match.groups ?? {};
    const kind = Object.keys(groups).find((k) => groups[k] !== undefined);
    if (kind === undefined) continue;
    if (kind === "ipv4") {
      // iBoot firmware uses long dotted version chains. The IPv4 regex
      // can match a four-component suffix (10151.140.19.700.2), which
      // the collector intentionally preserves. Recognize only the known
      // firmware values above. Unrecognized versions remain subject to
      // the PII scan until evidence establishes another safe exception.
      // Bound both search and parsing so each candidate costs at most
      // 256 characters, including on a payload full of dotted quads.
      const windowStart = Math.max(0, match.index - 128);
      const window = text.slice(windowStart, match.index + 128);
      const start = window.toLowerCase().lastIndexOf("iboot-", match.index - windowStart);
      if (start >= 0) {
        const version = /^iBoot-(\d{1,8})(?:\.\d{1,3}){4}(?![\w./-])/i.exec(
          window.slice(start),
        );
        const previous = text[windowStart + start - 1];
        if (version && IBOOT_VERSIONS.has(version[0].toLowerCase()) &&
            (!previous || !/[\w.-]/.test(previous)) &&
            windowStart + start + version[0].length >= match.index + match[0].length) {
          continue;
        }
      }
    }
    kinds[kind] = (kinds[kind] ?? 0) + 1;
    hits++;
    if (hits >= MAX_REPORTED_HITS) break;
    if (match[0].length === 0) re.lastIndex++;
  }
  return hits > 0 ? kinds : null;
}
