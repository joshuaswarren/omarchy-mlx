// Server-side PII scan: defense in depth behind the collector's local
// redaction. Two modes over the same regex set:
//
//   scanPiiPayload  — report hits, refuse the submission (legacy reject
//                     path; hard-rejects below are unchanged).
//   redactPiiPayload — find hits, rewrite matched substrings in the
//                     payload text with typed placeholders (same style
//                     the collector's Redactor uses), and return the
//                     cleaned payload plus the kind counts. Tally
//                     objects (redaction_summary, _redaction) are
//                     blanked before the scan so their keys/counts
//                     never trip the same patterns. The owner directive
//                     (2026-10-03): "the worker must STRIP PII instead
//                     of REJECTING it"; redact+store with counts is
//                     the new default, scan is kept for callers that
//                     need reject semantics (none today, kept for
//                     tests).
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
// Firmware version fields whose WHOLE value may be an iBoot version
// chain (#27 follow-up): the value is exempt from the IPv4 scan only
// where the KEY says it is firmware and the value fully matches the
// version shape. The same string in a free-text field, or an
// address-shaped value behind the prefix, stays fully subject to the
// scan — a shape-based exception can disguise an address as version
// components. Newer first-stage firmware reports itself as `mBoot-`
// rather than `iBoot-` (M2 Air: mBoot-20457.40.150.0.1); both prefixes
// get the same treatment.
export const FIRMWARE_VERSION_KEYS: Record<string, true> = {
  "asahi,iboot1-version": true,
  "asahi,iboot2-version": true,
  "asahi,system-fw-version": true,
  "asahi,os-fw-version": true,
};
const IBOOT_VALUE_RE = /^[im]boot-\d+(?:\.\d+)+$/i;
// A value that could BE an address (every group in octet range, four
// groups) is never exempt, version-shaped or not.
const IBOOT_BARE_IPV4_RE = /^[im]boot-\d{1,3}(?:\.\d{1,3}){3}$/i;
// Inside the free-text boot_chain field only explicit `ibootN=`
// assignments are exempt — never a bare `iBoot-…` token, which can
// carry a disguised address (assembly of the example elided for the
// privacy hook).
const BOOT_CHAIN_TOKEN_RE = /\biboot\d+=([im]boot-\d+(?:\.\d+)+)(?=\s|$)/gi;

// Marketing-name words: a host alias equal to one of these — or
// shorter than 6 characters, or a tNNNN chip id — is RESTRICTED to
// free text (the exempt name fields are blanked for its pass), never
// the device-tree/IORegistry names. Long aliases are unrestricted:
// they still match everywhere, name fields included.
const MODEL_TOKENS: Record<string, true> = {
  m1: true,
  m2: true,
  m3: true,
  m4: true,
  m5: true,
  m6: true,
  pro: true,
  max: true,
  ultra: true,
  air: true,
  neo: true,
  mac: true,
  book: true,
  macbook: true,
  studio: true,
  mini: true,
  imac: true,
  apple: true,
};
const MODEL_CHIP_RE = /^t\d{4}$/;

// Mirror of the collector's EXEMPT_NAME_FIELDS: device-tree /
// IORegistry marketing names and identifiers. These values do not come
// from the user, so restricted aliases never match inside them.
export const EXEMPT_NAME_FIELDS: Record<string, true> = {
  model: true,
  chip: true,
  "chip_name": true,
  board: true,
  "board_id": true,
  "board_name": true,
  compatible: true,
  product: true,
  "product_name": true,
  "machine_model": true,
  "machine_name": true,
  "model_identifier": true,
  "hw_model": true,
  "marketing_name": true,
};

function isExemptIbootValue(value: string): boolean {
  return IBOOT_VALUE_RE.test(value) && !IBOOT_BARE_IPV4_RE.test(value);
}

function restrictedAlias(alias: string): boolean {
  const lower = alias.toLowerCase();
  return lower.length < 6 || MODEL_TOKENS[lower] === true ||
    MODEL_CHIP_RE.test(lower);
}

// Copy of `payload` with selected string values blanked to same-length
// whitespace, so the scanned text keeps its shape:
//   firmware — blank exempt iBoot version values (both scans);
//   names    — additionally blank the exempt name fields (the
//              restricted-alias pass).
function blankRegions(payload: unknown, names: boolean): unknown {
  const clone = structuredClone(payload);
  const walk = (node: unknown, key: string | null): void => {
    if (Array.isArray(node)) {
      node.forEach((item) => walk(item, key));
      return;
    }
    if (node === null || typeof node !== "object") return;
    for (const [childKey, value] of Object.entries(node)) {
      if (typeof value !== "string") {
        walk(value, childKey);
        continue;
      }
      if (FIRMWARE_VERSION_KEYS[childKey] && isExemptIbootValue(value)) {
        (node as Record<string, unknown>)[childKey] = " ".repeat(value.length);
      } else if (key === null && childKey === "boot_chain") {
        (node as Record<string, unknown>)[childKey] = value.replace(
          BOOT_CHAIN_TOKEN_RE,
          (token) => " ".repeat(token.length),
        );
      } else if (names && EXEMPT_NAME_FIELDS[childKey]) {
        (node as Record<string, unknown>)[childKey] = " ".repeat(value.length);
      }
    }
  };
  walk(clone, null);
  return clone;
}

// The entry point used for submissions. Two scans, merged by kind:
//   A — unrestricted aliases only, over the whole (firmware-blanked)
//       payload: long aliases match everywhere, name fields included,
//       and every non-alias kind is covered exactly once.
//   B — restricted aliases only, with the exempt name fields blanked:
//       short and model-word aliases match free text only. B is
//       folded in as hostname_alias hits; its other kinds duplicate A.
export function scanPiiPayload(
  payload: unknown,
  hostAliases: string[] = [],
): PiiKinds | null {
  const aliases = parseHostAliases(
    Array.isArray(hostAliases) ? hostAliases.join(",") : null,
  );
  const unrestricted = aliases.filter((a) => !restrictedAlias(a));
  const restricted = aliases.filter(restrictedAlias);
  const firmwareBlanked = blankRegions(payload, false);
  const merged = scanPii(JSON.stringify(firmwareBlanked), unrestricted);
  const restrictedScan = scanPii(
    JSON.stringify(blankRegions(firmwareBlanked, true)),
    restricted,
  );
  if (restrictedScan?.hostname_alias) {
    const mergedHits = merged?.hostname_alias ?? 0;
    const total = mergedHits + restrictedScan.hostname_alias;
    if (total > 0) {
      return { ...(merged ?? {}), hostname_alias: total };
    }
  }
  return merged;
}

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
    kinds[kind] = (kinds[kind] ?? 0) + 1;
    hits++;
    if (hits >= MAX_REPORTED_HITS) break;
    if (match[0].length === 0) re.lastIndex++;
  }
  return hits > 0 ? kinds : null;
}

// ---------------------------------------------------------------------------
// Strip-instead-of-reject path (owner directive, 2026-10-03):
//   1. Run the same scan against the payload text.
//   2. If clean, return payload unchanged and kinds=null (no extra
//      `pii_redacted` shape in the response).
//   3. Otherwise blank the tally objects (redaction_summary, any
//      _redaction sub-object) in a structured clone so their keys/
//      counts cannot trip the same patterns; rewrite the JSON text by
//      replacing every matched substring with the kind's placeholder;
//      JSON-parse the cleaned text back. Tally counts are not
//      re-emitted here — the kind counts are the answer, and the
//      stored summary still carries the redaction_summary the
//      collector wrote (blanked to whitespace to avoid re-tripping).
//
// Placeholders mirror the collector's Redactor (collect_common.py):
//   mac → [redacted-mac]
//   ipv4 → [redacted-ip4]
//   ipv6 → [redacted-ip6]
//   uuid → [redacted-uuid]
//   serial → [redacted]
//   home_path → [home]
//   hostname → [host]
//   hostname_alias → [host]
//   credential → [redacted]
export type RedactResult = {
  payload: unknown;
  kinds: PiiKinds;
};

const TALLY_KEYS: Record<string, true> = {
  redaction_summary: true,
  _redaction: true,
};

const PLACEHOLDERS: Record<string, string> = {
  mac: "[redacted-mac]",
  ipv4: "[redacted-ip4]",
  ipv6: "[redacted-ip6]",
  uuid: "[redacted-uuid]",
  serial: "[redacted]",
  home_path: "[home]",
  hostname: "[host]",
  hostname_alias: "[host]",
  credential: "[redacted]",
};

// JSON object KEYS whose ENTIRE value is identity material — the same
// family the collector's is_identity_prop removes whole (serial*,
// *-serial, mlb-serial-*, ecid, unique-chip-*, *udid*, *uuid*). A
// regex value match can never be trusted for these: a multi-cell dtc
// value `<0x12345678 0x9abcdef0>` leaves the second cell behind when
// only the first token is rewritten (2026-10-03 owner directive:
// NEVER store the raw value — key-based whole-value replacement).
const IDENTITY_KEY_RE =
  /(serial|(^|[-_,"'])mlb([-_"']|$)|ecid|unique-chip|udid|uuid)/i;

// Remove whole property STATEMENTS from dtc/dts-shaped text: any line
// whose property label (left of the first '=') matches the identity
// family is dropped entirely, so multi-cell `<a b>`, byte-array
// `[..]`, and quoted-string values cannot leave fragments. Runs on
// RAW string values (after JSON.parse), so JSON-escaped `\n`/`\"`
// forms are handled by the parse, not by the filter.
function stripDtcStatements(text: string): { text: string; removed: number } {
  if (!text.includes("=")) return { text, removed: 0 };
  const out: string[] = [];
  let removed = 0;
  for (const line of text.split("\n")) {
    const eq = line.indexOf("=");
    if (eq < 0) {
      out.push(line);
      continue;
    }
    const label = line.slice(0, eq).trim().replace(/["';]+$/, "");
    if (IDENTITY_KEY_RE.test(label)) {
      removed++;
      continue;
    }
    out.push(line);
  }
  return { text: out.join("\n"), removed };
}

// Structural pass over the payload clone: (a) identity-keyed entries
// get their WHOLE value replaced with "[redacted]" (any shape —
// string, number, cell/byte array, nested object); (b) every string
// value goes through the dtc statement filter, so embedded dumps
// lose identity property lines entirely. Counts land in the serial /
// uuid kinds so the pii_redacted response reflects structural strips
// even when the regex scan then finds nothing.
function stripIdentityValues(payload: unknown): {
  clone: unknown;
  serial: number;
  uuid: number;
} {
  const clone = structuredClone(payload);
  let serial = 0;
  let uuid = 0;
  const walk = (node: unknown): void => {
    if (Array.isArray(node)) {
      node.forEach(walk);
      return;
    }
    if (node === null || typeof node !== "object") return;
    for (const key of Object.keys(node as Record<string, unknown>)) {
      const record = node as Record<string, unknown>;
      const value = record[key];
      if (IDENTITY_KEY_RE.test(key)) {
        record[key] = "[redacted]";
        if (/uuid|udid/i.test(key)) uuid++;
        else serial++;
        continue;
      }
      if (typeof value === "string") {
        const cleaned = stripDtcStatements(value);
        if (cleaned.removed > 0) {
          record[key] = cleaned.text;
          serial += cleaned.removed;
        }
        continue;
      }
      walk(value);
    }
  };
  walk(clone);
  return { clone, serial, uuid };
}

function blankTallies(payload: unknown): unknown {
  const clone = structuredClone(payload);
  const walk = (node: unknown, key: string | null): void => {
    if (Array.isArray(node)) {
      node.forEach((item, i) => walk(item, key));
      return;
    }
    if (node === null || typeof node !== "object") return;
    for (const [childKey, value] of Object.entries(node)) {
      if (key !== null && TALLY_KEYS[key] === true) {
        // Parent is a tally container: blank every value (preserve
        // keys/structure so downstream readers see a sane shape) so
        // the pattern scan never sees raw counts. Strings are
        // blanked with same-length spaces; numbers become short
        // strings (so the JSON stays valid AND the `serial":NN`
        // pattern cannot fire — the value is now a quoted
        // single-digit string, not a digit run).
        if (typeof value === "string") {
          (node as Record<string, unknown>)[childKey] = " ".repeat(value.length);
        } else if (typeof value === "number") {
          (node as Record<string, unknown>)[childKey] = "0";
        } else if (typeof value === "boolean") {
          (node as Record<string, unknown>)[childKey] = "0";
        } else if (value && typeof value === "object") {
          walk(value, "tally-leaf");
        }
      } else if (TALLY_KEYS[childKey] === true) {
        // The container itself: walk its children with the tally key
        // set so each value is blanked.
        walk(value, childKey);
      } else {
        walk(value, childKey);
      }
    }
  };
  walk(clone, null);
  return clone;
}

function buildScanner(hostAliases: string[]): {
  re: RegExp;
  count: (k: string) => number;
} {
  // We replicate the scanPii regex assembly so we can iterate over
  // every match (scanPii collapses to a count map). hostAliases
  // need the same alternation rules.
  const aliases = parseHostAliases(hostAliases.join(",")).slice(0, MAX_ALIASES);
  let source = PII_RE.source;
  if (aliases.length > 0) {
    const alternation = [...aliases]
      .sort((a, b) => b.length - a.length)
      .map((a) => a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|");
    source += `|(?<hostname_alias>(?<![A-Za-z0-9])(?:${alternation})(?![A-Za-z0-9]))`;
  }
  const re = new RegExp(source, PII_RE.flags + "g");
  return { re, count: () => 0 };
}

function kindOf(match: RegExpExecArray): string | null {
  const groups = match.groups ?? {};
  for (const k of Object.keys(groups)) {
    if (groups[k] !== undefined) return k;
  }
  return null;
}

// Replace every PII match inside ONE string value with the kind's
// placeholder and count the kinds. Value-level rewriting can never
// span JSON structure (keys, colons, braces), so the result is valid
// JSON by construction — no padded-placeholder arithmetic, no
// re-parse gamble.
function rewriteValueText(
  text: string,
  source: string,
): { text: string; kinds: PiiKinds } {
  const re = new RegExp(source, PII_RE.flags + "g");
  const kinds: PiiKinds = {};
  const out: string[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  let safety = 0;
  while ((match = re.exec(text)) !== null) {
    if (++safety > 50_000) break;
    const k = kindOf(match);
    if (k === null) {
      re.lastIndex = match.index + Math.max(1, match[0].length);
      continue;
    }
    out.push(text.slice(cursor, match.index));
    out.push(PLACEHOLDERS[k] ?? "[redacted]");
    cursor = match.index + match[0].length;
    re.lastIndex = cursor;
    if (match[0].length === 0) re.lastIndex++;
    kinds[k] = (kinds[k] ?? 0) + 1;
  }
  out.push(text.slice(cursor));
  return { text: out.join(""), kinds };
}

export function redactPiiPayload(
  payload: unknown,
  hostAliases: string[] = [],
): RedactResult {
  const talliesBlank = blankTallies(payload);
  // Structural pass FIRST: identity-keyed values replaced whole,
  // dtc statements dropped from string values. This is the only
  // guard that removes multi-cell serial material completely
  // (regex value matches leave trailing cells behind).
  const identity = stripIdentityValues(talliesBlank);
  // The rewrite runs over the firmware-blanked clone, so exempt
  // iBoot-version values can neither inflate the counts nor be
  // mangled.
  const firmwareBlanked = blankRegions(identity.clone, false);
  const aliases = parseHostAliases(hostAliases.join(","));
  const unrestricted = aliases.filter((a) => !restrictedAlias(a));
  const restricted = aliases.filter(restrictedAlias);

  // Full-pattern source with unrestricted aliases folded in (they
  // match everywhere, name fields included).
  let unrestrictedSource = PII_RE.source;
  if (unrestricted.length > 0) {
    const alternation = [...unrestricted]
      .sort((a, b) => b.length - a.length)
      .map((a) => a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("|");
    unrestrictedSource +=
      `|(?<hostname_alias>(?<![A-Za-z0-9])(?:${alternation})(?![A-Za-z0-9]))`;
  }

  const kinds: PiiKinds = {};
  if (identity.serial > 0) kinds.serial = (kinds.serial ?? 0) + identity.serial;
  if (identity.uuid > 0) kinds.uuid = (kinds.uuid ?? 0) + identity.uuid;

  // Per-value rewrite: unrestricted patterns apply to every string
  // value; restricted (short / marketing-word) aliases apply only to
  // free text — never the exempt name fields.
  const walk = (node: unknown, key: string | null): void => {
    if (Array.isArray(node)) {
      node.forEach((item) => walk(item, key));
      return;
    }
    if (node === null || typeof node !== "object") return;
    for (const [childKey, value] of Object.entries(node)) {
      if (typeof value === "string") {
        let current = value;
        const hit = rewriteValueText(current, unrestrictedSource);
        current = hit.text;
        for (const [k, v] of Object.entries(hit.kinds)) {
          kinds[k] = (kinds[k] ?? 0) + v;
        }
        if (restricted.length > 0 && !EXEMPT_NAME_FIELDS[childKey]) {
          const aliasSource =
            `(?<hostname_alias>(?<![A-Za-z0-9])(?:${[...restricted]
              .sort((a, b) => b.length - a.length)
              .map((a) => a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
              .join("|")})(?![A-Za-z0-9]))`;
          const aliasHit = rewriteValueText(current, aliasSource);
          current = aliasHit.text;
          for (const [k, v] of Object.entries(aliasHit.kinds)) {
            kinds[k] = (kinds[k] ?? 0) + v;
          }
        }
        if (current !== value) {
          // SAFETY: node came from Object.entries on a walked object,
          // so assigning back through the same key preserves shape.
          (node as Record<string, unknown>)[childKey] = current;
        }
        continue;
      }
      walk(value, childKey);
    }
  };
  walk(firmwareBlanked, null);

  if (Object.keys(kinds).length === 0) {
    // SAFETY: an empty kinds map IS the null contract — callers treat
    // "no hits" identically to the legacy scan's null return.
    return { payload, kinds: null as unknown as PiiKinds };
  }
  return { payload: firmwareBlanked, kinds };
}
