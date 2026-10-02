import { SchemaNode, validateSchema } from "./schema";

// Tolerance pass for macOS community submissions (#24, #26 and the M3
// 422 of 2026-10-02T14:21Z): v0.7.14+ collectors hex-encode devicetree
// bytes, new chip generations (M3 iop-ane / ascwrap, M4, M5) bring
// shapes the bundled schema never anticipated, and one unexpected
// diagnostic value must never reject a whole submission. Two passes in
// one tree walk, both deterministic, nothing invented:
//   coerce  — decode hex/NUL string encodings into the schema's string
//             arrays, cut over-long strings/arrays to the schema's own
//             limits (applies everywhere);
//   sanitize— inside the diagnostic blocks (ane_port_detail,
//             ane_macos, ane_linux) any value the schema would still
//             reject is parked as capped JSON under the payload root's
//             `unparsed` key and replaced by null / removed, so only
//             identity, privacy, size and schema_version can 422.
// The schema file and its identity hash stay untouched: the root
// object has no additionalProperties:false, so the extra `unparsed`
// key validates as-is. Every action is recorded for the response
// `coerced` field and the worker log.

const MAX_RECORDED = 50;
const UNPARSED_ENTRY_CHARS = 1024;
const UNPARSED_MAX_ENTRIES = 64;

export type SanitizeResult = {
  changes: string[];
  unparsed: Record<string, string>;
};

function typeIncludes(node: SchemaNode, type: string): boolean {
  if (node.type === undefined) return false;
  return (Array.isArray(node.type) ? node.type : [node.type]).includes(type);
}

// Decode a collector hex string into its NUL-separated parts. Returns
// null when the string is not even-length lowercase hex or does not
// decode to printable ASCII + NULs (i.e. it is raw binary hex like a
// `reg` dump, not encoded text).
function hexTextToParts(value: string): string[] | null {
  if (value.length < 2 || value.length % 2 !== 0 || /[^0-9a-f]/.test(value)) {
    return null;
  }
  let text = "";
  for (let i = 0; i < value.length; i += 2) {
    const byte = parseInt(value.slice(i, i + 2), 16);
    if (byte !== 0 && (byte < 0x20 || byte > 0x7e)) return null;
    text += byte === 0 ? "\u0000" : String.fromCharCode(byte);
  }
  const parts = text.split("\u0000").filter((p) => p !== "");
  return parts.length > 0 ? parts : null;
}

type Ctx = {
  changes: string[];
  unparsed: Record<string, string>;
  dropped: number;
};

function record(ctx: Ctx, path: string, what: string): void {
  if (ctx.changes.length < MAX_RECORDED) {
    ctx.changes.push(`${path}: ${what}`);
  }
}

// Sentinel returned by walk() meaning "this value cannot be made to
// satisfy the schema; remove it from its container".
const REMOVE = Symbol("remove");

function park(
  ctx: Ctx,
  path: string,
  value: unknown,
  reason: string,
): void {
  if (Object.keys(ctx.unparsed).length >= UNPARSED_MAX_ENTRIES) {
    ctx.dropped += 1;
    return;
  }
  let text: string;
  try {
    text = JSON.stringify(value) ?? String(value);
  } catch {
    text = String(value);
  }
  if (text.length > UNPARSED_ENTRY_CHARS) {
    text = text.slice(0, UNPARSED_ENTRY_CHARS) + "…";
  }
  ctx.unparsed[path] = text;
  record(ctx, path, `parked in unparsed: ${reason}`);
}

function truncateString(
  ctx: Ctx,
  path: string,
  value: string,
  maxLength: number,
): string {
  record(ctx, path, `truncated ${value.length} to ${maxLength} chars`);
  return value.slice(0, maxLength);
}

// Still schema-invalid after coercion? Used inside diagnostic blocks
// only, where the answer decides park-vs-keep.
function schemaViolations(
  value: unknown,
  schema: SchemaNode,
  path: string,
): string[] {
  const errors: string[] = [];
  validateSchema(value, schema, path, errors);
  return errors;
}

// Returns the (possibly reshaped) value, or REMOVE when a diagnostic
// value had to be dropped; callers assign it back, so a decoded array
// replaces its source string in the parent. Objects and arrays are
// otherwise mutated in place.
function walk(
  value: unknown,
  schema: SchemaNode,
  path: string,
  ctx: Ctx,
  diagnostic: boolean,
): unknown {
  if (value === null || value === undefined) return value;

  if (typeof value === "string") {
    const items = schema.items;
    const types = schema.type === undefined
      ? []
      : Array.isArray(schema.type) ? schema.type : [schema.type];
    if (types.includes("array") && items !== undefined) {
      const itemTypes = items.type === undefined
        ? []
        : Array.isArray(items.type) ? items.type : [items.type];
      if (itemTypes.includes("string")) {
        const hex = hexTextToParts(value);
        if (hex !== null) {
          record(ctx, path, `decoded hex string into ${hex.length}-item array`);
          return walk(hex, schema, path, ctx, diagnostic);
        }
        if (value.includes("\u0000")) {
          const parts = value.split("\u0000").filter((p) => p !== "");
          record(ctx, path, `split NUL-separated string into ${parts.length}-item array`);
          return walk(parts, schema, path, ctx, diagnostic);
        }
        record(ctx, path, "wrapped string into 1-item array");
        return walk([value], schema, path, ctx, diagnostic);
      }
    }
    let v = value;
    if (schema.maxLength !== undefined && v.length > schema.maxLength) {
      v = truncateString(ctx, path, v, schema.maxLength);
    }
    if (diagnostic && schemaViolations(v, schema, path).length > 0) {
      park(ctx, path, value, `string violates ${types.join("|")}`);
      return typeIncludes(schema, "null") ? null : REMOVE;
    }
    return v;
  }

  if (Array.isArray(value)) {
    const items = schema.items;
    if (items !== undefined) {
      for (let i = value.length - 1; i >= 0; i--) {
        const out = walk(value[i], items, `${path}[${i}]`, ctx, diagnostic);
        if (out === REMOVE) {
          value.splice(i, 1);
          record(ctx, `${path}[${i}]`, "removed item");
        } else {
          value[i] = out;
        }
      }
      // Items that are still invalid (e.g. missing a required
      // non-nullable field) are parked and dropped in diagnostic mode.
      if (diagnostic) {
        for (let i = value.length - 1; i >= 0; i--) {
          const violations = schemaViolations(value[i], items, `${path}[${i}]`);
          if (violations.length > 0) {
            park(ctx, `${path}[${i}]`, value[i], violations[0]);
            value.splice(i, 1);
            record(ctx, `${path}[${i}]`, "removed invalid item");
          }
        }
      }
    }
    if (schema.maxItems !== undefined && value.length > schema.maxItems) {
      record(ctx, path, `truncated ${value.length} to ${schema.maxItems} items`);
      value = value.slice(0, schema.maxItems);
    }
    return value;
  }

  const props = schema.properties;
  const extra = schema.additionalProperties;
  for (const [key, child] of Object.entries(value)) {
    const childSchema = props?.[key] ??
      // additionalProperties is `boolean | SchemaNode`; schemas only.
      (extra !== undefined && extra !== false && extra !== true
        ? extra
        : undefined);
    if (childSchema === undefined) continue;
    // SAFETY: value came from JSON.parse of the initiate body (see
    // handleInitiate), so its entries are plain string-keyed JSON.
    const out = walk(child, childSchema, `${path}.${key}`, ctx,
      diagnostic || (path === "$" &&
        (key === "ane_port_detail" || key === "ane_macos" ||
         key === "ane_linux")));
    if (out === REMOVE) {
      delete (value as Record<string, unknown>)[key];
      record(ctx, `${path}.${key}`, "removed invalid value");
    } else {
      (value as Record<string, unknown>)[key] = out;
    }
  }
  if (diagnostic) {
    for (const key of schema.required ?? []) {
      if (Object.prototype.hasOwnProperty.call(value, key)) continue;
      const childSchema = props?.[key];
      if (childSchema !== undefined && typeIncludes(childSchema, "null")) {
        (value as Record<string, unknown>)[key] = null;
        record(ctx, `${path}.${key}`, "filled required key with null");
      } else {
        record(ctx, `${path}.${key}`, "missing required key");
      }
    }
    // The object survived per-key repair but still violates the schema
    // (e.g. a required non-nullable field was dropped above): park the
    // whole block and let the container drop it, so one bad block never
    // rejects the submission.
    const violations = schemaViolations(value, schema, path);
    if (violations.length > 0) {
      park(ctx, path, value, violations[0]);
      return REMOVE;
    }
  }
  return value;
}

export function coerceToSchema(
  payload: unknown,
  schema: SchemaNode,
): SanitizeResult {
  const ctx: Ctx = { changes: [], unparsed: {}, dropped: 0 };
  const out = walk(payload, schema, "$", ctx, false);
  if (out !== REMOVE && typeof out === "object" && out !== null) {
    const unparsed = ctx.unparsed;
    if (ctx.dropped > 0) {
      unparsed["unparsed:dropped"] = `${ctx.dropped} entries beyond ${UNPARSED_MAX_ENTRIES}`;
    }
    if (Object.keys(unparsed).length > 0) {
      (out as Record<string, unknown>)["unparsed"] = unparsed;
    }
  }
  return { changes: ctx.changes, unparsed: ctx.unparsed };
}
