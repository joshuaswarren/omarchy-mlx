import { SchemaNode } from "./schema";

// Tolerance pass for v0.7.14 macOS collectors (#24, #26): they
// hex-encode every IODeviceTree bytes value, so `compatible` /
// `interrupt-names` arrive as hex strings where the schema wants string
// arrays, and a few dt_nodes strings exceed the schema maxLength. This
// pass runs BEFORE validation: it reshapes those values to what the
// schema already describes, so the schema file and its identity hash
// stay untouched. Deterministic, nothing invented: hex is decoded only
// when it yields NUL-separated printable text; truncation only removes
// entries or characters the schema already refuses. Every change is
// recorded and returned so it is auditable (response `coerced` field +
// worker log).

const MAX_RECORDED = 50;

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

function record(changes: string[], path: string, what: string): void {
  if (changes.length < MAX_RECORDED) {
    changes.push(`${path}: ${what}`);
  }
}

// Returns the (possibly reshaped) value; callers assign it back, so a
// decoded array replaces its source string in the parent. Objects and
// arrays are otherwise mutated in place.
function walk(
  value: unknown,
  schema: SchemaNode,
  path: string,
  changes: string[],
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
          record(changes, path, `decoded hex string into ${hex.length}-item array`);
          return walk(hex, schema, path, changes);
        }
        if (value.includes("\u0000")) {
          const parts = value.split("\u0000").filter((p) => p !== "");
          record(changes, path, `split NUL-separated string into ${parts.length}-item array`);
          return walk(parts, schema, path, changes);
        }
        record(changes, path, "wrapped string into 1-item array");
        return walk([value], schema, path, changes);
      }
    }
    if (schema.maxLength !== undefined && value.length > schema.maxLength) {
      record(changes, path, `truncated ${value.length} to ${schema.maxLength} chars`);
      return value.slice(0, schema.maxLength);
    }
    return value;
  }

  if (Array.isArray(value)) {
    const items = schema.items;
    if (items !== undefined) {
      for (let i = 0; i < value.length; i++) {
        value[i] = walk(value[i], items, `${path}[${i}]`, changes);
      }
    }
    if (schema.maxItems !== undefined && value.length > schema.maxItems) {
      record(changes, path, `truncated ${value.length} to ${schema.maxItems} items`);
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
    if (childSchema !== undefined) {
      // SAFETY: value came from JSON.parse of the initiate body (see
      // handleInitiate), so its entries are plain string-keyed JSON.
      (value as Record<string, unknown>)[key] =
        walk(child, childSchema, `${path}.${key}`, changes);
    }
  }
  return value;
}

export function coerceToSchema(
  payload: unknown,
  schema: SchemaNode,
): string[] {
  const changes: string[] = [];
  walk(payload, schema, "$", changes);
  return changes;
}
