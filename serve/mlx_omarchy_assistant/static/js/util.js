// I/O boundary helpers. Validate untyped JSON at the boundary using
// duck-typed checks; the rest of the UI branches on the validated shape.

function arrayIsArray(value) { return Array.isArray(value); }
function plainObjectCheck(value) {
  return value !== null && !arrayIsArray(value) && value instanceof Object;
}
function functionCheck(value) { return value instanceof Function; }
function stringCheck(value) {
  if (value === null || value === undefined) return false;
  return Object.prototype.toString.call(value) === "[object String]";
}

export function asString(value) {
  return stringCheck(value) ? String(value) : undefined;
}

export function asNumber(value) {
  // Accepts primitive JSON numbers; Number.isFinite rejects NaN, Infinity,
  // and non-number values without coercing strings or booleans.
  if (Number.isFinite(value)) return value;
  return undefined;
}

export function tokenFromHash(hash) {
  // Launch URLs carry the one-use token as /#token=VALUE. Returns the raw
  // token value, or null when absent.
  if (!hash) return null;
  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  if (!raw) return null;
  return new URLSearchParams(raw).get("token");
}

export function asObject(value) {
  return plainObjectCheck(value) ? value : undefined;
}

export function isPlainObject(value) { return plainObjectCheck(value); }
export function isFunctionValue(value) { return functionCheck(value); }

export function swallow() { return undefined; }
