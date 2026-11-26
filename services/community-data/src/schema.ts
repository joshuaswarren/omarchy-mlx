// Minimal JSON Schema validator covering exactly the keywords the
// payload schema uses: type (incl. unions), const, enum, required,
// properties, additionalProperties (false | schema), items, maxItems,
// maxLength, minimum, maximum, pattern. One source of truth lives in
// schema/payload-v1.schema.json; this validator interprets it.

export type SchemaNode = {
  type?: string | string[];
  const?: unknown;
  enum?: unknown[];
  required?: string[];
  properties?: Record<string, SchemaNode>;
  additionalProperties?: boolean | SchemaNode;
  items?: SchemaNode;
  maxItems?: number;
  maxLength?: number;
  minimum?: number;
  maximum?: number;
  pattern?: string;
};

const patternCache = new Map<string, RegExp>();

function compiled(pattern: string): RegExp {
  let re = patternCache.get(pattern);
  if (!re) {
    re = new RegExp(pattern);
    patternCache.set(pattern, re);
  }
  return re;
}

function matchesType(value: unknown, type: string): boolean {
  switch (type) {
    case "object":
      return value !== null && typeof value === "object" && !Array.isArray(value);
    case "array":
      return Array.isArray(value);
    case "string":
      return typeof value === "string";
    case "integer":
      return typeof value === "number" && Number.isInteger(value);
    case "number":
      return typeof value === "number" && Number.isFinite(value);
    case "boolean":
      return typeof value === "boolean";
    case "null":
      return value === null;
    default:
      return false;
  }
}

function sameJson(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

export function validateSchema(
  value: unknown,
  schema: SchemaNode,
  path: string,
  errors: string[],
): void {
  if (schema.type !== undefined) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((t) => matchesType(value, t))) {
      errors.push(`${path}: expected type ${types.join(" | ")}`);
      return;
    }
  }
  if (schema.const !== undefined && !sameJson(value, schema.const)) {
    errors.push(`${path}: must equal ${JSON.stringify(schema.const)}`);
  }
  if (schema.enum !== undefined && !schema.enum.some((v) => sameJson(value, v))) {
    errors.push(`${path}: must be one of ${JSON.stringify(schema.enum)}`);
  }
  if (value !== null && (Array.isArray(value) || typeof value === "object")) {
    const entries: [string, unknown][] = Array.isArray(value)
      ? value.map((v, i) => [String(i), v])
      : Object.entries(value as Record<string, unknown>);
    for (const key of schema.required ?? []) {
      if (!entries.some(([k]) => k === key)) {
        errors.push(`${path}: missing required property ${key}`);
      }
    }
    const props = schema.properties ?? {};
    const extra = schema.additionalProperties;
    for (const [key, item] of entries) {
      if (Object.prototype.hasOwnProperty.call(props, key)) {
        validateSchema(item, props[key], `${path}.${key}`, errors);
      } else if (extra === false) {
        errors.push(`${path}: additional property ${key} is not allowed`);
      } else if (extra !== undefined && extra !== true) {
        validateSchema(item, extra, `${path}.${key}`, errors);
      }
    }
    if (Array.isArray(value) && schema.maxItems !== undefined && value.length > schema.maxItems) {
      errors.push(`${path}: more than ${schema.maxItems} items`);
    }
  }
  if (Array.isArray(value) && schema.items !== undefined) {
    value.forEach((item, i) => {
      validateSchema(item, schema.items as SchemaNode, `${path}[${i}]`, errors);
    });
  }
  if (typeof value === "string") {
    if (schema.maxLength !== undefined && value.length > schema.maxLength) {
      errors.push(`${path}: longer than ${schema.maxLength} characters`);
    }
    if (schema.pattern !== undefined && !compiled(schema.pattern).test(value)) {
      errors.push(`${path}: does not match ${schema.pattern}`);
    }
  }
  if (typeof value === "number" && schema.minimum !== undefined && value < schema.minimum) {
    errors.push(`${path}: less than ${schema.minimum}`);
  }
  if (typeof value === "number" && schema.maximum !== undefined && value > schema.maximum) {
    errors.push(`${path}: greater than ${schema.maximum}`);
  }
}

export function validateSchemaRoot(
  value: unknown,
  schema: SchemaNode,
): string[] {
  const errors: string[] = [];
  validateSchema(value, schema, "$", errors);
  return errors;
}
