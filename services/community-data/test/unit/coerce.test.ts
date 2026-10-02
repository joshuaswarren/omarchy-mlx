import { describe, expect, test } from "bun:test";
import payloadSchemaJson from "../../schema/payload-v1.schema.json";
import { SchemaNode, validateSchemaRoot } from "../../src/schema";
import { coerceToSchema } from "../../src/coerce";
import fixtureValue from "./fixtures/payload-v1.json";

// SAFETY: payload-v1.schema.json is the repo's own checked-in schema,
// byte-verified by schema-identity.test.ts.
const schema = payloadSchemaJson as SchemaNode;

// SAFETY: the fixture is the repo's own checked-in v1 payload;
// validateSchemaRoot and coerceToSchema both take `unknown`.
const fixture = structuredClone(fixtureValue) as Record<string, any>;

// v0.7.14 collectors hex-encode bytes: "616e652c743830323000" is
// "ane,t8020\0". Shapes below are synthetic mutations of the macos
// block structure observed in published community rows (M3 iop-ane,
// M4 t8020-class, M5 Max) plus a Neo-style junk block.
const hex = (s: string) => Buffer.from(s, "latin1").toString("hex");

function payloadWithMacos(macos: Record<string, unknown>) {
  const payload = structuredClone(fixture);
  payload.ane_port_detail.macos = macos;
  return payload;
}

// Structural base mirroring the published M4 Pro / M5 Max rows.
function macosBase(): Record<string, unknown> {
  return {
    available: true,
    instances: [{ name: "ane,t8122", matched: "ane,t8122", firmware_loaded: true }],
    ane_nodes: [{
      name: "ane0",
      compatible: ["ane,t8122"],
      phandle: 361,
    }],
    dart_nodes: [{ name: "mapper-ane0" }],
    mailbox_nodes: [],
    pmgr_nodes: [],
    coreml: { available: true, compute_units: null, error: null },
    powermetrics: { available: false, power_mw: null, error: "unsupported" },
  };
}

describe("coerceToSchema (#24/#26 macOS 422s)", () => {
  test("#26 exact shape: hex compatible + over-long instance, before and after", () => {
    const payload = payloadWithMacos({
      ...macosBase(),
      dt_nodes: [
        { name: "ane0", compatible: hex("ane,t8020\0") },
        {
          name: "ane0",
          compatible: hex("ane,t8020\0"),
          instance: hex("0123456789abcdef0123456789abcdef01234"),
        },
        { name: "ane0", compatible: hex("ane,t8103\0ane,t8100\0") },
      ],
    });
    const before = validateSchemaRoot(payload, schema);
    expect(before.some((e) => e.includes("compatible"))).toBe(true);
    expect(before.some((e) => e.includes("instance: longer than 64"))).toBe(true);

    const { changes } = coerceToSchema(payload, schema);
    expect(changes.some((c) =>
      c.startsWith("$.ane_port_detail.macos.dt_nodes[0].compatible: decoded hex"))).toBe(true);
    expect(changes.some((c) => c.includes("[1].instance: truncated 74 to 64"))).toBe(true);
    expect(validateSchemaRoot(payload, schema)).toEqual([]);
    const dn = payload.ane_port_detail.macos.dt_nodes;
    expect(dn[0].compatible).toEqual(["ane,t8020"]);
    expect(dn[1].compatible).toEqual(["ane,t8020"]);
    expect(dn[2].compatible).toEqual(["ane,t8103", "ane,t8100"]);
    expect(dn[1].instance).toHaveLength(64);
  });

  test("#24 shape: hex compatible on 11 nodes, hex interrupt-names, >16-entry list", () => {
    const nodes: Record<string, unknown>[] = [];
    for (let i = 0; i < 11; i++) {
      nodes.push({
        name: `ane${i}`,
        compatible: hex("ane,t8142\0ane,t8140\0"),
        "interrupt-names": hex("irq0\0irq1\0"),
      });
    }
    nodes.push({
      name: "dart-ane0",
      compatible: hex(
        Array.from({ length: 20 }, (_, i) => `c${i}`).join("\0") + "\0",
      ),
    });
    const payload = payloadWithMacos({ ...macosBase(), dt_nodes: nodes });
    expect(validateSchemaRoot(payload, schema).length).toBeGreaterThan(0);

    const { changes } = coerceToSchema(payload, schema);
    expect(validateSchemaRoot(payload, schema)).toEqual([]);
    const macos = payload.ane_port_detail.macos;
    expect(macos.dt_nodes[0]["interrupt-names"]).toEqual(["irq0", "irq1"]);
    expect(macos.dt_nodes[11].compatible).toHaveLength(16);
    expect(changes.some((c) => c.includes("truncated 20 to 16 items"))).toBe(true);
  });

  test("raw binary hex is not decoded as text; plain strings wrap or split", () => {
    const payload = payloadWithMacos({
      ...macosBase(),
      dt_nodes: [
        // "deadbeef" decodes to non-printable bytes: kept as one item.
        { name: "ane0", compatible: "deadbeef00" },
        // Plain compatible text with no encoding: wrapped, not rejected.
        { name: "ane1", compatible: "ane,t6000" },
        // NUL-separated plain text: split on NUL.
        { name: "ane2", compatible: "ane,t8103\0ane,t8100\0" },
      ],
    });
    const { changes } = coerceToSchema(payload, schema);
    expect(validateSchemaRoot(payload, schema)).toEqual([]);
    const dn = payload.ane_port_detail.macos.dt_nodes;
    expect(dn[0].compatible).toEqual(["deadbeef00"]);
    expect(dn[1].compatible).toEqual(["ane,t6000"]);
    expect(dn[2].compatible).toEqual(["ane,t8103", "ane,t8100"]);
    expect(changes.some((c) => c.includes("wrapped string"))).toBe(true);
  });

  test("schema identity inputs are untouched", () => {
    expect(validateSchemaRoot(structuredClone(fixture), schema)).toEqual([]);
    const payload = payloadWithMacos({
      ...macosBase(),
      dt_nodes: [{ name: "ane0", compatible: hex("ane,t8020\0") }],
    });
    coerceToSchema(payload, schema);
    expect(validateSchemaRoot(structuredClone(fixture), schema)).toEqual([]);
  });
});

// The 2026-10-02T14:21Z M3 class: one unexpected diagnostic value must
// never reject a submission. Everything below is shaped after the M3
// (iop-ane / ascwrap-v6), M4 (t8020-class) and M5 Max structures seen
// in the community rows, with synthetic junk where the new generation
// would send something the schema never anticipated.
describe("coerceToSchema never rejects diagnostics", () => {
  test("M3 shape: unknown ascwrap-v6 key kept, bad items parked, block survives", () => {
    const payload = payloadWithMacos({
      ...macosBase(),
      dt_nodes: [
        {
          name: "iop-ane0",
          // New-generation key the schema never knew: unknown keys are
          // allowed, so this is stored as-is.
          "ascwrap-v6": { version: 6, state: "unknown" },
          // Nested list where a string list is expected.
          compatible: [["ane,t8112", "ane,t8110"]],
        },
        // Missing name: dt_nodes.name is nullable, so it is filled
        // with null and the node is kept.
        { compatible: ["ane,t8112"] },
      ],
      // Missing the required NON-nullable name: whole item parked.
      ane_nodes: [{ compatible: ["ane,t8112"] }],
    });
    const before = validateSchemaRoot(payload, schema);
    expect(before.length).toBeGreaterThan(0);

    const { changes, unparsed } = coerceToSchema(payload, schema);
    expect(validateSchemaRoot(payload, schema)).toEqual([]);

    // Unknown key kept inline.
    const macos = payload.ane_port_detail.macos;
    expect(macos.dt_nodes[0]["ascwrap-v6"]).toEqual({ version: 6, state: "unknown" });
    // Nested list item parked and dropped, list left empty-but-valid.
    expect(macos.dt_nodes[0].compatible).toEqual([]);
    expect(macos.dt_nodes).toHaveLength(2);
    expect(macos.dt_nodes[1].name).toBeNull();
    // The unnameable ane_nodes entry was parked under the root
    // `unparsed` key and removed from the array.
    expect(macos.ane_nodes).toHaveLength(0);
    expect(unparsed["$.ane_port_detail.macos.ane_nodes[0]"]).toBeDefined();
    expect(changes.length).toBeGreaterThan(0);
  });

  test("M3 shape: a corrupted required scalar drops the whole block with a flag", () => {
    const payload = payloadWithMacos({
      ...macosBase(),
      // Required boolean carrying a string: unrepairable, so the whole
      // macos block is parked and removed instead of 422-ing.
      available: "yes",
    });
    expect(validateSchemaRoot(payload, schema).length).toBeGreaterThan(0);

    const { unparsed } = coerceToSchema(payload, schema);
    expect(validateSchemaRoot(payload, schema)).toEqual([]);
    expect(payload.ane_port_detail.macos).toBeUndefined();
    expect(unparsed["$.ane_port_detail.macos.available"]).toBe('"yes"');
    expect(unparsed["$.ane_port_detail.macos"]).toBeDefined();
  });

  test("M4 shape plus wrong-typed instance and boolean junk still validates", () => {
    const payload = payloadWithMacos({
      ...macosBase(),
      dt_nodes: [
        {
          name: "ane0",
          compatible: hex("ane,t8020\0ane,t8021\0"),
          // Object where the schema wants a string.
          instance: { raw: "not-a-string" },
          // Boolean where a string is expected.
          "device_type": false,
        },
      ],
      instances: [{ name: "ane,t8020", matched: "ane,t8020", firmware_loaded: "maybe" }],
    });
    expect(validateSchemaRoot(payload, schema).length).toBeGreaterThan(0);

    const { unparsed } = coerceToSchema(payload, schema);
    expect(validateSchemaRoot(payload, schema)).toEqual([]);
    const dn = payload.ane_port_detail.macos.dt_nodes;
    expect(dn[0].compatible).toEqual(["ane,t8020", "ane,t8021"]);
    expect(dn[0].instance).toBeUndefined();
    expect(dn[0]["device_type"]).toBeUndefined();
    // The instances entry lost its boolean firmware_loaded and was
    // parked wholesale (required non-nullable scalar).
    expect(
      Object.keys(unparsed).some((k) => k.includes("instances")),
    ).toBe(true);
  });

  test("Neo-style junk: oversized unknown-key strings pass, unparsed caps entries", () => {
    const junk: Record<string, unknown> = {};
    for (let i = 0; i < 80; i++) {
      junk[`neo-key-${i}`] = "x".repeat(50);
    }
    const payload = payloadWithMacos({
      ...macosBase(),
      dt_nodes: [{ name: "ane0", ...junk }],
    });
    // Unknown keys carry no constraints: no 422 even before sanitizing.
    expect(validateSchemaRoot(payload, schema)).toEqual([]);
    const { unparsed } = coerceToSchema(payload, schema);
    // Unknown keys are kept inline (schema permits extras), nothing parked.
    expect(Object.keys(unparsed).length).toBe(0);
    expect(payload.ane_port_detail.macos.dt_nodes[0]["neo-key-79"]).toBe("x".repeat(50));
  });

  test("unparsed entries are size-capped and the overflow is flagged", () => {
    const payload = payloadWithMacos({
      ...macosBase(),
      dt_nodes: Array.from({ length: 20 }, (_, i) => ({
        name: "ane0",
        instance: { blob: "y".repeat(5000), idx: i },
      })),
    });
    const { unparsed, changes } = coerceToSchema(payload, schema);
    expect(validateSchemaRoot(payload, schema)).toEqual([]);
    for (const text of Object.values(unparsed)) {
      expect(text.length).toBeLessThanOrEqual(1025);
    }
    expect(Object.keys(unparsed).length).toBeLessThanOrEqual(65);
    expect(
      changes.some((c) => c.includes("unparsed")) || unparsed["unparsed:dropped"] !== undefined,
    ).toBe(true);
  });
});
