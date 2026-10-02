import { describe, expect, test } from "bun:test";
import payloadSchemaJson from "../../schema/payload-v1.schema.json";
import { SchemaNode, validateSchemaRoot } from "../../src/schema";
import { coerceToSchema } from "../../src/coerce";
import fixtureValue from "./fixtures/payload-v1.json";

// SAFETY: payload-v1.schema.json is the repo's own checked-in schema,
// byte-verified by schema-identity.test.ts.
const schema = payloadSchemaJson as SchemaNode;

// The dt_nodes keys these tests exercise, with the schema's own types.
type DtNode = {
  name: string;
  compatible?: string[] | string | null;
  "interrupt-names"?: string[] | string | null;
  instance?: string | null;
};

type PayloadFixture = {
  ane_port_detail: { macos: { dt_nodes: DtNode[] } };
};

// SAFETY: the fixture is the repo's own checked-in v1 payload, whose
// ane_port_detail.macos block this type mirrors for dt_nodes access.
const fixture = structuredClone(fixtureValue) as PayloadFixture;

// v0.7.14 collectors hex-encode bytes: "616e652c743830323000" is
// "ane,t8020\0". The shapes below are the failing ones reported in
// #24 (M5 Max) and #26 (M1 Pro), with synthetic values.
const hex = (s: string) => Buffer.from(s, "latin1").toString("hex");

function payloadWithDtNodes(dtNodes: DtNode[]): PayloadFixture {
  const payload = structuredClone(fixture);
  payload.ane_port_detail.macos.dt_nodes = dtNodes;
  return payload;
}

describe("coerceToSchema (#24/#26 macOS 422s)", () => {
  test("#26 exact shape: hex compatible + over-long instance, before and after", () => {
    const payload = payloadWithDtNodes([
      { name: "ane0", compatible: hex("ane,t8020\0") },
      {
        name: "ane0",
        compatible: hex("ane,t8020\0"),
        instance: hex("0123456789abcdef0123456789abcdef01234"),
      },
      { name: "ane0", compatible: hex("ane,t8103\0ane,t8100\0") },
    ]);
    const before = validateSchemaRoot(payload, schema);
    expect(before.some((e) => e.includes("compatible"))).toBe(true);
    expect(before.some((e) => e.includes("instance: longer than 64"))).toBe(true);

    const changes = coerceToSchema(payload, schema);
    expect(changes.some((c) =>
      c.startsWith("$.ane_port_detail.macos.dt_nodes[0].compatible: decoded hex"))).toBe(true);
    expect(changes.some((c) => c.includes("[1].instance: truncated 74 to 64"))).toBe(true);
    expect(validateSchemaRoot(payload, schema)).toEqual([]);
    expect(payload.ane_port_detail.macos.dt_nodes[0].compatible).toEqual(["ane,t8020"]);
    expect(payload.ane_port_detail.macos.dt_nodes[1].compatible).toEqual(["ane,t8020"]);
    expect(payload.ane_port_detail.macos.dt_nodes[2].compatible).toEqual([
      "ane,t8103",
      "ane,t8100",
    ]);
    expect(payload.ane_port_detail.macos.dt_nodes[1].instance).toHaveLength(64);
  });

  test("#24 shape: hex compatible on 11 nodes, hex interrupt-names, >16-entry list", () => {
    const nodes: DtNode[] = [];
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
    const payload = payloadWithDtNodes(nodes);
    expect(validateSchemaRoot(payload, schema).length).toBeGreaterThan(0);

    const changes = coerceToSchema(payload, schema);
    expect(validateSchemaRoot(payload, schema)).toEqual([]);
    expect(payload.ane_port_detail.macos.dt_nodes[0]["interrupt-names"]).toEqual([
      "irq0",
      "irq1",
    ]);
    expect(payload.ane_port_detail.macos.dt_nodes[11].compatible).toHaveLength(16);
    expect(changes.some((c) => c.includes("truncated 20 to 16 items"))).toBe(true);
  });

  test("raw binary hex is not decoded as text; plain strings wrap or split", () => {
    const payload = payloadWithDtNodes([
      // "deadbeef" decodes to non-printable bytes: kept as one item.
      { name: "ane0", compatible: "deadbeef00" },
      // Plain compatible text with no encoding: wrapped, not rejected.
      { name: "ane1", compatible: "ane,t6000" },
      // NUL-separated plain text: split on NUL.
      { name: "ane2", compatible: "ane,t8103\0ane,t8100\0" },
    ]);
    const changes = coerceToSchema(payload, schema);
    expect(validateSchemaRoot(payload, schema)).toEqual([]);
    expect(payload.ane_port_detail.macos.dt_nodes[0].compatible).toEqual(["deadbeef00"]);
    expect(payload.ane_port_detail.macos.dt_nodes[1].compatible).toEqual(["ane,t6000"]);
    expect(payload.ane_port_detail.macos.dt_nodes[2].compatible).toEqual([
      "ane,t8103",
      "ane,t8100",
    ]);
    expect(changes.some((c) => c.includes("wrapped string"))).toBe(true);
  });

  test("schema identity inputs are untouched", () => {
    expect(validateSchemaRoot(structuredClone(fixture), schema)).toEqual([]);
    const payload = payloadWithDtNodes([
      { name: "ane0", compatible: hex("ane,t8020\0") },
    ]);
    coerceToSchema(payload, schema);
    expect(validateSchemaRoot(structuredClone(fixture), schema)).toEqual([]);
  });
});
