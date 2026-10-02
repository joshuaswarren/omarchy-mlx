import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "bun:test";
import payloadSchemaJson from "../../schema/payload-v1.schema.json";
import payloadE2ESchemaJson from "../../schema/payload-v1-e2e.schema.json";
import { SCHEMA_IDENTITY } from "../../src/routes";
import { sha256Hex } from "../../src/hash";

const schemaDir = join(import.meta.dir, "..", "..", "schema");
const schemaFiles = ["payload-v1-e2e.schema.json", "payload-v1.schema.json"];
const schemaBuffers = schemaFiles.map((name) =>
  readFileSync(join(schemaDir, name)),
);
const allFields = [
  ...Object.keys(payloadSchemaJson.properties),
  ...Object.keys(payloadE2ESchemaJson.properties),
];
const expectedFieldsHash = sha256Hex([...new Set(allFields)].sort().join("\n"));
const expectedSchemaHash = sha256Hex(
  new Uint8Array(Buffer.concat(schemaBuffers)),
);

describe("SCHEMA_IDENTITY", () => {
  test("schema_versions match the bundled JSON schema", () => {
    const declared = payloadSchemaJson.properties.schema_version.enum;
    expect(SCHEMA_IDENTITY.schema_versions).toEqual(declared);
  });

  test("fields_sha256 matches the sorted field names of the bundled schema", async () => {
    expect(SCHEMA_IDENTITY.fields_sha256).toBe(await expectedFieldsHash);
  });

  test("schema_sha256 matches the raw bundled schema bytes", async () => {
    expect(SCHEMA_IDENTITY.schema_sha256).toBe(await expectedSchemaHash);
  });

  test("ane_port is declared in the bundled schema (the 2026-09-17 incident)", () => {
    expect(payloadSchemaJson.properties).toHaveProperty("ane_port");
  });

  test("ane_port_detail is declared in the bundled schema (v0.6.1)", () => {
    expect(payloadSchemaJson.properties).toHaveProperty("ane_port_detail");
  });

  test("e2e-only fields are declared only in the e2e schema", () => {
    for (const field of ["test_id", "install_path", "asahi_image", "overall", "encryption", "boot_separate"]) {
      expect(payloadE2ESchemaJson.properties).toHaveProperty(field);
      expect(payloadSchemaJson.properties).not.toHaveProperty(field);
    }
  });
});
