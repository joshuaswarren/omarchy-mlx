import { describe, expect, test } from "bun:test";
import { getPublished, initiateSubmission, rebuildCaches } from "../../src/store";

// Minimal D1 fake: records every SQL string, and each bound statement
// either returns a row or throws. Used to pin the 2026-10-03 incident:
// a code deploy that selects submissions.pii_redacted against a D1
// where migration 0003 has not been applied must degrade to a
// legacy-column read (pii_redacted = null), never 500.

const FULL_ROW = {
  content_sha256: "a".repeat(64),
  received_at: 1,
  updated_at: 1,
  kind: "deep",
  schema_version: 2,
  arch: "aarch64",
  model: "Mac mini",
  chip: "apple,t8103",
  kernel: "6.9.1",
  mesa_driver: null,
  mesa_device: null,
  mlx_version: null,
  mlx_device: null,
  summary: "{}",
  archive_total_bytes: null,
  archive_chunk_bytes: null,
  archive_chunk_count: null,
  archive_chunk_sha256: null,
  pow_difficulty: 18,
  published: 1,
  published_at: 1,
  pii_redacted: '{"serial":4}',
};

class FakeD1 {
  statements: string[] = [];
  constructor(
    private readonly behavior: (sql: string) => (
      { row?: unknown; error?: string; changes?: number }
    ),
  ) {}

  prepare(sql: string) {
    const all = async () => {
      this.statements.push(sql);
      const out = this.behavior(sql);
      if (out.error) throw new Error(out.error);
      return { results: out.rows ?? [] };
    };
    return {
      bind: (..._args: unknown[]) => ({
        first: async () => {
          this.statements.push(sql);
          const out = this.behavior(sql);
          if (out.error) throw new Error(out.error);
          return out.row ?? null;
        },
        run: async () => {
          this.statements.push(sql);
          const out = this.behavior(sql);
          if (out.error) throw new Error(out.error);
          return { meta: { changes: out.changes ?? 1 } };
        },
        all,
      }),
      all,
    };
  }

  batch(stmts: unknown[]) {
    return Promise.resolve(stmts.map(() => ({})));
  }
}

describe("store row reads tolerate a pre-0003 D1 (2026-10-03 incident)", () => {
  test("new-schema DB returns the pii_redacted column", async () => {
    const db = new FakeD1(() => ({ row: FULL_ROW }));
    const row = await getPublished(db as never, "a".repeat(64));
    expect(row).not.toBeNull();
    expect(row?.pii_redacted).toBe('{"serial":4}');
  });

  test("pre-0003 DB falls back to legacy columns with pii_redacted null", async () => {
    const db = new FakeD1((sql) => {
      if (sql.includes("pii_redacted")) {
        return { error: "D1_TYPE_ERROR: no such column: pii_redacted" };
      }
      return { row: { ...FULL_ROW, pii_redacted: undefined } };
    });
    const row = await getPublished(db as never, "a".repeat(64));
    expect(row).not.toBeNull();
    expect(row?.pii_redacted).toBeNull();
    // The fallback actually ran the legacy SELECT.
    expect(db.statements.some((s) => !s.includes("pii_redacted"))).toBe(true);
  });

  test("pre-0003 DB: initiate inserts via the legacy statement", async () => {
    const db = new FakeD1((sql) => {
      if (sql.includes("INSERT OR IGNORE") && sql.includes("pii_redacted")) {
        return { error: "D1_TYPE_ERROR: no such column: pii_redacted" };
      }
      if (sql.includes("SELECT")) {
        return { row: { ...FULL_ROW, pii_redacted: undefined } };
      }
      return { changes: 1 };
    });
    const row = await initiateSubmission(db as never, {
      contentSha: "b".repeat(64),
      now: 1,
      kind: "deep",
      schemaVersion: 2,
      arch: null,
      model: null,
      chip: null,
      kernel: null,
      mesaDriver: null,
      mesaDevice: null,
      mlxVersion: null,
      mlxDevice: null,
      summary: "{}",
      archive: null,
      powDifficulty: 18,
      piiRedacted: { serial: 3 },
    });
    expect(row).not.toBeNull();
    // The legacy INSERT (without pii_redacted) ran.
    expect(db.statements.some(
      (s) => s.includes("INSERT OR IGNORE") && !s.includes("pii_redacted"),
    )).toBe(true);
  });

  test("unrelated select errors are never swallowed", async () => {
    const db = new FakeD1(() => ({ error: "D1_CONNECTION_ERROR: offline" }));
    await expect(
      getPublished(db as never, "c".repeat(64)),
    ).rejects.toThrow("offline");
  });

  test("rebuildCaches writes an index whose body parses as JSON", async () => {
    // 2026-10-03: the results-cache template shipped without the
    // root object's closing brace, so GET /v1/results served JSON
    // that failed at EOF (strict consumers broke; latest.jsonl
    // parsed fine and hid it). Pin the template.
    let savedText: string | null = null;
    const db = {
      prepare: (_sql: string) => {
        const runBind = (args: unknown[]) => async () => {
          for (const a of args) {
            if (typeof a === "string" && a.startsWith("{") &&
                a.includes('"results":[')) {
              savedText = a;
            }
          }
          return { meta: { changes: 1 } };
        };
        return {
          bind: (...args: unknown[]) => ({ run: runBind(args) }),
          all: async () => ({
            results: [{
              content_sha256: "d".repeat(64),
              summary: '{"model":"Mac mini"}',
            }],
          }),
        };
      },
      batch: async (stmts: Array<{ run: () => Promise<unknown> }>) =>
        Promise.all(stmts.map((s) => s.run())),
    };
    await rebuildCaches(db as never, 1760000000);
    expect(savedText).not.toBeNull();
    const doc = JSON.parse(savedText as string);
    expect(doc.count).toBe(1);
    expect(doc.results).toHaveLength(1);
    expect(doc.results[0].content_sha256).toBe("d".repeat(64));
  });
});
