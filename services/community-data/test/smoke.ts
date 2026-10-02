// Local end-to-end smoke against `wrangler dev` (local D1, no
// Cloudflare credentials, no network). Run: bun test/smoke.ts
//
// Covers: full multi-chunk submission, resumed upload (only missing
// chunks), replay dedupe, hash mismatch, oversize archive, oversize
// chunk, failed/absent proof of work, PII refusal, incomplete complete,
// archive round-trip, cron cache rebuild, and GC of stale submissions.
import { spawn, spawnSync } from "node:child_process";
import { rmSync } from "node:fs";
import { join } from "node:path";

const PORT = Number(process.env.SMOKE_PORT ?? 8799);
const BASE = "http://127.0.0.1:" + PORT;
const SERVICE = join(import.meta.dir, "..");
const STATE = join(SERVICE, ".wrangler", "smoke-state");
const WRANGLER = join(SERVICE, "node_modules", ".bin", "wrangler");
const DB_NAME = "mlx-omarchy-community";
const CHUNK = 768 * 1024;
const POW_BITS = 18;

function sha256Hex(data: Uint8Array): string {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(data);
  return hasher.digest("hex");
}

function leadingZeroBits(hex: string): number {
  let bits = 0;
  for (const ch of hex) {
    const nibble = parseInt(ch, 16);
    if (nibble === 0) { bits += 4; continue; }
    if (nibble < 2) bits += 3;
    else if (nibble < 4) bits += 2;
    else if (nibble < 8) bits += 1;
    break;
  }
  return bits;
}

function solvePow(contentSha: string, difficulty = POW_BITS): string {
  for (let nonce = 0; ; nonce++) {
    const hasher = new Bun.CryptoHasher("sha256");
    hasher.update(`${contentSha}:${nonce}`);
    if (leadingZeroBits(hasher.digest("hex")) >= difficulty) return String(nonce);
  }
}

function payload(overrides: Record<string, unknown> = {}) {
  return {
    schema_version: 1,
    kind: "deep",
    generated_at: "2026-09-03T12:00:00Z",
    arch: "aarch64",
    model: "Apple Mac mini (M1, 2020)",
    chip: "apple,t8103",
    kernel: "6.9.1-asahi",
    mesa_driver: "Asahi Vulkan",
    mesa_device: "Apple M1 (G13G B0)",
    mlx_version: "0.3.2",
    mlx_device: "gpu",
    source_commit: null,
    repo_dirty: false,
    redaction_summary: { mac: 2 },
    files: [],
    ane_dt_node: false,
    ane_dt_compatible: null,
    ane_port: null,
    ane_port_detail: null,
    boot_chain: null,
    cmdline: null,
    core_shortfall: false,
    cpu_online: 8,
    cpu_present: 8,
    hotplug_control: true,
    benchmark: [],
    ...overrides,
  };
}

type D1Rows = Array<{ results: Array<Record<string, unknown>> }>;

async function d1(sql: string): Promise<D1Rows> {
  const proc = spawnSync(WRANGLER, [
    "d1", "execute", DB_NAME, "--local", "--persist-to", STATE, "--json", "--command", sql,
  ], { cwd: SERVICE, encoding: "utf8" });
  if (proc.status !== 0) {
    throw new Error(`d1 execute failed: ${proc.stdout} ${proc.stderr}`);
  }
  return JSON.parse(proc.stdout) as D1Rows;
}

function delay(ms: number): Promise<void> {
  const { promise, resolve } = Promise.withResolvers<void>();
  setTimeout(resolve, ms);
  return promise;
}

function makeArchive(chunks = 3, seed = 0): Uint8Array {
  const data = new Uint8Array((chunks - 1) * CHUNK + 12345);
  for (let i = 0; i < data.length; i++) data[i] = (i * 7 + 13 + seed) & 0xff;
  return data;
}

async function initiate(
  archive: Uint8Array,
  overrides: Record<string, unknown> = {},
) {
  const sha = sha256Hex(archive);
  const count = Math.ceil(archive.length / CHUNK);
  const hashes: string[] = [];
  for (let i = 0; i < count; i++) {
    hashes.push(sha256Hex(archive.subarray(i * CHUNK, (i + 1) * CHUNK)));
  }
  const body = {
    schema_version: 1,
    kind: "deep",
    content_sha256: sha,
    payload: payload(),
    archive: {
      total_bytes: archive.length,
      chunk_bytes: CHUNK,
      chunk_count: count,
      chunk_sha256: hashes,
    },
    pow: { nonce: solvePow(sha), difficulty: POW_BITS },
    ...overrides,
  };
  const res = await fetch(`${BASE}/v1/submit`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  return { res, body: await res.json(), sha, hashes };
}

async function uploadChunk(sha: string, idx: number, bytes: Uint8Array) {
  const res = await fetch(`${BASE}/v1/submit/${sha}/chunk/${idx}`, {
    method: "POST",
    headers: { "content-type": "application/octet-stream" },
    body: bytes,
  });
  return { res, body: await res.json() };
}

async function complete(sha: string) {
  const res = await fetch(`${BASE}/v1/submit/${sha}/complete`, { method: "POST" });
  return { res, body: await res.json() };
}

async function d1(sql: string): Promise<any> {
  const proc = spawnSync(WRANGLER, [
    "d1", "execute", DB_NAME, "--local", "--persist-to", STATE, "--json", "--command", sql,
  ], { cwd: SERVICE, encoding: "utf8" });
  if (proc.status !== 0) {
    throw new Error(`d1 execute failed: ${proc.stdout} ${proc.stderr}`);
  }
  return JSON.parse(proc.stdout);
}

type Scenario = { name: string; fn: () => Promise<string> };
const scenarios: Scenario[] = [];
function scenario(name: string, fn: () => Promise<string>) {
  scenarios.push({ name, fn });
}

function expect(cond: unknown, message: string): void {
  if (!cond) throw new Error(message);
}

scenario("full multi-chunk submission publishes and serves", async () => {
  const archive = makeArchive(3);
  const init = await initiate(archive);
  expect(init.res.status === 200, `initiate ${init.res.status}`);
  expect(init.body.status === "awaiting_chunks", JSON.stringify(init.body));
  expect(
    JSON.stringify(init.body.missing_chunks) === "[0,1,2]",
    `missing ${JSON.stringify(init.body.missing_chunks)}`,
  );
  expect(typeof init.body.receipt_url === "string", "no receipt_url");

  for (let idx = 0; idx < 3; idx++) {
    const up = await uploadChunk(init.sha, idx, archive.subarray(idx * CHUNK, (idx + 1) * CHUNK));
    expect(up.res.status === 200 && up.body.status === "stored", `chunk ${idx}: ${up.res.status}`);
  }
  const fin = await complete(init.sha);
  expect(fin.res.status === 200 && fin.body.status === "stored", JSON.stringify(fin.body));

  const probe = await fetch(`${BASE}/v1/submit/${init.sha}`);
  expect(probe.status === 200, `probe ${probe.status}`);
  const record = await fetch(`${BASE}/v1/results/${init.sha}`);
  expect(record.status === 200, `record ${record.status}`);
  const doc = await record.json();
  expect(doc.summary.chip === "apple,t8103", "chip column missing");
  expect(doc.archive.chunk_count === 3, "archive meta missing");
  return `receipt ${init.body.receipt_url}`;
});

scenario("Linux ANE probe remains inline in a published local row", async () => {
  const archive = makeArchive(1, 23);
  const probe = {
    schema_version: 1, soc: "t6000", board: "air", model: "MacBook Air",
    compatible: ["apple,t6000"], ane_nodes: [], dart_ane_nodes: [],
    pmgr_ane_nodes: [], mailbox_nodes: [], unknown_ane_like: [],
    kernel: { release: "6.17", version: "Linux" }, cmdline: "quiet",
    cmdline_dropped: 0, modules: [], interrupts: [], accel: [],
    platform_devices: [], genpd: [], debug_ane: [], packages: [],
    installed: { driver_loaded: false }, dmesg: { matched: 0, lines: [] },
    check: { rc: 0, lines: [], lines_total: 0 },
    firmware: { rc: 0, lines: [], lines_total: 0 },
    soc_table: { table: "t6000", compared: true, match: true, dt_vs_table: [] },
    unreadable: [], elapsed_ms: 12, truncated: false,
  };
  const init = await initiate(archive, { payload: payload({ ane_linux: { ane_probe: probe } }) });
  expect(init.res.status === 200, "probe initiate " + init.res.status + ": " + JSON.stringify(init.body));
  await uploadChunk(init.sha, 0, archive);
  const fin = await complete(init.sha);
  expect(fin.res.status === 200, "probe complete " + fin.res.status);
  const response = await fetch(BASE + "/v1/results/" + init.sha);
  const doc = await response.json();
  expect(doc.summary.ane_linux.ane_probe.board === "air", "probe board was not stored inline");
  expect(doc.summary.ane_linux.ane_probe.model === "MacBook Air", "probe model was not stored inline");
  expect(doc.summary.unparsed === undefined, "valid probe was parked under unparsed");
  return "synthetic local probe stored inline";
});

scenario("e2e kind initiates and publishes (regression: kind CHECK dropped the row)", async () => {
  const archive = makeArchive(1, 9);
  const e2ePayload = { ...payload(), kind: "omarchy-mac-e2e", test_id: "smoke", install_path: "encrypted", asahi_image: "Minimal BTRFS", encryption: true, boot_separate: true, overall: "PASS" };
  const init = await initiate(archive, { kind: "omarchy-mac-e2e", payload: e2ePayload });
  expect(init.res.status === 200, `initiate ${init.res.status}: ${JSON.stringify(init.body)}`);
  expect(init.body.status === "awaiting_chunks", JSON.stringify(init.body));
  const up = await uploadChunk(init.sha, 0, archive);
  expect(up.res.status === 200, `chunk ${up.res.status}`);
  const fin = await complete(init.sha);
  expect(fin.res.status === 200 && fin.body.status === "stored", JSON.stringify(fin.body));
  const record = await fetch(`${BASE}/v1/results/${init.sha}`);
  expect(record.status === 200, `record ${record.status}`);
  const doc = await record.json();
  expect(doc.summary.kind === "omarchy-mac-e2e", "kind not stored");
  return `receipt ${init.body.receipt_url}`;
});

scenario("resumed upload sends only missing chunks", async () => {
  const archive = makeArchive(3, 1);
  const first = await initiate(archive);
  expect(first.res.status === 200, "initiate failed");
  // Upload only chunk 1, then re-initiate: server must report [0, 2].
  await uploadChunk(first.sha, 1, archive.subarray(CHUNK, 2 * CHUNK));
  const second = await initiate(archive);
  expect(second.res.status === 200, "re-initiate failed");
  expect(
    JSON.stringify(second.body.missing_chunks) === "[0,2]",
    `expected [0,2], got ${JSON.stringify(second.body.missing_chunks)}`,
  );
  await uploadChunk(first.sha, 0, archive.subarray(0, CHUNK));
  await uploadChunk(first.sha, 2, archive.subarray(2 * CHUNK));
  const fin = await complete(first.sha);
  expect(fin.res.status === 200 && fin.body.status === "stored", JSON.stringify(fin.body));
  return "only chunks [0,2] resent";
});

scenario("replays are duplicates", async () => {
  const archive = makeArchive(1, 2);
  const first = await initiate(archive);
  const up = await uploadChunk(first.sha, 0, archive.subarray(0, archive.length));
  await complete(first.sha);
  const replayInit = await initiate(archive);
  expect(replayInit.body.status === "duplicate", JSON.stringify(replayInit.body));
  const replayComplete = await complete(first.sha);
  expect(replayComplete.body.status === "duplicate", JSON.stringify(replayComplete.body));
  const replayChunk = await uploadChunk(first.sha, 0, archive.subarray(0, CHUNK));
  expect(replayChunk.res.status === 409, `chunk replay ${replayChunk.res.status}`);
  expect(up.body.status === "stored", "first upload not stored");
  return "initiate/complete dedupe, published chunk upload refused";
});

scenario("chunk hash mismatch is refused", async () => {
  const archive = makeArchive(2, 3);
  const init = await initiate(archive);
  const good = archive.subarray(0, CHUNK);
  const corrupt = good.slice();
  corrupt[0] ^= 0xff;
  const bad = await uploadChunk(init.sha, 0, corrupt);
  expect(bad.res.status === 422, `expected 422, got ${bad.res.status}`);
  expect(bad.body.error === "chunk_hash_mismatch", JSON.stringify(bad.body));
  return `error=${bad.body.error}`;
});

scenario("oversize archive is rejected by name", async () => {
  const big = makeArchive(11); // 11 chunks worth, > 8 MB total
  const total = big.length + 1024 * 1024; // force total past the cap
  const res = await fetch(`${BASE}/v1/submit`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      schema_version: 1,
      kind: "deep",
      content_sha256: "a".repeat(64),
      payload: payload(),
      archive: {
        total_bytes: total,
        chunk_bytes: CHUNK,
        chunk_count: Math.ceil(total / CHUNK),
        chunk_sha256: Array.from({ length: Math.ceil(total / CHUNK) }, (_, i) => String(i).padStart(64, "0")),
      },
      pow: { nonce: "1", difficulty: POW_BITS },
    }),
  });
  expect(res.status === 413, `expected 413, got ${res.status}`);
  const body = await res.json();
  expect(body.error === "archive_too_large", JSON.stringify(body));
  return `error=${body.error}`;
});

scenario("oversize single chunk is rejected by name", async () => {
  const archive = makeArchive(2, 4);
  const init = await initiate(archive);
  const res = await fetch(`${BASE}/v1/submit/${init.sha}/chunk/0`, {
    method: "POST",
    headers: { "content-type": "application/octet-stream" },
    body: new Uint8Array(CHUNK + 1),
  });
  expect(res.status === 413, `expected 413, got ${res.status}`);
  const body = await res.json();
  expect(body.error === "chunk_too_large", JSON.stringify(body));
  return `error=${body.error}`;
});

scenario("weak and absent proof of work are rejected", async () => {
  const archive = makeArchive(1, 5);
  const weak = await initiate(archive, { pow: { nonce: "0", difficulty: POW_BITS } });
  expect(weak.res.status === 403 && weak.body.error === "pow_invalid", JSON.stringify(weak.body));
  const absent = await initiate(archive, { pow: undefined });
  expect(absent.res.status === 400 && absent.body.error === "pow_missing", JSON.stringify(absent.body));
  return `weak=${weak.body.error} absent=${absent.body.error}`;
});

scenario("MAC address in payload is refused, nothing stored", async () => {
  const archive = makeArchive(1, 6);
  const sha = sha256Hex(archive);
  const res = await fetch(`${BASE}/v1/submit`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      schema_version: 1,
      kind: "deep",
      content_sha256: sha,
      payload: payload({ model: 'eth0 00:1A:2B:3C:4D:5E' }),
      archive: null,
      pow: { nonce: solvePow(sha), difficulty: POW_BITS },
    }),
  });
  expect(res.status === 422, `expected 422, got ${res.status}`);
  const body = await res.json();
  expect(body.error === "pii_detected", JSON.stringify(body));
  const check = await fetch(`${BASE}/v1/results/${sha}`);
  expect(check.status === 404, "refused payload became readable");
  return `error=${body.error} kinds=${JSON.stringify(body.detail.kinds)}`;
});

scenario("completing with a chunk missing returns 409 and stays invisible", async () => {
  const archive = makeArchive(3, 7);
  const init = await initiate(archive);
  await uploadChunk(init.sha, 0, archive.subarray(0, CHUNK));
  const fin = await complete(init.sha);
  expect(fin.res.status === 409, `expected 409, got ${fin.res.status}`);
  expect(fin.body.error === "incomplete", JSON.stringify(fin.body));
  expect(
    JSON.stringify(fin.body.detail.missing_chunks) === "[1,2]",
    `missing ${JSON.stringify(fin.body.detail.missing_chunks)}`,
  );
  const record = await fetch(`${BASE}/v1/results/${init.sha}`);
  expect(record.status === 404, "incomplete submission visible on read route");
  const probe = await fetch(`${BASE}/v1/submit/${init.sha}`);
  expect(probe.status === 404, "incomplete submission visible on dedup probe");
  return `error=${fin.body.error} missing=${JSON.stringify(fin.body.detail.missing_chunks)}`;
});

scenario("archive round-trips byte-identical", async () => {
  const archive = makeArchive(3, 8);
  const init = await initiate(archive);
  for (let idx = 0; idx < 3; idx++) {
    await uploadChunk(init.sha, idx, archive.subarray(idx * CHUNK, (idx + 1) * CHUNK));
  }
  await complete(init.sha);
  const res = await fetch(`${BASE}/v1/results/${init.sha}/archive`);
  expect(res.status === 200, `archive ${res.status}`);
  const back = new Uint8Array(await res.arrayBuffer());
  expect(back.length === archive.length, `length ${back.length} != ${archive.length}`);
  expect(sha256Hex(back) === sha256Hex(archive), "sha256 mismatch after round-trip");
  return `${back.length} bytes identical`;
});

scenario("/v1/schema reports the worker's schema identity", async () => {
  const res = await fetch(`${BASE}/v1/schema`);
  expect(res.status === 200, `schema ${res.status}`);
  const body = await res.json();
  expect(JSON.stringify(body.schema_versions) === "[1,2]", "schema_versions missing");
  expect(typeof body.fields_sha256 === "string", "fields_sha256 missing");
  expect(typeof body.schema_sha256 === "string", "schema_sha256 missing");
  expect(body.fields_sha256.length === 64, `fields_sha256 length ${body.fields_sha256.length}`);
  expect(body.schema_sha256.length === 64, `schema_sha256 length ${body.schema_sha256.length}`);
  // The smoke fixture payload now carries the fields the schema expects;
  // the live worker reports the same identity. POST 405 on the route too.
  const bad = await fetch(`${BASE}/v1/schema`, { method: "POST" });
  expect(bad.status === 405, `schema POST ${bad.status}`);
  return `fields_sha256=${body.fields_sha256.slice(0, 12)}…`;
});

scenario("cron rebuilds dataset; unpublished rows never appear", async () => {
  const cron = await fetch(`${BASE}/cdn-cgi/local/scheduled`);
  expect(cron.status === 200, `scheduled ${cron.status}`);
  const list = await fetch(`${BASE}/v1/results`);
  expect(list.status === 200, `results ${list.status}`);
  const doc = (await list.json()) as { count: number; results: unknown[] };
  expect(doc.count >= 3, `count ${doc.count}`);
  function isIndexedEntry(r: unknown): r is { content_sha256: string } {
    return typeof r === "object" && r !== null &&
      "content_sha256" in r &&
      typeof (r as { content_sha256: unknown }).content_sha256 === "string";
  }
  expect(
    doc.results.every(isIndexedEntry),
    "entries missing content_sha256",
  );
  // The cached bulk index must NOT carry the per-row detail; readers
  // pull /v1/results/<sha> for the full summary. Stripping at cache
  // rebuild time keeps the JSONL index small as submissions grow.
  expect(
    doc.results.every((r) => isIndexedEntry(r) &&
      !("ane_port_detail" in r)),
    "ane_port_detail leaked into the bulk index",
  );
  const jsonl = await fetch(`${BASE}/v1/dataset/latest.jsonl`);
  expect(jsonl.status === 200, `jsonl ${jsonl.status}`);
  const text = await jsonl.text();
  const lines = text.trim().split("\n");
  expect(lines.length === doc.count, `lines ${lines.length} != count ${doc.count}`);
  for (const line of lines) JSON.parse(line);
  expect(!text.includes("ane_port_detail"),
    "ane_port_detail leaked into /v1/dataset/latest.jsonl");
  expect(!text.includes(makeArchive(3) && "incomplete-never"), "sanity");
  // Per-row record still carries the full payload (ane_port_detail may
  // be null for smoke submissions; the route must serve the field).
  const lastEntry = doc.results[doc.results.length - 1];
  if (!isIndexedEntry(lastEntry)) {
    throw new Error("last entry missing content_sha256");
  }
  const lastSha = lastEntry.content_sha256;
  const one = await fetch(`${BASE}/v1/results/${lastSha}`);
  expect(one.status === 200, `per-row ${one.status}`);
  const oneDoc = await one.json() as { summary: unknown };
  if (!oneDoc.summary || typeof oneDoc.summary !== "object") {
    throw new Error("per-row summary missing");
  }
  expect("ane_port_detail" in oneDoc.summary,
    "per-row summary missing ane_port_detail key");
  return `count=${doc.count}, lines=${lines.length}`;
});

scenario("GC removes incomplete submissions older than the window", async () => {
  const stale = "e".repeat(64);
  const old = Math.floor(Date.now() / 1000) - 8 * 24 * 3600;
  await d1(
    `INSERT OR REPLACE INTO submissions (content_sha256, received_at, updated_at, kind, schema_version, summary, pow_difficulty, published)
     VALUES ('${stale}', ${old}, ${old}, 'deep', 1, '{}', 18, 0)`,
  );
  const cron = await fetch(`${BASE}/cdn-cgi/local/scheduled`);
  expect(cron.status === 200, "scheduled failed");
  const rows = await d1(
    `SELECT COUNT(*) AS n FROM submissions WHERE content_sha256 = '${stale}'`,
  );
  expect(rows[0].results[0].n === 0, "stale row survived GC");
  return "stale row deleted by scheduled handler";
});

async function main() {
  rmSync(STATE, { recursive: true, force: true });
  console.log("== applying migrations (local D1) ==");
  const migrate = spawnSync(WRANGLER, [
    "d1", "migrations", "apply", DB_NAME, "--local", "--persist-to", STATE,
  ], { cwd: SERVICE, encoding: "utf8" });
  if (migrate.status !== 0) {
    console.error(migrate.stdout, migrate.stderr);
    process.exit(1);
  }
  console.log(migrate.stdout.trim().split("\n").slice(-3).join("\n"));

  console.log("== starting wrangler dev (local) ==");
  const dev = spawn(WRANGLER, [
    "dev", "--local", "--port", String(PORT), "--ip", "127.0.0.1",
    "--persist-to", STATE,
  ], { cwd: SERVICE, stdio: ["ignore", "pipe", "pipe"] });
  dev.stderr.on("data", (d) => process.stderr.write(d));

  let ready = false;
  for (let i = 0; i < 120 && !ready; i++) {
    await delay(500);
    try {
      const res = await fetch(`${BASE}/v1/results`);
      ready = res.status === 200;
    } catch { /* not up yet */ }
  }
  if (!ready) {
    console.error("wrangler dev did not become ready");
    dev.kill("SIGTERM");
    process.exit(1);
  }
  console.log("== worker ready ==");

  let failures = 0;
  for (const s of scenarios) {
    try {
      const note = await s.fn();
      console.log(`PASS ${s.name}${note ? ` — ${note}` : ""}`);
    } catch (err) {
      failures++;
      console.log(`FAIL ${s.name} — ${(err as Error).message}`);
    }
  }

  console.log(`\n${scenarios.length - failures}/${scenarios.length} smoke scenarios passed`);
  dev.kill("SIGTERM");
  process.exit(failures === 0 ? 0 : 1);
}

await main();
