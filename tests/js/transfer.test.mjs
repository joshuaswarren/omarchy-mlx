// Executable assertions for the transfer helper logic. Run directly:
//   bun tests/js/transfer.test.mjs   (or: node tests/js/transfer.test.mjs)

import assert from "node:assert/strict";
import { normalizeLicenses, licenseDetails, missingWheels, totalBytes,
         buildTransferRequest, planRows }
  from "../../serve/mlx_omarchy_assistant/static/js/transfer.js";

// normalizeLicenses: canonical sorted string list.
assert.deepEqual(normalizeLicenses({ licenses: ["apache-2.0", "mit"] }),
                 ["apache-2.0", "mit"]);
assert.deepEqual(normalizeLicenses({ licenses: [] }), []);
// Defensive: object-map form degrades to keys; absent/invalid to [].
assert.deepEqual(normalizeLicenses({ licenses: { mit: {}, "apache-2.0": {} } }),
                 ["mit", "apache-2.0"]);
assert.deepEqual(normalizeLicenses({}), []);
assert.deepEqual(normalizeLicenses({ licenses: "mit" }), []);
assert.deepEqual(normalizeLicenses({ licenses: [42, "mit"] }), ["mit"]);

// licenseDetails: detail map passes through; junk degrades to {}.
assert.deepEqual(licenseDetails({ license_details: { mit: { models: ["m"], bytes: 5 } } }),
                 { mit: { models: ["m"], bytes: 5 } });
assert.deepEqual(licenseDetails({}), {});

// missingWheels: data, not error.
assert.deepEqual(missingWheels({ wheels_complete: false, missing_wheels: ["mlx==0.32.3", "numpy==2.1"] }),
                 ["mlx==0.32.3", "numpy==2.1"]);
assert.deepEqual(missingWheels({ wheels_complete: true, wheels: ["mlx-0.whl"] }), []);
assert.deepEqual(missingWheels({}), []);

// totalBytes: primitive numbers only.
assert.equal(totalBytes({ total_bytes: 123456 }), 123456);
assert.equal(totalBytes({ total_bytes: "123456" }), undefined);
assert.equal(totalBytes({}), undefined);

// buildTransferRequest: only present optional keys, voice always boolean.
assert.deepEqual(buildTransferRequest({ action: "plan", pairIds: ["everyday"], voice: true }), {
  action: "plan", pair_ids: ["everyday"], voice: true,
});
assert.deepEqual(buildTransferRequest({ action: "inspect", bundle: "/tmp/b" }), {
  action: "inspect", bundle: "/tmp/b", voice: false,
});
assert.deepEqual(buildTransferRequest({
  action: "prepare", output: "/out", pairIds: ["everyday"], voice: false,
  approvedLicenses: ["mit"], wheelCaches: [],
}), {
  action: "prepare", output: "/out", pair_ids: ["everyday"], voice: false,
  approved_licenses: ["mit"], wheel_caches: [],
});
assert.deepEqual(buildTransferRequest({ action: "install", bundle: "/b", pairIds: [] }), {
  action: "install", bundle: "/b", voice: false,
});

// planRows: names every section the reviewer needs.
const plan = planRows({
  pairs: [{ pair_id: "everyday", arch: "arm64", licenses: ["apache-2.0"], file_count: 3, bytes: 2048 }],
  voice: [{ pack_id: "tts-0.6b", license: "cc-by", bytes: 512 }],
  wheels: ["mlx-0.whl", "numpy-0.whl"],
  wheels_complete: true,
  missing_wheels: [],
  total_bytes: 4096,
  requirements: { mlx: "0.32.3" },
});
const rowLabels = plan.map(([k]) => k);
assert.ok(rowLabels.includes("Pair everyday"));
assert.ok(rowLabels.includes("Voice pack tts-0.6b"));
assert.ok(rowLabels.includes("Cached wheels (2 present in wheelhouse)"));
assert.ok(rowLabels.includes("Total transfer size"));
assert.ok(rowLabels.includes("Requirements"));
const missingPlan = planRows({ wheels_complete: false, missing_wheels: ["mlx==0.32.3"] });
assert.deepEqual(missingPlan.find(([k]) => k === "Missing wheels (need download)")[1], "mlx==0.32.3");

console.log("transfer js tests passed");
