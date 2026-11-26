# Serve integration map — conflict resolution for Main

Prepared by ModelCatalogQualification, 2026-09-20. Covers the serve-lane
branches; no dirty-main edits are proposed or included.

## Branch inventory (verified heads at mapping time)

| branch | head | content |
| --- | --- | --- |
| agent/model-catalog-qualification | `a304efcf` | full 10-entry seed catalog, refresher + CI workflow, contract doc, 38-test suite; already merged schema-owner lineage through `fabfb6be` |
| feat/serve-cli-catalog | `bff20ef4`+ (`d02ff391` atomic API, `665ca04d` resident floors) | serve package (catalog.py/budget.py/__main__.py), install.sh wiring, their test suites incl. hermetic bootstrap fixes |
| agent/bonsai2-serving | `69c85977` | mlx_omarchy_bonsai2 module (packed.py/loader.py/server.py), frozen window criteria, parity probe + receipt |
| agent/laya-serving | `a98a7078` + UNCOMMITTED guard | mlx_omarchy_laya module; uncommitted: relabel floor+workspace ≤ admitted guard + drift test (one RED test to fix first, see below) |
| agent/distill-moe-enablement | `905ff807` | device qualification plan doc only; artifact is LOCAL-ONLY (no catalog entry; HF publication not authorized) |
| agent/serve-docs-integration | `7b9e6f09` | docs/serve.md + README serve section (recommendation policy, catalog summary) |
| origin/main | `5b183060`+ | carries `receipts/2026-09-20-qwen38-text-install/` (the Qwen3.8 qualification receipt my catalog references) and the user's earlier docs/demo updates |

## Conflict points and resolutions

1. `serve/mlx_omarchy_serve/catalog.json` — add/add between the full seed
   (my branch) and the schema owner's bundled single entry. RESOLVED:
   full seed wins (Main directive, validator-PASS proven); already merged
   in my branch at `497c2765`. Bundled fallback = the same file; no embed.
2. `feat/serve-cli-catalog` advanced past `fabfb6be` (atomic budget
   `d02ff391`, resident floors `665ca04d`, bootstrap fixes `bff20ef4`).
   My branch does NOT contain those yet. Resolution: merge the current
   feat/serve-cli-catalog tip into the integration; expected conflicts
   none on my files (their changes are package .py + their tests +
   install.sh; install.sh already merged cleanly once).
3. `tests/test_serve_bootstrap.py` — 3 tests failed against my seed
   pre-`bff20ef4` (non-hermetic env). Their fix is on their branch; after
   the merge BOTH suites must run green in one tree — that is the merge
   gate, not a conflict.
4. `docs/serve.md` — rewritten on agent/serve-docs-integration;
   origin/main carries an older rewrite of the same file. Prefer
   serve-docs-integration (`7b9e6f09`, newest, already reflects the
   recommendation policy + my catalog summary).
5. `demo/chat.py` / `tests/test_demo_chat_model.py` — dirty in the MAIN
   WORKING TREE (user edits) and rewritten on origin/main `5b183060`.
   NOT touched by any lane branch. Resolution: leave the user's dirty
   state alone; the serve package (mlx-omarchy-serve CLI) is the
   catalog-driven path, the demo chat remains user-owned with no gate
   wiring required from this integration.
6. `receipts/2026-09-20-qwen38-text-install/` — exists ONLY on
   origin/main. My catalog's qualified claim references it. SEQUENCE
   REQUIREMENT: the integration must include origin/main (or that receipt
   path) or the qualified claim and its receipt never coexist in one tree.
7. `.github/workflows/serve-catalog-refresh.yml` + `tools/refresh_serve_catalog.py`
   + `tests/test_serve_catalog_refresh.py` — unique to my branch; the
   workflow depends on the serve package (validator import) and the tests
   import their budget module, so it merges AFTER the serve package.

## Recommended merge order

1. Base: `agent/model-catalog-qualification` (already contains the schema
   owner through `fabfb6be` and the full seed).
2. Merge `feat/serve-cli-catalog` tip (atomic budget, floors, bootstrap
   fixes).
3. Merge `agent/bonsai2-serving` `69c85977` (catalog id
   `bonsai-2-27b-mlx-2bit` already in the seed; serve.module points at
   their `mlx_omarchy_bonsai2.server`).
4. Merge `agent/laya-serving` — ONLY after the owner lands the uncommitted
   relabel guard (see Blocking item) and `tests/test_laya_admission_math.py`
   runs green.
5. Merge `origin/main` (Qwen3.8 receipt + user docs; expect the
   docs/serve.md content to be superseded by step 6).
6. Merge `agent/serve-docs-integration` `7b9e6f09`.
7. Merge `agent/distill-moe-enablement` `905ff807` (plan doc; no catalog
   entry — local artifact, publication unauthorized).

## Merge gates (one tree, all green)

- `tests/test_serve_catalog_refresh.py` (38, mine)
- schema owner suites: test_serve_catalog / test_serve_budget /
  test_serve_cli / test_serve_bootstrap (their counts, incl. the 3
  hermetic fixes)
- bonsai2 + laya CPU suites (46 + 28 with fixtures)
- `python3 tools/refresh_serve_catalog.py --check` exit 0 (live pins)
- lint per repo standard

## Blocking item (before step 4)

agent/laya-serving's uncommitted relabel guard is correct in shape
(`relabel_admissible = floor + workspace <= admitted`, drift test covers
model-size drift with actual parameter bytes) but its test
`test_floor_is_parameter_bytes_not_allocator_global` is RED: expected
842,609,210 (fp16 FILE bytes incl. safetensors padding) while the new
param_elements-preferred floor correctly yields 842,587,660
(421,293,830 elements x 2). The elements-based floor is the honest
resident figure (padding is not resident); the test constant needs the
update, then the file commits.

## Non-goals

- No edits to the user's dirty main-tree files.
- No HF publication of the Distill artifact (unauthorized).
- No qualification/recommended changes from tooling — receipts only.
