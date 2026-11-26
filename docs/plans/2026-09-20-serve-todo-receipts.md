# Serve TODO integration receipts — periodic refresh / manual HF / approve / pick

Per-TODO evidence for the four pending integration items, with concrete
commands, outputs, and paths. Status as of 2026-09-20, MCQ lane.

## 1. Periodic refresh (scheduled workflow)

- Workflow: `.github/workflows/serve-catalog-refresh.yml`
  (weekdays 04:43 UTC + workflow_dispatch; suite gate under the stdlib
  unittest runner — no pytest install needed).
- Live verification (real HF API, this tree):
  `python3 tools/refresh_serve_catalog.py --catalog serve/mlx_omarchy_serve/catalog.json`
  → exit 0, "all availability sizes already current; catalog not written"
  (zero-noise verified; sizes exact at all 10 pinned revisions, zero drift).
- Contract tests: `tests/test_serve_catalog_refresh.py` — 43 tests green
  under `python3 -m unittest` (refresher core: availability-only writes,
  drift retains vetted fields, abort-without-write, zero-noise no-op;
  publish flow: 4 local-git-fixture cases incl. the second-run and
  stale-base cases; workflow contract: unittest runner, PR-only, no
  push-to-main, drift-exit handling).
- Publish flow script: `tools/serve_catalog_publish.sh` (branch from
  current HEAD, --force-with-lease, explicit PR query then create-or-fail;
  gh failures loud).

STATUS: ready; blocked from landing only by the serve-cli route fixes.

## 2. Manual HF (maintainer metadata authoring)

- Same tool, maintainer-run: provenance chain verified against real data —
  convaiinnovations/laya @ 1c5edc17 ships 2,262.7 MiB across 38 files;
  `extension.availability_files` restricts the refresh to the served root
  variant (live-verified: laya resolves 842,609,210 B, not the repo sum).
- All 10 entry pins verified against the API this session: every revision
  exists upstream, zero drift, sizes exact.

STATUS: ready; route fix (serve-through-conversion) pending from the
schema owner before the laya entry's unified-serve path is claimed.

## 3. Approve-first downloads

- Catalog-side contract enforced and tested: the refresher structurally
  cannot write qualification/recommended/revision/priority (pre-write
  guard + drift-retention tests); recommended=false on all 10 entries;
  enablement audited (simulated post-HTTP qwen3.8 state validates).
- CLI-side approve-first gating is ServeCatalogImplementation's lane
  (their bootstrap/hermetic tests); currently blocked per Main — no
  eligible recommended chat entry exists to approve.

STATUS: catalog side ready; CLI side pending route fix + an eligible entry.

## 4. Pick (recommendation chooser)

- Real CLI run on the integrated tree
  (`PYTHONPATH=serve python -m mlx_omarchy_serve recommend --kind chat`):
  `pick: none — no entry passes the auto-serve gate` — the honest state;
  no chooser prompt is offered for unqualified entries.
- Per-entry memory estimates in the same output match the independent
  formula recomputation byte-exact (qwen3.8-27b-4bit 34.69 GiB =
  16,054,541,349 weights + 17,179,869,184 KV + 4,013,635,337 workspace).
- Enablement audit: simulating qwen3.8-27b-4bit post-HTTP
  (http=qualified + recommended=true) validates — the eventual flip is
  one commit, no missing fields.

STATUS: working and honest; task completes when qwen3.8-27b-4bit earns a
real HTTP receipt and the owner flips recommended.
