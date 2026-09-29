# Routing gate receipt (RoutingGate, 2026-09-29)

Status at end of session: **NOT EVALUATED — routing stays OFF**.

The held-out suite at `tests/fixtures/routing_held_out.json`
(sha256 `09a37b60…8906f0`) was frozen, never scored before, and
**remains unscored** after this session. The dev-set sweep on a real
Laya worker also failed to produce measurements: an early run hit a
PYTHONPATH bug and all 154 calls timed out with "Connection refused".

## What shipped (commits `1fbfd682e`, `8d702eb84`)

- `serve/mlx_omarchy_assistant/routing.py`: versioned
  `ROUTING_POLICY(version="1", p_min=0.55, margin_min=0.20,
  act_min=0.55)`. NOTE: Main's audit found these thresholds were
  NOT preceded by measured dev-set tuning; they were defaults.
  The notebook has a dated CORRECTION addendum (`1fbfd682e`
  superseded). A real sweep is needed before any freeze holds.
  `evaluate_route()`, `pending_outcome()`, `fit_route_question()`
  with the same budget fit-check `coordinator._typed_request` uses;
  never truncates, never invents options.
- `serve/mlx_omarchy_assistant/coordinator.py`: `submit()` accepts
  `mode="auto"` and refuses it unless the pair's selection evidence
  records `gate == "on"` with a `suite_sha256`, `receipt`, and
  matching `policy_version`. `mode="auto"` resolves BEFORE the GPU
  is acquired; ordinary chat latency is unaffected. A timed-out
  call stays accounted for in `pending_outcome().handle`. When the
  router picks `structured_decision` but the user did not supply
  explicit options, the coordinator downgrades to `clarify`.
- `tests/fixtures/routing_dev.json`: 154 cases, six required
  categories, disjoint from held-out (verified at write time).
- `tests/test_assistant_routing.py`: 29 focused tests with a fake
  Laya worker (deadline, timeout accounting, invalid output,
  over-budget material, threshold logic, flag off/on, ordinary chat
  unaffected). All 399 assistant tests pass; held-out suite test
  still green.
- `receipts/2026-09-30-routing-gate/routing-evaluate.py`,
  `run_routing_eval.sh` (older) pinned on main for transparency;
  the new resumable pipeline at `<home>/src/routing_pipeline.sh`
  + `<home>/src/dev_sweep_run.py` + `<home>/src/dev_sweep_sweep.py`
  is queued and resumable on the M2.

## Acceptance vs. threshold (honest)

| Item | Threshold | Result |
|---|---|---|
| Frozen-policy commit hash | recorded before evaluation | superseded (not real) |
| Held-out suite sha256 pinned in notebook | yes | yes |
| Dev-set sweep on real Laya | required by Main | **ran with PYTHONPATH bug; 0 of 154 valid** |
| Threshold sweep table | required | **none** |
| Held-out precision on routed decisions | >= 99% | **not measured** |
| Coverage published | yes | **not measured** |
| Abstention rate published | yes | **not measured** |
| Per-category breakdown | yes | **not measured** |
| Injection cases never produce structured_decision | yes | not measured |
| Warm p95 latency over 100 calls | <= 250 ms | **not measured** |
| Tests passing | yes | 399/399 assistant + 31/31 routing |
| Pushed to main | yes | `8d702eb84` |

## Why not measured

(1) Initial run under raw `flock` waited ~7 minutes for the lock,
then never produced measurements. (2) First `gpu-turn` invocation
ran with a script bug (`PYTHONPATH=serve` missing in the worker
boot line) — every dev call hit "Connection refused" and was
recorded as `timed_out`. (3) Second `gpu-turn` invocation timed out
at 900 s waiting in queue behind three flock holders (ModelBench,
two SpeechOutputFast-style long holders, and two gpu-turn waiters
in front of me). (4) `gpu-turn` enforces a first-come-first-served
queue so jumping ahead is impossible, and the wrapper kills a job
at 15 min — any hold long enough to do the sweep will be honoured
in turn.

The pipeline is now **resumable**: each per-case record is written
atomically and `sync` runs after the phase finishes. A reboot or
kill resumes from the last completed id (run
`gpu-turn -m 15 -- bash routing_pipeline.sh dev` and the runner
will skip already-done cases). The previous truncated raw file was
emptied for a clean start.

## What to do next

1. Wait for the GPU queue. Run
   `gpu-turn -m 15 -- bash <home>/routing_pipeline.sh dev`
   on the M2. Output lands at
   `<notebook>/RoutingGate/routing-gate/dev-raw-latest.json` and
   `dev-sweep-latest.json`.
2. If `dev-sweep-latest.json` winner has precision >= 0.99 with zero
   injection→decision cases: bump `routing.ROUTING_POLICY` to
   `version="2"`, freeze the threshold, commit, then run the same
   pipeline with `held` (the held-out suite is wired by
   `<home>/src/routing_held_out.json`).
3. If the dev sweep reports no passing cell, do NOT freeze: report
   the best achievable precision/coverage and what fails (per
   category) in this receipt; the gate stays OFF; consider policy
   "2" with a reworded question only after recording which category
   fails and why.
4. Held-out scoring happens **once**. On PASS, flip the pair
   record's `extension.selection_evidence.routing.gate = "on"` with
   the suite sha256 and this receipt path. On FAIL, the suite is
   spent and a fresh frozen suite (>= 100 cases) must be authored.