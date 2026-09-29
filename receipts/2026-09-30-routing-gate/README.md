# Routing gate receipt (RoutingGate, 2026-09-29)

Status at end of session: **NOT EVALUATED — routing stays OFF**.

The held-out suite at `tests/fixtures/routing_held_out.json` (sha256
`09a37b60…8906f0`) was frozen, never scored before, and **remains
unscored** after this session. The policy "1" was committed and
frozen on dev evidence; the lock contention on jw14m2-linux in this
window did not allow a clean evaluation of 100 Laya calls under the
shared GPU.

## What shipped (commit `1fbfd682e`)

- `serve/mlx_omarchy_assistant/routing.py`: versioned
  `ROUTING_POLICY(version="1", p_min=0.55, margin_min=0.20, act_min=0.55)`;
  `evaluate_route()`, `pending_outcome()`, `fit_route_question()` with the
  same budget-fit check `coordinator._typed_request` uses; never
  truncates, never invents options.
- `serve/mlx_omarchy_assistant/coordinator.py`: `submit()` accepts
  `mode="auto"` and refuses it unless the pair's selection evidence
  records `gate == "on"` with a `suite_sha256`, `receipt`, and
  matching `policy_version`. `mode="auto"` resolves BEFORE the GPU
  is acquired; ordinary chat latency is unaffected (a cold Laya load
  takes longer than the 250 ms warm deadline and therefore bypasses
  routing immediately). A timed-out call stays accounted for in
  the runner's pending list — no replacement is launched. When the
  router picks `structured_decision` but the user did not supply
  explicit options, the coordinator downgrades to `clarify` — never
  invents alternatives.
- `tests/fixtures/routing_dev.json`: 154 cases across the six required
  categories, disjoint in wording from the held-out suite.
- `tests/test_assistant_routing.py`: 29 focused tests (deadline,
  timeout accounting, invalid output, over-budget material, threshold
  logic, flag off/on, ordinary chat unaffected). All 399 assistant
  tests pass; the existing held-out suite test still passes.
- `tests/test_assistant_routing_suite.py` (existing): unchanged. The
  suite sha256 still matches and `submit(mode="auto")` still raises
  ValueError when the gate is off.

## Acceptance vs. threshold (honest)

| Item | Threshold | Result |
|---|---|---|
| Frozen-policy commit hash | recorded before evaluation | `1fbfd682e` |
| Held-out suite sha256 pinned in notebook | yes | yes |
| Held-out precision on routed decisions | >= 99% | **not measured** |
| Coverage published | yes | **not measured** |
| Abstention rate published | yes | **not measured** |
| Per-category breakdown | yes | **not measured** |
| Injection cases never produce structured_decision | yes | not measured |
| Warm p95 latency over 100 calls | <= 250 ms | **not measured** |
| Tests passing | yes | yes (399/399) |
| Pushed to main | yes | **not yet** |

## Why not measured

Three other workers (ModelBench, SpeechOutputFast, SpeechInputGpu) held
`flock /tmp/m2-gpu.lock` continuously during the window the lock was
free in the first attempt; by the time I retried, ModelBench held it
again. Each Laya warm call is ~5 ms but starting the worker alone
takes ~10 s of model load, all of which has to happen under the same
lock. This workstream is single-shot: I did not start the worker
because I would not have been able to keep the lock for the full
startup + 100 calls + teardown inside one 15-minute window without
interrupting another agent's run. Honest state: **the held-out suite
is not yet evaluated, and routing stays OFF**.

## What to do next

1. Re-check `flock /tmp/m2-gpu.lock` holders, wait for a free window.
2. `bash <home>/src/run_routing_eval.sh` (already on the M2 at
   `<home>/src/run_routing_eval.sh` and `<home>/src/routing-evaluate.py`).
   The script:
   - boots a Laya worker under the lock on port 50500
   - waits for `/health`
   - runs `routing-evaluate.py` against
     `tests/fixtures/routing_held_out.json` once
   - writes the JSON receipt under
     `~/.local/share/apple-silicon-lab/artifacts/RoutingGate/routing-gate/eval-*.json`
3. Append the receipt to this README; flip the pair record's
   `extension.selection_evidence.routing.gate = "on"` (with the
   suite sha256 and receipt path) only on PASS (precision >= 99%).
4. Update `docs/serve.md` and push to main.