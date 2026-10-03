# Automatic routing ON by default, 2026-10-03 (ships in v0.7.23)

**Decision (owner, 2026-10-03):** automatic decision routing is **ON by
default** for every pair, beginning with v0.7.23, with a documented kill
switch. This closes open-decisions row 1 in `docs/serve.md`.

**Release note (verbatim, for the v0.7.23 changelog):**

> MLX Chat now routes eligible turns to a typed comparison automatically
> ("Which should I take — the tram or the ferry?"). The decision runs on the
> local Laya model inside a 250 ms budget before the chat model answers, and
> ordinary chat never runs the router. Set `MLX_OMARCHY_ROUTING=0` in the
> service environment to turn it off.

## Kill switch and precedence

- `MLX_OMARCHY_ROUTING`: set to `0`, `off`, `false`, or `no` in the
  assistant's service environment to disable automatic routing for that
  process. The variable is read at every `auto` turn submission.
- A saved explicit `gate: "off"` in the pair's selection evidence also
  disables, for deployments that manage the choice per pair.
- Precedence: env kill switch > saved explicit choice > default ON. There is
  no env value that forces routing on past an explicit off.

## What changed (commit `8b1226da6` on main)

- `serve/mlx_omarchy_assistant/coordinator.py`: the routing gate defaults
  ON. Previously `submit(mode="auto")` was refused unless the pair record
  carried routing evidence (gate on + suite sha256 + receipt + policy
  version) — which the production pair manager never exposed, so `auto`
  was unreachable in the shipped app. The policy record
  (`ROUTING_POLICY`, version "3"), its thresholds, the stage order
  (injection guard → structure extractor → deterministic stage → Laya head
  on grey turns), and the 250 ms warm deadline are unchanged.
- `serve/mlx_omarchy_assistant/server.py`: `/api/status` reports
  `routing`: `enabled`, `disabled_reason`, `policy_version`, `head_ready`
  (true only when a pair is resident with a decision worker), `pair_state`,
  and the last head call — `last_head_ms`, `last_route`, `last_reason`,
  `last_timed_out`, `last_at`. `last_head_ms` is null when the last routing
  attempt never reached the head (deterministic or skipped turn).
- Tests: `tests/test_assistant_routing.py` (default-on, kill-switch values,
  saved-off precedence, env beats saved choice, broken-manager status,
  submit rejections, honest-status assertions, degrade paths),
  `tests/test_assistant_routing_suite.py` (the frozen held-out bytes stay
  pinned; automatic mode admitted by default).
- No new files under `serve/`, so the installer file lists are unchanged
  (`tests/test_install_sh_contract.py` passes).

## Degrade paths (unchanged behavior, now on by default)

Routing turns degrade silently to the chat model — never an error, never a
blocked turn — when: the pair is not set up or fails to start
(`pair_start_failed`); the decision worker is starting or unreachable; the
material does not fit the decision model's 512-token limit
(`material_does_not_fit`; nothing is truncated, no option is invented); a
head call misses the 250 ms deadline (`deadline_miss`); a previous head
call is still in flight (`previous_call_pending`); or the head returns
invalid output or misses the thresholds. A routed comparison still runs the
existing fit check before any decision request.

## Memory

The head is the pair's own Laya decision worker, already admitted by memory
admission with the pair — routing adds no resident model. Per-pair peaks
are in `docs/serve.md` (16 GB tier: Compact pair, measured peak
5.2–6.8 GiB; Everyday 9.5 GiB with one borderline 16 GB run). If admission
refuses the pair there is no chat either; routing never tips a pair over.

## Gate evidence the decision rests on

- [Routing gate, 2026-09-30](../2026-09-30-routing-gate/README.md):
  frozen policy 3 scored precision 1.000 (35/35, 0 false positives) and
  0 of 15 injection cases routed to a decision on the single held-out run.
- [Head latency, 2026-10-02/03](../2026-10-02-laya-head-latency/README.md):
  rope tables at engine load took the warm head call from p95 347 ms to
  **196.8 ms** (p50 186.9 ms, 100 warm calls on dev turns, same harness as
  the gate), held-out re-passed once with that head: 35/35, 0/15
  injections, answers bit-identical to the 2026-09-30 run, p95 195.4 ms on
  the 85 fit-check-eligible turns.
- Standing limit (unchanged from that receipt): the measured runs used the
  cap wheel; the rope change on the installed release wheel without the
  cap has not been measured. The held-out suite is spent; a future policy
  change needs a fresh one.

## Verification for this flip (dev box, CPU, 2026-10-03)

- `PYTHONPATH=serve python3 -m unittest discover -s tests -q`: **996 tests,
  0 failures, 27 skipped**, and exactly 15 errors — all
  `ModuleNotFoundError: No module named 'mlx'` from the pre-existing
  environment-dependent modules (`test_bonsai2_loader`,
  `test_bonsai2_packed`, `test_bonsai2_server`, `test_laya_admission_math`,
  `test_laya_reference`), the same set as before the change.
- JS: all 6 files under `tests/js` pass under bun (exit 0):
  assistant-ui, assistant-ui-cards, speak-queue, transfer, util,
  voice-recorder.
- Dev-set replay (no GPU; recorded dev head answers re-scored through this
  tree's production functions with the gate's
  `receipts/2026-09-30-routing-gate/scripts/routing_replay.py`,
  `--raw .../raw/dev-raw.json --suite tests/fixtures/routing_dev.json`):
  head_answers precision 1.0 (tp 56, fp 0, 0 breaches, 1/154 head calls);
  `production_250ms` precision **1.0, recall 0.6548, tp 55, fp 0, 0
  injection breaches, 1/154 head calls** — identical routes to the gate's
  dev baseline. The frozen held-out set was NOT run again.

## Product-path smoke on hardware (M1 Max, T6001)

Host: 16-inch M1 Max (T6001), Omarchy Linux, kernel 7.1.13-3-2-ARCH, boot
`4b353848`, run inside the host's GPU window discipline
(`gpuwin.sh`; the resident inference service was stopped for the window and
restored by the window's own trap). MLX runtime: published v0.7.22 wheel
`0.32.4.dev202610031525+58724762`, provenance `version_match: true`
(`scripts/mlx_provenance.py` against the wheel RECORD; wheel sha256
`bb8e31acd2153161ea74e683777fa6780c95be384a41a7c3760d2c2c3cbf188b`
verified against the release `SHA256SUMS`). Serve packages: this tree at
`1bd12adfb` + `8b1226da6`, copied into the venv. Fresh home
`/var/tmp/routingon-home`; the Laya conversion was pre-staged on CPU
(pinned source `convaiinnovations/laya` at `55cf4c4e`, converter writes
`catalog_id "laya-mlx"`), and the chat model came from the host's HF cache
at the pinned revision `938d8919` (no download).

Flow driven through the product's own HTTP client helper
(`mlx_omarchy_assistant.__main__.request`) against the real server:
setup (`everyday`, downloads approved) → wait ready → ordinary chat →
`mode:"auto"` grey turns (head call measured) → `mode:"auto"` explicit
options turn (routed comparison decided by Laya) → ordinary chat → status.

**Result: pass.** Setup reached `ready` with zero downloads (chat from the
host HF cache at the pinned revision; decision reused the pre-staged
conversion). Five turns, **0 error events, 0 tracebacks** in the server
log:

| Turn | Result |
|---|---|
| ordinary chat | streamed 259 chars, 8.4 s |
| `auto` grey turn (cold) | head called in **198.1 ms** — under the 250 ms deadline; head said `structured_decision` (p 0.6145, act 1.0), the turn carries no criteria, so it became `clarify` (`missing_options_or_criteria`) and the chat model answered |
| `auto` grey turn (warm) | head called in **190.8 ms**, same distribution |
| `auto` explicit options ("Options: tram; ferry. Criteria: lowest cost.") | deterministic `structured_decision` (`explicit_structure`, no head call) → the routed comparison ran on the real Laya worker and a typed decision event was emitted (21.4 s incl. the chat explanation) |
| ordinary chat | 6.0 s |

`/api/status` after the run reported `routing.enabled: true`,
`head_ready: true`, `pair_state: "ready"`, policy "3", and the last
routing attempt (`last_route "structured_decision"`, `last_reason
"explicit_structure"`); `last_head_ms` is honestly null because the last
attempt never reached the head — the two grey turns before it measured
198.1 / 190.8 ms. The warm head number matches the gate expectation
(~200 ms warm) and is inside the 250 ms deadline on this chip.

A first window earlier the same hour behaved correctly on its own and was
superseded when the chat worker failed to start: the venv was missing the
vendored `mlx-lm==0.31.3` pin and the repo's serve patches (both are part
of a real install — the release wheel ships only the MLX runtime).
Installed per the installer's recipe, patches applied cleanly, and the
window was rerun; that first attempt's single grey turn measured
250.4 ms — a `deadline_miss`, the chat model answered, exactly the
documented degrade path for a cold head.

After both windows the resident inference service was verified restored
(`health` 200 plus a real completion probe inside the window's own
restore; re-verified `active` + `health` 200 from outside). Artifacts in
`raw/` (sha256 in `raw/SHA256SUMS`): the smoke client's `result.json`
(full per-turn event streams), the server log, the decision worker log,
and the per-phase TTFT lines. The smoke client and window runner were
syntax-checked (`bash -n`, `py_compile`) before the window, per the
18:20Z lane rule.

## Files

- `raw/result.json` — the smoke client's full per-turn event streams and
  the final `/api/status` routing block. One status string carried the
  host's home path (a theme fallback message); it is redacted to `~`
  before first commit, per the public-repo privacy rule. Everything else
  is verbatim.
- `raw/server.log`, `raw/everyday-decision.log` — server and Laya worker
  logs from the passing window.
- `raw/ttft-phases.jsonl` — per-phase TTFT lines for the five turns.
- `raw/SHA256SUMS` — hashes of all of the above.
