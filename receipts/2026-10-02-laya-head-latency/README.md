# Laya head latency: 347 ms -> 197 ms p95, decisions bit-identical, 2026-10-02/03

**Result:** the warm Laya head call now meets the 250 ms deadline. On the
same harness as the 2026-09-30 gate (5 warm-ups + 100 timed calls on dev
turns), warm **p95 is 196.8 ms** (p50 186.9 ms, max 200.2 ms, 0 failures),
down from 347 ms. The frozen held-out suite ran **once** with the declared
candidate and passed: precision **35/35**, **0 of 15** injection cases routed
to a decision, policy 3 unchanged. Its raw head answers are bit-identical to
the 2026-09-30 run on all 100 held-out cases. Over the 85 held-out turns
that pass the fit check (the only ones production can send to the head),
head **p95 is 195.4 ms** (max 198.8 ms).

Automatic routing stays **off** by default; turning it on is the owner's
call now that the head call meets the 250 ms criterion.

| Criterion | Before (2026-09-30) | After (this receipt) |
|---|---|---|
| Warm head p95, 100 calls on dev turns | 347 ms | **196.8 ms** |
| Held-out precision of `structured_decision` | 35/35 | **35/35** |
| Injection cases routed to a decision | 0/15 | **0/15** |
| Raw head answers vs 2026-09-30, held-out | - | **100/100 bit-identical** |
| Raw head answers vs shipped head, dev | - | **154/154 bit-identical** |

## What was slow, and the fix

`_encoder_attention` rebuilt the rotary cos/sin table on every one of the
28 encoder layers. On this backend each `cos`/`sin` evaluation passes a
trig-argument accuracy gate that reads the argument magnitude back to the
host, which forces a GPU join. A GPU profile of the head forward (the
diagnostics wheel `0.32.3.dev202610022039+diag.83eb57a`, 14 forwards)
recorded **785 joins, 784 of them `trig_argument_gate`**: exactly two per
layer per call, with 3.14 s of host wait in total. The GPU was busy only
61 % of the span; the rest was the stall at each gate and the resubmission
behind it.

The fix (`serve/mlx_omarchy_laya/model.py`, `rope_tables`): the float32
cos/sin tables for positions 0..`max_len`-1 are built once when the engine
loads, one pair per rope theta. Each layer gathers the rows it needs and
casts them to the activation dtype. Row *t* of the table comes from the same
elementwise operations on the same inputs as the old per-length
computation, so the values are bit-identical, and a request never evaluates
a trig function. `tests/test_laya_unit.py::RopeTableTests` pins that
equality; it runs where mlx is installed.

## Measurements

Host: M2 Max (T6021); provenance `verified: match` beside every run; each
GPU run held its own fair-queue turn. Per-call records carry boot id,
uptime, load average and CPU PSI; every number below comes from runs with
load < 0.5 and PSI cpu `some avg10 = 0`, more than 6 minutes after boot.

### Decomposition (shipped head, before the fix)

In-process phases, n = 40 dev turns (91-112 tokens), on the 2026-09-30 gate
wheel `0.32.3.dev202609291615+06711ad`:

| Phase | p50 ms | p95 ms |
|---|---|---|
| tokenize | 0.7 | 1.1 |
| collate + array build | 0.2 | 0.2 |
| forward record (host) | 15.9 | 26.0 |
| **mx.eval (execute + sync)** | **278.4** | **314.4** |
| readback + post-processing | 1.1 | 2.1 |
| wall | 295.9 | 341.2 |

HTTP against the resident worker added a 2.4 ms p50 residual. The cost was
flat over sequence length (91-163 tokens) and over call order.

### Levers tested

All A/B arms ran same-boot, alternating, with the wheel stamp asserted equal
across arms and full 154-case dev sweeps for decision equality.

| Lever | Result (wall p50 / p95 ms) | Decisions |
|---|---|---|
| `mx.compile` on the forward | eager 265.5 / 290.9 vs compiled 268.6 / 295.1 (eval ms) | equal |
| `MLX_OMARCHY_BATCH_WORK` (cap wheel `0.32.4.dev202610020734+4b2929a`) | 0: 271.0 / 303.4; 500: 315.9-323.7 / 327.6-332.7; 10000: 257.7 / 272.3; 20000: 260.5 / 281.5; **40000: 254.6-257.8 / 270.0-272.8**; 80000: 255.3 / 277.7 | 0 bit diffs |
| worker dtype, cap 40000 | float16 255.4 / 275.0; bfloat16 249.1-253.2 / 255.2-276.0; float32 250.2-252.2 / 280.4-285.5 | 0 route diffs (probabilities differ) |
| **rope tables at load, cap 40000** | **187.3 / 195.2 and 187.3 / 196.6** vs base 253.9-254.7 / 272.1-273.0 | **0 bit diffs** |

Compilation and dtype do not touch the cost because neither removes the
trig-gate joins. dtype is also flat for a second reason: the residual stream
promotes to float32 after layer 0 whatever the worker dtype (the profile
shows MatmulF32Coopmat at 45 % of GPU time with float16 weights). That
promotion is unchanged here because changing it changes the numbers.

### Final run (the single held-out use)

Declared configuration: rope-table serve tree, float16, cap wheel
`0.32.4.dev202610020734+4b2929a` at `MLX_OMARCHY_BATCH_WORK=40000` (the
default baked on main since the submission-cap work). Same harness as
2026-09-30: `dev_sweep_run.py` raw answers (deadline 2.0 s), then
`routing_latency.py`, then no-GPU scoring with the production functions in
`routing_replay.py`.

| Latency suite (100 warm calls) | p50 ms | p95 ms | max ms |
|---|---|---|---|
| dev turns (the gate measurement) | 186.9 | **196.8** | 200.2 |
| held-out, fit-check-eligible turns (85) | 184.5 | **195.4** | 198.8 |
| held-out, all 100 turns | 186.1 | 379.5 | 392.4 |

The all-turns held-out tail is the 15 `oversized` cases (p50 378.7 ms at
the 512-token cap). The fit check refuses every one of them before any head
call, so production never sends them to the head.

## What remains

- The measured candidate ran the cap wheel from a private venv. The rope
  change alone, on the installed release wheel without the cap, was not
  measured. The cap default (40000) is already on main, so the next release
  wheel carries both changes.
- The float32 promotion after layer 0 (see above) costs casts and float32
  matmuls on every call. Removing it changes the head's numbers, so it needs
  its own decision-equality gate before anyone ships it.
- Correction to the harness: until this receipt, the `other_python_mlx_procs`
  field written by `head_probe.py` counted the probe's own processes instead
  of foreign ones (an inverted condition, now fixed). Foreign-process
  absence for the runs above rests on gpu-turn's exclusive lock, the
  recorded load and PSI, and end-of-run process snapshots that show only
  the probe's own worker.

## Tests

`PYTHONPATH=serve python3 -m unittest discover -s tests -q` on the dev box:
982 tests, 0 failures, the 15 pre-existing laya/bonsai2 environment errors,
26 skipped (`RopeTableTests` skips there because the dev box has no mlx).
`RopeTableTests` passed on an M1 Max (T6001) with the Omarchy GPU backend
(wheel `0.32.3+5b18306`, provenance `verified: match`, tree `5b8b6c586`):
table rows are bitwise equal to the per-length formula for both rope
thetas at T = 1, 96, 163 and 512.

## Files

- `raw/`: probe NDJSON, dev and held-out raw head answers, scored held-out
  rows, latency runs (boot ids truncated to 8 hex characters).
- `scripts/`: `head_probe.py` (resumable per-call NDJSON probe: phases and
  http modes, answer-equality gate, host-state records), `run_ab.sh`
  (alternating same-boot arms; per-arm env, `LAYA_DTYPE`, `LAYA_SERVE`),
  `run_ticket*.sh` / `run_stage2_*.sh` (gpu-turn ticket pipelines),
  `run_final.sh` (the held-out run), `analyze.py` (percentiles, host-validity
  gating, route equality), and copies of the gate's `dev_sweep_run.py` and
  `routing_latency.py`. The 8 MB GPU profile and its analyzer output are in
  the private lab notebook, not in this public repository.
