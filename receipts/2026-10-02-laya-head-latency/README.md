# Laya head latency: decomposition and floor, 2026-10-02/03

**Result:** the warm head call cannot meet the 250 ms deadline on this stack
today. The best measured configuration reaches warm p95 **270-282 ms**
(p50 255-261) across five independent same-boot arm runs, with decisions
**bit-identical** to the shipped head on the full 154-case routing dev set.
Automatic routing stays **off**; the owner's latency definition (the head
call itself) is unmet for a measured, named reason.

## What was measured

Host: M2 Max (T6021), one boot throughout (`df82d24a…`), timing windows with
`loadavg` < 0.5 and PSI cpu `some avg10 = 0` recorded per run; provenance
`verified: match` beside every run. Suites: `tests/fixtures/routing_dev.json`
(154 cases) through the 2026-09-30 gate harness shape (5 warm-ups, direct
HTTP POST to the resident Laya worker, float16, B=1). The 2026-09-30 gate
wheel (`0.32.3.dev202609291615+06711ad`) was still the installed runtime, so
the baseline reproduces the gate configuration exactly.

### Phase decomposition (in-process, n = 40 dev turns, seq 91-112 tokens)

| Phase | p50 ms | p95 ms |
|---|---|---|
| tokenize | 0.7 | 1.1 |
| collate | 0.1 | 0.1 |
| array build | 0.05 | 0.1 |
| forward record (host dispatch) | 15.9 | 26.0 |
| **mx.eval (execute + sync)** | **278.4** | **314.4** |
| readback | 0.9 | 1.8 |
| post-processing | 0.2 | 0.3 |
| wall | 295.9 | 341.2 |

HTTP adds little against a resident worker: wall p50 276.6 / p95 310.9 ms
with server `predicted_ms` 274.2/306.3 and a 2.4/3.7 ms HTTP+JSON residual
(n = 40). Host-side preparation is not the problem; the GPU execution of the
graph is.

The execution time is a **fixed cost**: flat over call order (283.5 ms
first-10 vs 287.3 ms last-10 mean) and flat over sequence length (273-317 ms
across 91, 96, 112, 117, 163-token sequences, three independent probes).

### Lever: mx.compile (falsified)

Compiling the forward (`mx.compile` over `encoder_forward` + head) leaves the
floor in place: eager eval p50 265.5 ms vs compiled p50 268.6 ms (n = 12/30,
same text, same process conditions). Answers stayed exactly equal to the
shipped engine. Elementwise fusion does not reduce this cost, so the floor is
not fusable-chain overhead.

### Lever: submission cap (real but insufficient; decisions bit-identical)

The submission-cap wheel (`0.32.4.dev202610020734+4b2929a`, the 2026-10-02
`MLX_OMARCHY_BATCH_WORK` implementation, run from a private venv — the shared
venv was not modified) was A/B'd same-boot, same wheel stamp, env-only arms,
alternating rounds, n = 25-30 per arm:

| `MLX_OMARCHY_BATCH_WORK` | wall p50 ms | wall p95 ms |
|---|---|---|
| 0 (off) | 271.0 | 303.4 |
| 500 | 315.9-323.7 | 327.6-332.7 |
| 10000 | 257.7 | 272.3 |
| 20000 | 260.5 | 281.5 |
| **40000 (the shipping default)** | **254.6-257.8** | **270.0-272.8** |
| 80000 | 255.3 | 277.7 |

The cap at >= 10000 reliably saves ~15-25 ms; aggressive splitting (500)
costs ~50 ms. Correctness on the full dev set: **0 route differences and 0
bit differences** between cap 0, 500, 40000 arms and the shipped gate-wheel
head (154/154 cases each comparison). Compiled + cap40000 combined shows no
additional gain (in-process eval p50 253.8 vs 262.0 ms eager, within
cross-process noise).

## Floor analysis

Warm head-call wall = ~2.5 ms HTTP+JSON + ~1 ms host prep + ~16 ms host
dispatch record + **~250-270 ms GPU execute+sync** + ~1 ms readback/post.
The execute+sync term:

- does not scale with sequence length (91-163 tokens),
- is not reduced by kernel fusion (mx.compile),
- shrinks only ~20 ms with the submission cap, with a plateau above 10k
  work-group budgets and a penalty below that,
- reproduces at 254-314 ms across two different wheels, two Python
  environments, and in-process vs HTTP paths.

Working inference (not separately instrumented): the cost is per-dispatch
work in the Omarchy Vulkan submit path — roughly 700 eager dispatches for
this encoder at ~350-500 us each — which no serve-side change can remove.
The levers that would move it are backend-level (per-dispatch submit cost,
or a fused-graph/ANE path), outside this receipt's scope.

**Consequence for routing:** the head-free path (p95 43 ms, gate receipt)
passes the deadline; the head call does not (floor ~270 ms). With the owner's
decision that the gate measures the head call, automatic routing stays off.
The cap-default wheel already on main brings head p95 to ~270 ms on its next
install; it does not close the gap.

## Tests

`PYTHONPATH=serve python3 -m unittest discover -s tests -q` at this receipt's
commit: 961 tests, 1 failure
(`test_serve_reservations.ConcurrentSetReservationTests.
test_concurrent_set_reservation_atomic`, fails identically on pristine
origin/main, pre-existing), 15 pre-existing laya/bonsai2 environment errors,
25 skipped. This receipt adds no code; no behavior changed.

## Files

- `raw/`: ticket NDJSON/JSON pulled from the measurement host (boot ids
  truncated to 8 hex chars in public copies; full ids in the private lab
  notebook).
- `scripts/`: `head_probe.py` (resumable NDJSON probe: phases/http modes,
  answer-equality gate, host-state records), `run_ticket1.sh`-`run_ticket4.sh`
  (gpu-turn ticket pipelines), `run_ab.sh` (alternating same-boot arms),
  `run_final.sh` (unused: the held-out single run is reserved for a declared
  winner, and none was declared), `analyze.py` (percentiles, host-validity
  gating, route-equality comparison).
