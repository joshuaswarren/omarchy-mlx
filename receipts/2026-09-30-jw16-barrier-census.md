# 2026-09-30 — jw16 (M1 Max) decode: dependency-barrier census on the deployed stack — the tracker is RAW-tight; the remaining lever is emission order

Lane: Jw16BarrierElide. Host jw16 (T6001/G13X), boot 36539f5a. Serving stack UNTOUCHED:
venv /var/tmp/v072-venv-fused (0.32.3.dev202609292324+82f2f482f), system ICD 9d949d4-vec2
(0x17f barrier word + submit poll + dependency-tracked per-launch CDM barrier 5a520cf3d35).
Census on a diag wheel = serving source + env-gated diagnostics only
(0.32.3.dev202609302210+diag.enc.dag3, branch `agent/jw16-barrier-census`, extends H157's
`agent/barrier-reason` commits; every window via gpuwin.sh, restore health 200 +
finish=length; decode digest 07c515e0 = the production d128 pin in EVERY instrumented run).
Notebook: apple-silicon-lab entries/Jw16BarrierElide/20260930T215000Z (pre-registered),
artifacts/Jw16BarrierElide/w1-census/ with SHA256SUMS.

## Question

Which of the barriers the MLX gated tracker still emits per decode token are avoidable —
emitted with no real RAW/WAR/WAW at hardware-visible granularity? Candidate classes from
the lane assignment: read-read pairs through unknown access masks, range over-approximation,
allocator-recycled WAR/WAW. The driver side: the installed ICD emits the hardware CDM
barrier ONLY on an app-declared vkCmdPipelineBarrier (5a520cf3d35; MLX's 1.0 calls reach it
through vk_common_CmdPipelineBarrier's 2.0 conversion) — so every MLX-emitted dependency
barrier is a hardware drain and every skipped one leaves the launch unordered.

## Method

d32/d64/d128 decode cells (limit 1, warmup 1, 5 passes, prefill 512) with
MLX_OMARCHY_BARRIER_REASON=1 (RAW/WAW/WAR counters in batch_needs_barrier), plus one d128
with MLX_OMARCHY_DAG_DUMP=1 + MLX_OMARCHY_BARRIER_PAIRS (per-node read/write ranges +
encoder ids + colliding pairs). Offline replay (be1_dag_analyze2.py) reproduces the
production tracker decision-for-decision. Per-token numbers from the d64->d128 slope (576
tokens, warmup+prefill subtracted). No timing claims from profiled runs (the profiler
inserts an execution barrier per dispatch, which 5a520cf3d35 turns into a hardware barrier).

## Census result (deployed stack)

| measure | value |
|---|---|
| node decisions/token (dispatch+copy+fill) | ~250 (slope), 267.5 (dag run) |
| emitted dependency barriers/token | 226.1 (84.5%) |
| skipped (proved disjoint) /token | 41.4 (15.5%) |
| emitted split | **RAW 173,532 / head 78 / WAW 0 / WAR 0** |
| replay vs runtime | **0 mismatches over 205,410 decisions** |
| collider distance | 99.8% immediate predecessor; dist>=2 = 236 (joins) |
| post-submit heads | 78 per 768 tokens (~0.10/token; ordered by the in-order completion semaphore) |
| totals d32/d64/d128 | raw 50,003 / 90,963 / 172,883; **waw=0, war=0 in every run** |

Cross-checks: the reason totals reproduce exactly across runs with/without the dag dump;
emitted(raw+heads) equals the SUBMIT-ENTER count (111) in the traced run.

## Findings

1. **The tracker layer is tight: every emitted barrier is a true RAW hit.** WAW and WAR are
   zero across d32/d64/d128 on jw16, matching jwm1's H157. The assignment's avoidable
   classes are all empty: (A) read-read via unknown masks — none (rw10 SPIR-V reflection
   covers the decode kernels; read-read pairs skip — e.g. the parallel-GEMV group at dag
   nodes 2836-2838 carries only the submit-head, and joins pay one barrier for all edges);
   (B) range over-approximation — none (exact byte ranges, no WHOLE_SIZE); (C) recycled
   WAR/WAW — none. Aliasing buffers, in-place updates (GDN state), and cross-queue hazards
   are all correctly forced.
2. **Drain arithmetic (chain-slope attribution, NOT wall-clock):** 226.1 emitted/token x
   12.2 us (DecodeGap3's per-dependent-dispatch slope) ~= 2.76 ms/token of the 9.86 ms/token
   wall (101.4 tok/s d128 unprofiled control in the same lane). Drains hide behind 10-40 us
   kernels; only the un-hidden part is real win.
3. **CORRECTED 2026-10-01 (Jw16LevelBatch): ORDER is NOT a lever.** An earlier version of this
   paragraph claimed greedy level batching needs 41.8 levels/token vs 226.1 emitted barriers
   (ceiling ~2.25 ms/token). That number was a packing count: `level_batch` in
   `be1_dag_analyze2.py` places a node in the first level whose members do not conflict with it,
   without requiring it to come after its producers' levels, so it is not a schedule. The true
   longest-path depth is ~267 levels per decode token (frontier algorithm == brute-force O(N^2)
   reference on the last-token slice of the d128 DAG dump, 0 mismatches over 321 nodes),
   accumulating ~270 levels/token across a batch. The recorded order (~273 barriers/token) is
   within ~15% of optimal. A full implementation (`MLX_OMARCHY_LEVEL_BATCH`, branch
   agent/jw16-level-batch f44a35aab) was bit-exact (all decode pins, pf512 records, logits gate)
   and SLOWER by 8.5-9.2% (d64-d512), -8.1% on pf512 ttft. The remaining dispatch-boundary lever
   is node-count reduction (fusion). See receipts of Jw16LevelBatch in the private notebook.

## Per-lever ledger for dispatch-boundary cost on G13X (all closed with receipts)

| lever | receipt | verdict |
|---|---|---|
| CDM barrier-word trim | DecodeGap3 (2026-09-29) | closed: 0x17f minimal correct word; 0x170 corrupts at d256+ |
| defer flush to end-of-batch | MesaLaunchCost W1 (2026-09-30) | closed: -3.9% AND corrupt |
| coherent loads/stores | Jw16MacParity 0240Z (2026-09-29) | closed: corrupt + 27% slower |
| tracker-layer elision (masks/ranges/recycling) | **this census** | closed: RAW-only, 0 avoidable |
| cheaper in-queue RAW semantics | DecodeGap3 finding 1 | firmware semantics, out of driver reach |
| level-batched emission (order) | Jw16LevelBatch (2026-10-01) | closed: true depth ~267 levels/token (census 41.8 was a packing count); implemented bit-exact but -8.5..-9.2% decode |

## Reproduce

jw16: branch `agent/jw16-barrier-census` (wheel via `DIAG=1 bash
/var/tmp/appbar/build_wheel_venv.sh <src> <tag> <venv> /var/tmp/v072-venv-fused`),
env `MLX_OMARCHY_BARRIER_REASON=1 MLX_OMARCHY_DAG_DUMP=1 MLX_OMARCHY_BARRIER_PAIRS=20000`
inside `gpuwin.sh`, analyze with be1_dag_analyze2.py (artifact copy in the notebook).
Live serving stack and ICD untouched; llm-inference restored after every window.
