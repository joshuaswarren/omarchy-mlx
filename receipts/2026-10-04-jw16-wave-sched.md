# Wave scheduling (MLX_OMARCHY_WAVE_SCHED) — barrier census, bit-exactness, and jw16 A/B

Date: 2026-10-04. Lane: BarrierSched. Branch: `agent/wave-sched` (f0f806120 +
5a0819ae8). Private-notebook artifacts: `BarrierSched/{census1,abA,abB,abC,ab4b*,census4b}`
(SHA256SUMS at each level; raw per-cell JSONs retained).

## What landed

`MLX_OMARCHY_WAVE_SCHED=1` (default OFF) buffers the open batch's dispatch/copy/fill
nodes and records them at submit in greedy earliest-wave order: one full dependency
barrier per wave, tape order preserved within a wave. Hazards are RAW/WAW/WAR over the
exact tracked byte ranges with the SPIR-V reflected read/write split, so every tape
dependency stays ordered and outputs are unchanged by construction. The schedule
function `wave_levels` is unit-tested for the contract (no intra-wave hazard pair,
every hazard edge crosses waves, levels are the earliest legal wave).
`omarchy_runtime_tests` 42/42 on the dev box and, with the gate ON, on the M1 Max.
Requires `MLX_OMARCHY_GATED_BARRIERS` (default); inert without it.

Also on the branch: `barriers_emitted`/`barriers_skipped` added to the
`mlx_omarchy_trace_snapshot` C ABI (census without a diagnostics build) and
`MLX_OMARCHY_WAVE_DIAG` (per-flush hazard-class histogram, counts only).

## Numbers (M1 Max, Omarchy Linux, Honeykrisp fork driver, boot-fixed host, gpuwin windows)

Decode, qwen3-2B hybrid (d64->d512 slope, serving-parity env):

| arm | dispatches/tok | barriers/tok | skipped/tok |
|---|---|---|---|
| wave off | 229.21 | 223.08 | 20.13 |
| wave on  | 229.21 | 223.03 | 20.19 |

Wave-diag census (d512, 1358 flushes, 130,588 nodes): waves/nodes = 0.904; forcing-edge
class none=9,822 **RAW=120,766 WAW=0 WAR=0**. The deployed 2B decode graph is a
0.90-density RAW chain: after the fused-GEMV/gated-norm/fused-GDN work there is no
intra-layer independence left to hoist, so reordering cannot cut decode barriers on
this model. Prefill: -8.4% hardware barriers per pf512 pass, but wall moved only
+0.82% (pf512, disjoint, n=5) and -0.09% (pf1024, n=5) — prefill barrier cost hides
behind GPU-busy execution.

qwen3-4B dense (36 layers, per-head q/k norms and rope NOT folded — that independence
is exactly what the scheduler can hoist): barriers/tok 370.2 -> 334.3 (**-9.7%**).

A/B (>= 5 alternating ctl/cand rounds per cell, same wheel both arms, greedy digest
asserted identical every round):

- 2B: d64 +0.09% / d128 -0.16% / d256 -0.25% / d512 -0.17% (sub-noise), pf512 +0.82%
  disjoint / pf1024 -0.09%.
- 4B d64: **+3.2%, +3.4% medians in two windows, min-max disjoint in both**
  (ctl 63.18-63.97 vs cand 64.50-66.10; ctl 63.29-63.62 vs cand 65.68-65.83 tok/s).
- 4B d512: window 1 +2.9% median with one interference outlier (-9.7% pair);
  window 2 **+3.0% median, disjoint** (ctl 55.84-55.97 vs cand 57.54-57.60).
- 4B digests == the production pins (`e2c919be...` d64, `fff6d03b...` d512) in both
  arms and both windows: bit-exact against the deployed stack, not just self-consistent.

## Gates run

omarchy suites with `MLX_OMARCHY_WAVE_SCHED=1` on the M1 Max:
`omarchy_runtime_tests` 42/42 (22,731 assertions), `omarchy_primitive_tests` 104/104
(2,743,003), `omarchy_fused_chain_tests` 36/36 (346,272), `omarchy_matmul_family_tests`
23/23 (82,941,807), `omarchy_capability_sim_tests` 7/7 on all five profiles
(m1-honeykrisp-fork, m1-stock-no-coopmat, subgroup-size-64, small-shared-memory,
no-cooperative-matrix). llm-inference restored (health 200, finish_reason=length)
after every window.

## Landing decision

The win is 4B-specific (+3.2-3.4% d64 / +2.8-3.0% d512, two windows, disjoint,
digest-pinned); the 2B is flat-to-slightly-negative, so the default stays OFF and
4B serving hosts opt in with `MLX_OMARCHY_WAVE_SCHED=1`. A default-ON flip needs the
2B regression addressed (or a 2B win) first. The reorder lever for the 2B decode is
closed with a measured structure: the remaining lever on that axis is fewer RAW edges
(more fusion), not their order.

## Confounder on record

Main tip at e6b1de3aa changes 2B greedy ids at d128+ relative to the previous serving
wheel (b581d5c-era pins: d128 a9a7eef8 -> now 21691e38; d64 cb3e8770 unchanged); 4B ids
are unchanged. Within-lane bit-exactness is wave-on == wave-off on the same wheel,
which held on every cell.
