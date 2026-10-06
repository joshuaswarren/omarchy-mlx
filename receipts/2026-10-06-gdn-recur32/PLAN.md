# 2026-10-06 — GdnRecur32: macOS-shape per-token GDN prefill for G13G (PLAN + receipt)

Branch `agent/GdnRecur32` @ <sha>. Lane: GdnRecur32 (worker, dev box — no
hardware). Owner for the runs below: w71 on **jwm1** (G13G, T8103). Everything
here is default-OFF; enabling is `MLX_OMARCHY_GDN_RECUR32=1`.

## What landed

- `shaders/gated_delta_prefill_recur32.comp`: one 32-lane subgroup per
  (hv, dv) row, Dk/32 = 4 f32 state elements per lane in registers,
  sequential token loop exactly like macOS `gated_delta_step`
  (`state *= g; kv = subgroupAdd(sum state*k); delta = (v - kv) * beta;
  state += k*delta; o = subgroupAdd(sum state*q)`; lane 0 stores bf16 y
  RNE). 128-thread WGs = 4 rows; grid (Hv, Dv/4); no shared memory, no
  barriers, no coopmat. h0/hf round-trip supported; scalar-g maskless
  square-bf16 shape only (anything else keeps existing routes).
- New enum `GatedDeltaPrefillRecur32BF16`, pipeline row, shader build row,
  route gate in `primitives.cpp` (`MLX_OMARCHY_GDN_RECUR32=1 && T >= 64 &&
  maskless && g.ndim()==3 && subgroup_size==32`). Env is read per call so
  the test battery can toggle it in-process. Unset env: all existing routes
  byte-identical (pure insertion before the coopmat dispatch).
- `test_gdn_maskless_correctness.cpp` + test case "GDN recur32 per-token
  route matches fp64 reference": same fp64 sweep (T 63/64/65/96/352/512/519
  x rep 1-3 + non-zero-h0 T 512/519) under the env, tolerances y 0.02 abs /
  state 2e-4 abs. This is the numerics gate; NOT a hash gate.
- Host model `hostmodel_recur32.py` (this dir): numpy fp32 model of the
  exact kernel op order (per-lane sequential 4-term partials, pairwise tree
  over 32 lanes, bf16 RNE y). ALL PASS on the full sweep: y_err <= 9.7e-4
  (20x under tolerance), state_err <= 3.8e-8 (5000x under).

## Pre-registered run sheet for w71 (jwm1)

Build the branch, install into the lane venv, then:

1. Numerics gate (must pass before any timing counts):
   `ctest -R omarchy_gdn_maskless_correctness_tests` (or run the binary).
   All cases green, including the recur32 case. Falsifier: any FAIL, any
   NaN in state/y, or fixture-tolerance failures in the default-route cases
   (would mean env leakage — the guard restores the env; investigate before
   proceeding).
2. Kernel-only A/B, 3 gated repeats each, order-balanced (ABBA), load1
   < 0.1 (gdn_micro2.py from artifacts/jwm1-parity/h292):
   - ship: `env -u MLX_OMARCHY_GDN_RECUR32 -u MLX_OMARCHY_GDN_HOIST $V gdn_micro2.py`
   - recur32: `MLX_OMARCHY_GDN_RECUR32=1 env -u MLX_OMARCHY_GDN_HOIST $V gdn_micro2.py`
   - both-on: `MLX_OMARCHY_GDN_RECUR32=1 MLX_OMARCHY_GDN_HOIST=1 $V gdn_micro2.py`
   (hoist must NOT apply when recur32 is selected; both-on is a no-op
   control — same numbers as recur32 alone).
3. e2e: pf512 qwen3.8-2B protocol, same three arms, 3 reps.

## Predictions (falsifiable)

- P1 kernel-only: recur32 removes the per-chunk barrier/RT chain that costs
  jwm1 ~123 us per 64-token chunk (54.9 us state walk + 18.3 + 20.6 us
  LOOPK/LOOPS per stage-stub evidence). If the macOS 6.4 us/token regime is
  reachable on G13G, T=512 goes 6.97 -> ~3.3 ms/call (chain 125 -> ~60 ms),
  pf512 e2e -8..-10%. Realistic floor given G13G's 8 cores and f32 state
  traffic: 2-3x. FALSIFIER: recur32 within noise of ship (|delta| < 3%)
  means G13G is latency-bound elsewhere (state load/store or k/q fetch);
  record and stop.
- P2 T=63 behaves identically with env on/off (route keeps the scan below
  64). Falsifier: any T=63 delta.
- P3 The both-on arm equals recur32-alone within noise. Falsifier: any
  difference > noise — hoist interception leaked.
- P4 Maskless-route dispatch count at T>=512: exactly ONE GDN prefill
  dispatch per call under recur32 (profiler NDJSON), vs 1 (coopmat) or 2
  (hoist) under ship. Falsifier: 0 or >1.

## Kill switch

`MLX_OMARCHY_GDN_RECUR32=0` or unset restores the shipped route selection
exactly (per-call env read; no static). No default flips without this
plan's P1 landing >= 1.5x kernel-only on jwm1 AND the numerics gate green.

## Verified on the dev box (no GPU)

- `glslc -O --target-env=vulkan1.3` clean; `spirv-val` clean; SPIRV
  disassembly shows 2 OpGroupNonUniformFAdd, zero barriers, zero
  workgroup-storage variables.
- `ninja CMakeFiles/mlx.dir/mlx/backend/omarchy/{compute,primitives}.cpp.o`
  clean (fresh configure, MLX_BUILD_OMARCHY=ON; embeds the new shader).
- `ninja omarchy_gdn_maskless_correctness_tests` linked clean (tests skip
  without a Vulkan device on this box).
- Host model: see above. NOT verifiable here: on-device numerics, occupancy,
  timing — that is the run sheet above.
