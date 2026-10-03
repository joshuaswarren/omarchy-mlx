# SDPA backward SIGABRT root-cause and fix (SdpaVjpFix, 2026-10-03)

## Defect

`omarchy_fast_ops_tests` case "scaled_dot_product_attention backward
matches finite differences" (B=2, H=1, qL=2, D=4, f32, `vjp` over the
fused forward) aborted with
`SmallVector<int, 10>::operator[] assertion 'size() > index' failed`
(mlx/small_vector.h:315) on every asserts-enabled build; NDEBUG builds
passed silently (the defect was a silent out-of-bounds `Shape` write).

## Root cause

`ScaledDotProductAttentionVJP::eval_gpu` (the FUSED VJP primitive, not
the composed graph - the rep=1 f32 route serves the fused path) built
the dK/dV tile shapes by indexing a fixed 5-D layout:

```cpp
Shape tile_shape = q5.shape();   // rep=1: 4-D {B,H,qL,D}
tile_shape[3] = kL;
tile_shape[4] = D;               // index 4 into a 4-element vector
```

At rep=1 the GQA head split is a no-op, so `q5` is 4-D and
`tile_shape[4]` writes past the end. In NDEBUG builds the write lands
in unused inline storage and the still-correct 4-D shapes flow on -
which is why every Release battery (M2 G14X, jw16 G13C, dev-box
llvmpipe) stayed green since the kernels landed on 2026-10-01.

## Fix

`fb0aac16c` - derive the tile shapes from the score plane's rank:

```cpp
Shape tile_shape = S.shape();
tile_shape.back() = D;   // dkt
tile_shape.back() = Dv;  // dvt
```

One rank-agnostic construction for both the 4-D (rep=1) and 5-D (GQA)
routes; no guard, no special case. Sibling audit: the only other
index-4 shape write in the backend (`regroup_view`, forward path)
constructs `Strides(5)` explicitly and runs only at rep>1.

## Evidence

Dev box (llvmpipe, asserts-enabled Debug):

- Pre-fix: doctest single case SIGABRT (gdb bt names eval_gpu,
  primitives.cpp:13673, index=4); standalone repro with the same
  vjp call SIGABRT (exit 134). The standalone repro passing outside
  the test binary (NormApple observation) was reproduced as an
  artifact of linking the NDEBUG library - not test pollution.
- Post-fix: case 17/17; standalone exit 0; full suite green except
  pre-existing rope deltas (below).

jw16 G13C (M1 Max) hardware, one gpuwin window, asserts-enabled build
of eaf0fb5a4 + fb0aac16c:

| suite | result |
|---|---|
| omarchy_fast_ops_tests | 40/40 SUCCESS (12 failed assertions = the documented may_fail composed-dk legs, unchanged) |
| omarchy_primitive_tests | 104/104 |
| omarchy_runtime_tests | 41/41 |
| omarchy_fused_chain_tests | 36/36 |
| omarchy_compiled_tape_tests | 13/13 |
| omarchy_matmul_family_tests | 22/22 (82.9M assertions) |
| omarchy_gdn_fast_route_repeat_tests | 3/3 (NormApple blocker green) |

llm-inference restored after the window (health_ok=1,
probe_finish=length, active).

## New coverage and new defects

The fd sweep harness (`tests: fd sweep ...`, 02df7071f) pins dq/dk/dv
against host central differences at rep=1 neighbors: B=1/2, qL=1/2/5,
kL=1/2/5/7, D=4/8/64, causal and maskless. It surfaced two value
defects in the fused VJP, now may_fail doctests with signatures
recorded in docs/known-defects.md:

1. qL=1 with kL>1 maskless: dk/dv all zero (llvmpipe AND jw16 M1 Max
   hardware; forward and dq correct). qL=1 is decode geometry, so a
   fine-tune backward through a single-query step silently loses
   dk/dv today.
2. B=1 with kL=5: exact-zero/uniform dk/dv spots (llvmpipe).

Standalone probes clear matmul-with-inner-dim-1 and the {kL,1}
transposed-view General copies; the defects live in the eval_gpu
composition. Next lever: instrument s_t_dense/p_t_dense and dkt/dvt
inside eval_gpu at (1,1,2,4).

## Attributions (other lanes)

- The 3 rope bf16 doctests red on dev-box llvmpipe (fused-vs-composed
  1-ulp deltas) were introduced by 6cbf55d8f (trig Cody-Waite range
  reduction): e292f45ac passes, 6cbf55d8f^ passes, f751646ac fails,
  MLX_OMARCHY_NORM_APPLE=0 makes no difference. Green on jw16
  hardware (NormApple 10/11 run). TrigContract's lane.
- rope_rms_norm bit-exact doctest fails standalone on llvmpipe but
  passes in-suite (order-dependent) - NormApple lane, dev box only.
- The composed-chain dk defect (may_fail legs) is broader than the
  three documented shapes: the same element signature reproduces at
  more rep=1 shapes on llvmpipe; documented in docs/compatibility.md.
  The crash fix and the may_fail legs do NOT share a cause (fused VJP
  tile-shape write vs composed-chain value defect), so the legs stay
  as documented per the assignment.

## Not done this lane

- Root-causing the two new fused VJP value defects (open entry).
- LoRA smoke on the M2 (probe at /var/tmp/vjp-lora-probe.py on the
  M2): skipped - M2 is booked by w73 and was not available this turn;
  the earlier lane already recorded the Q4-tile LoRA blocker
  separately.
- jw16 rerun with the final sweep commit (02df7071f): the jw16 battery
  ran fb0aac16c (the fix); the sweep's strict legs are
  dev-box-validated and hardware-consistent; the may_fail legs are
  the documentation.
