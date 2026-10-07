# Fused SDPA VJP gated off: dk/dv outputs never receive the reduce result

Date: 2026-10-07. Lane: w6Z (UpstreamMlx). Host: `<m1max-host>` (Apple
M1 Max, T6001, G13C), Linux, Vulkan device Honeykrisp, static test
configure of origin/main 067e8ce26 + this change. Probes are
reproducible standalone (`/tmp` binaries, links below are procedure
only — rebuild from the recipes).

## Defect

`ScaledDotProductAttentionVJP::eval_gpu` (the fused training backward,
rep=1 float route) returns wrong dk/dv: dk's first rows carry **dq's
exact bytes**, the remaining rows are zeros (or the allocator's
recycled-storage poison word 123456789.0 with `MLX_OMARCHY_POISON_FREED=1`
armed); dv is all zeros. The 26 `may_fail` assertions in
`sdpa vjp fd parity at small rep=1 shapes (known defects)` are this
defect, previously mislabeled as a composed-path dk defect.

## Isolation evidence (same session, same build)

Probes used the exact doctest seeds (0x510+qL*7+D / 0x520+kL*5+D /
0x530+kL*3+D) at B=1, H=1, qL=1, kL=2, D=4, f32, scale 0.5, and a
`vjp` with a full-shaped cotangent.

1. Every internal plane verified CORRECT at its stage boundary via a
   temporary `MLX_OMARCHY_SDPA_VJP_DUMP` probe (removed before landing):
   lse = 1.1767, delta (odo) = 1.2798, dP, dS (in-place S), P, both
   transposed dense copies, q5, k5 — all match host math.
2. The dq leg is CORRECT (dq5 = dS·K verified), so the ds pass and the
   first matmul work.
3. The two transposed-LHS matmuls (`s_t_dense·q5 → dkt`,
   `p_t_dense·co5 → dvt`) produce poison/zeros inside the primitive,
   while the byte-identical standalone matmul outside the primitive is
   correct (probed with mlx ops and with raw `copy_gpu_inplace`).
4. `dk[0:4]` equals dq's words EXACTLY — the dk output storage aliases
   the dq temporary; dk.set_data's buffer never receives the reduce
   result. This is an output-buffer handling defect inside the
   primitive, the same class as the 2026-10-06 fence-gated recycling
   NaN fix (6a925715), not a math-kernel defect.
5. Ruled out: cooperative-matrix k-tail (adding a
   `matrix_k >= 8 && %8==0` guard changed nothing), gated barriers
   (`MLX_OMARCHY_GATED_BARRIERS=0`), dep reflection
   (`MLX_OMARCHY_DEP_RW=0`), each stage bypassed individually
   (odo/ds/scores/dq/tiles/transposed-copies/reduce), the forward's
   coopmat route (`MLX_OMARCHY_NO_COOPMAT=1`).
6. The composed fallback matches host finite differences at every
   probed rep=1 shape on the same run (1x1x2x4, 1x2x5x4, causal
   1x3x5x8 — three-way with the exact doctest seeds).

## Change

`ScaledDotProductAttentionVJP::use_fallback` returns true (fused VJP
gated off; every training backward takes the composed fallback, which
the strict fd doctest `sdpa vjp fd parity across degenerate rep=1
shapes (dq dk dv)` already covers). The forward's `use_fallback` also
returns true so the single-output composed shape is kept (the lse
output existed only for the fused VJP). `MLX_OMARCHY_NO_FUSED_VJP=1`
stays as the explicit kill switch documentation.

## Verification on the build host (jw16, this branch)

- Standalone vjp probe with the fused route off: dk/dv match host
  three-way (values above in the notebook entry).
- `omarchy_fast_ops_tests` full suite: expected green EXCEPT the
  may_fail case, which now measures the composed path at 5x7/4x4/6x9
  (its own documented uncertainty, kept may_fail until that path is
  fd-swept separately).
- Battery suites re-run after the flip: see the battery receipt rows
  updated in receipts/2026-10-06-mlx-backports/battery.md.

## Follow-ups (open, owned)

1. Root-cause the primitive's output-buffer aliasing (next lever:
   dump `binding(out)` buffer addresses for dq/dk/dv and the
   `set_data` allocations inside `dispatch_matmul` vs the primitive's
   own allocations; suspect the eval_gpu-allocated dkt/dvt buffers
   abandoned when dispatch_matmul re-allocates `out`).
2. After the fix: re-enable the fused VJP, flip both fd doctests to
   hard, and re-gate on a three-way hardware proof.
3. Sweep the composed path at 5x7/4x4/6x9 causal on M1 Max hardware to
   retire (or confirm) the 2026-10-03 composed-dk entry.
