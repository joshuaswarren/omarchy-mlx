# 2026-09-29 jwm1 (M1, T8103): fused gated RMSNorm made bit-exact vs the composed chain and routed by default

Problem (receipts/2026-09-28-jwm1-gdn-qk-scaled.md): `mx.fast.rms_norm_gated` flipped greedy tokens versus mlx-lm's composed `_precise_swiglu(rms_norm)` chain,
so its 90-dispatch/token fusion could not ship. First fix (4244f5454) removed intermediate bf16 rounding (36% -> ~7e-6 of elements). This change removes the rest.

Root cause (measured, jwm1 M1, Honeykrisp): NOT the sigmoid. The fused kernel's exp(-gate) and sigmoid equal `mx.exp` / `mx.sigmoid` bit for bit for every finite bf16 gate
(both are inexact relative to float64 in ~64% of the mid range, identically). The two real differences were compiler reassociation of fmul chains:
1. `silu * normed` (source `(gate*sigmoid)*normed`) was compiled in another order; fixed with `precise` on both products.
2. `value * norm * weight`: the reference FastRmsNormBF16 compiles it as `value * (norm * weight)`; the fused kernel's compiled order differed. Explicit variants over 5.43M elements,
   fused normed vs `mx.fast.rms_norm`: (value*norm)*weight 25 mismatches, (value*weight)*norm 14, value*(norm*weight) 0. Now precise `value*(norm*weight)`.
Also: `RMSNormGated` dispatches a gated-only kernel build (no runtime mode branch; measured not to matter for exactness, kept because it is the tested configuration);
RMSNormScaled keeps the two-mode kernel.

Exactness evidence on the shipped build (wheel from this branch, jwm1): `scripts/check-rms-norm-gated-exact.py` max mismatch rate 0.0 (gated shapes 16x128, 1x2048, 2x2048, 7x128, 32x128, 64x128, 256x128;
rms_norm_scaled 16-256 rows); exhaustive sweep over all 65,280 finite bf16 gate values x 100 random (h, w) trials: 0 mismatches in every |z| range including the denormal and overflow regions.

Performance/pins (interleaved A/B vs the live stack, which has q/k + conv patches; candidate = same venv + this wheel + the gated patch, routing at size <= 2048 i.e. decode rows):
decode64 40.70 -> 41.66 tok/s (1.024, n=15), decode128 39.90 -> 40.58 (1.017, n=10); ordered_records_sha256 identical in every round (7fe6badf4d560e25 / da5568eeb4b6a1c1).
prefill512 digest identical (ccb601895581d89f). The candidate wheel is main-tip plus this change, so its prefill (247 -> 255-257 tok/s) also includes two earlier main commits
(conv_dw1d.comp, standalone silu chain) that were not in the previously deployed wheel; that gain is NOT attributable to this change.

Requirement: `patches/mlx-lm-qwen35-gated-norm.patch` routes to `rms_norm_gated` guarded only by `hasattr`; it must be paired with a wheel containing this kernel (older wheels' fused kernel is not bit-exact).
Notebook entries (private): jwm1-parity H12, H18, H19-H24.
