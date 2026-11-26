# 2026-09-28 jwm1 (M1, T8103): GDN q/k rms_norm_scaled routing shipped; gated-norm fusion rejected (diverges)

Host: Apple M1 Linux (Honeykrisp), installed wheel 0.32.3.dev202609260251+42fbbc5, mlx-lm 0.31.3, model
Qwen3.8-2B-mlx-4Bit @0867d98b, greedy temp 0, protocol `benchmarks/qwen38-mlx-bench.py --limit 5 --prefill-tokens 0
--warmup 2 --passes 1`, one flock hold per A/B, arms interleaved. Private notebook entries (not in this repo):
jwm1-parity 20260928T224800Z (H3), 20260928T225100Z (H4).

Context: the paired macOS legs (ane-linux-experiments receipts/2026-09-28-jwm1-macos-parity-legs) show Linux decode at
0.76-0.80x of macOS. A gated-barrier A/B (env MLX_OMARCHY_GATED_BARRIERS=1) was 1.002x with identical digests: barriers
are not the bottleneck (the decode chain is mostly truly dependent; 18% of barriers skipped). The census shows 513 dispatches/token.

| arm | decode64 tok/s median (n=15) | vs ctl | ordered_records_sha256 |
|---|---:|---:|---|
| installed venv (ctl) | 38.98 | 1.000 | 7fe6badf4d560e25 |
| q/k `rms_norm_scaled` only | 39.81 | 1.021 | 7fe6badf4d560e25 (identical) |
| gated `rms_norm_gated` only | 39.72 | 1.019 | 22eadc46db9f3340 (DIFFERS) |
| both (existing patch script) | 40.51 vs ctl 39.20 | 1.033 | 22eadc46db9f3340 (DIFFERS) |

The gated arm flips greedy tokens on 3 of 5 prompts (first divergences at generated tokens 14, 24, 19), so the
previously receipted "bit-exact by construction" claim for `FastNormGatedBF16` gated mode does not hold on the real model
(consistent with the older "exp tie-rounding" note). It is NOT shipped. The q/k site is bit-identical.

Shipped-path validation (`patches/mlx-lm-qwen35-qk-scaled.patch` applied to a copy of the installed venv; files byte-identical
to the validated q/k-only venv; reapply detected as already applied): decode128 ctl 38.34 vs 39.12 tok/s (1.020x), decode256
ctl 37.38 vs 38.05 (1.018x), digests identical (128: da5568eeb4b6a1c1, 256: 7d0523ae795bf83f). Routing is bf16, <=256 rows,
guarded on `hasattr(mx.fast, "rms_norm_scaled")`, so prefill and other models are untouched.

Load caveat: the H4 run drifted to loadavg 0.77 (own ssh polling) in later rounds; arms were interleaved.

Open: root-cause `FastNormGatedBF16` gated mode (would add a further ~1.2% and 90 dispatches/token); remaining gap after this
change is still ~0.78x decode.

## Addendum (same day): gated-norm root cause and residual

Root cause of the gated divergence, measured at tensor level (`scripts/check-rms-norm-gated-exact.py`, on jwm1): `fast_norm_gated.comp`
mode 0 rounded `sigmoid` and `silu` through bf16, but mlx-lm 0.31.3 `_precise_swiglu` (and the primitive's own C++ fallback)
keep the chain in f32 and round once. Composed == all-f32 reference (0 mismatches); the old fused kernel == the
bf16-rounded reference exactly (14828 / 14650 / 6452 / 235674 mismatching elements at 16x128 / 1x2048 / 7x128 / 256x128 = ~36%).

Fix (this commit): drop the two intermediate roundings. Result on a wheel built from it (35bc8bc75): mismatch vs composed falls to
~1e-5 of elements (2.4e-5 max over the checked shapes; 0 at several). The norm stage alone is exact over 9.8M elements
(weighted and weightless), so the remaining deviation is in the sigmoid/multiply epilogue, most likely f32 `exp`/reciprocal
codegen differing between this kernel and the FusedChainF32 interpreter (not confirmed; needs ISA-level inspection).

That residual is still enough to flip greedy tokens over a real run: A/B with the gated site routed at size <= 2048 (decode shapes
only) on top of the shipped q/k patch: decode64 40.59 vs 39.92 tok/s (1.017x) and decode128 39.83 vs 38.87 (1.025x), but digests
DIFFER from control (6a5841bd1c474703 vs 7fe6badf4d560e25; 7577500edb264fa4 vs da5568eeb4b6a1c1). Per the exactness policy the
gated site is NOT shipped: no installer patch is added and `rms_norm_gated` stays unrouted by default. The check script exits 1
on a systematic deviation (rate > 1e-4), which the pre-fix wheel fails (0.36) and this build passes.
