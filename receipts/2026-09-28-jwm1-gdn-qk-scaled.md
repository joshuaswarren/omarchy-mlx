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
