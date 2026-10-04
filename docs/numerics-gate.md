# Numerics gate

A decode-route change must pass all three checks against the deployed composed route:

1. **Per-op error:** fp32/fp64 output and state error is no worse than the deployed path for captured operands.
2. **Token choices:** teacher-forced top-1 agreement is at least 99%. At every free-run first divergence, the composed path's top-2 gap must be no more than one bf16 ULP at the magnitude of its top logit. For normal bf16 values, the spacing is 2^(floor(log2(|x|))-7); for example, the ULP is 0.125 at magnitudes in [16, 32), and 0.0625 in [8, 16). The boundary is inclusive.
3. **Perplexity:** route-sensitive S=1 PPL differs by no more than 0.1%.

The free-run condition is evaluated at each prompt's first divergence, not by overall prefix identity. A one-ULP difference can change argmax, then later tokens and hidden states can diverge further; a low full-run identity does not by itself fail this gate. Teacher-forced agreement still has its own 99% minimum.

The former absolute 0.05 near-tie cutoff was not scale-aware. At logit magnitudes [16, 32), one bf16 ULP is 0.125, so a 0.05 threshold rejected even a one-ULP difference. The ULP threshold replaces that absolute cutoff.

For the published v0.7.26 9B fused route, the captured results were: per-op fp64 identical; teacher-forced agreement 99.51% (5,095/5,120); 10/10 free-runs diverged, with six first-divergence gaps of 0.125 (one ULP at the measured logit magnitude) and four gaps of 0; S=1 PPL changed by -0.072%. Prefix identity was 29.43% over 10 prompts × 512 tokens. Under this scale-aware criterion, all three gates pass. The v0.7.27 default remains ON; set MLX_OMARCHY_GDN_RAW_REPEAT=0 to select the composed route.

Analyzer code: tools/dfuse/bf16_ulp.py, free_run_gaps.py, and compare_tf9.py. The gate evidence and historical 0.05 decision are retained in the dated audit receipts.

The independent G13 H257 run measured 99.53% teacher-forced agreement, identical per-op fp64 results, S=1 PPL -0.079%, and 9 first-divergence gaps within one ULP (six at 0.125 and three below); free-run identity was about 39% over 512 tokens. Both chip classes meet the same bar.
