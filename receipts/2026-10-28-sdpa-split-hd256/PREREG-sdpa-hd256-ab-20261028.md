# Pre-registration: split-K decode attention for head width 256 (written 2026-10-28 16:10Z, before any run of the new kernel on a device)

Change under test: branch `w7q/sdpa-split-hd256` of omarchy-mlx (head `ff4b52ac`): the split-K pass pair of the bf16 single-query decode SDPA extended from head width 128 to 256. The Qwen3.5 and Qwen3.8 full-attention layers (9B, 2B) have head width 256 and have had only the fused one-workgroup-per-head kernel.

Observation that motivates it (measured 2026-10-28, 13-inch M1, packaged wheel): decode time per token grows with context at about 2 GB/s of KV-cache read against a 58.7 GB/s copy roof; attention alone runs at 3.0 to 3.2 GB/s for 8192 and 16384 keys on the 9B, 2B and 4B shapes (artifacts `decprof-20261028/`).

## Arms
- A (baseline): the same build with `MLX_OMARCHY_SDPA_DECODE_SPLIT=0` (the split route off: the fused kernel as in every release).
- B (treatment): the same build, variable unset (the route's default on that part: from 128 keys on the 13-inch M1, 320 on the M2 Max).
One build for both arms: no build-to-build difference. Interleaved A, B, A, B, A, B on each model and context.

## Measurements
- Primary: decode tokens per second after a prefill of N random token ids (N = 8192 and 32768 for Qwen3.8-2B, 8192 for Qwen3.5-9B), 128 generated tokens, the same runner as the earlier cells (`t7b-mlx-omarchy.py`, sha256 `5fe9de49...`), 3 interleaved pairs per cell.
- Secondary: decode tokens per second at context 256 (no regression check) with `decode-profile.py`.
- Attention alone: the micro-benchmark `sdpa-decode-bench.py` per shape and length, both arms.
- Correctness: the split-K tests against a float64 reference on the device (new head-width-256 cases), the primitive suite, and a greedy decode of a fixed real prompt, 48 tokens, both arms: token ids compared.
Machines: 13-inch M1 (G13G) first, then 14-inch M2 Max (G14C) when a slot is free.

## Hypotheses and decision rule (fixed now)
- H1: B decodes the Qwen3.5-9B at 8192 context at least 1.25 times as fast as A (A is about 4.70 tokens per second).
- H2: B decodes the Qwen3.8-2B at 32768 context at least 1.5 times as fast as A (A is about 4.60).
- H3: B at context 256 is at least 0.97 times A (no regression on short contexts).
- Accept for a PR: H1 and H3 hold on the M1 (median of the three pair ratios meets the bound AND every pair ratio is above 1), the device tests pass, and the greedy token ids of A and B agree on at least 44 of 48 tokens (float32 order differs, bit equality is not expected).
- If a bound is missed the result is reported as measured; the bound is not changed afterwards.
- Not claimed in advance: any number on the M2 Max, output quality beyond the token comparison.

## Addendum 2026-10-28 17:03Z (before the numerics run)
- Numerics rule (the 2026-10-23 hd128 receipt found +0.27 percent perplexity at 7600 tokens with the cause unknown, so this change gets the same kind of check): teacher-forced decode over 400 scored steps after a 3000-token prefill of real text (repository docs), arms A and B, Qwen3.8-2B and Qwen3.5-9B. Pass if the mean NLL difference B minus A is within +-0.01 nats per token on both models. Larger differences are reported as measured; a rise above +0.01 means the route does not ship as default for head width 256 without a follow-up.
- Token agreement (already run, before this line): 2B and 9B 48 of 48 token ids equal, A vs B.
