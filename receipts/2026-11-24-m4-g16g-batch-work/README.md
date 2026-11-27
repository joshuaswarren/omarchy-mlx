# G16G (M4) batch work budget: 60000

Machine: MacBook Pro 14-inch M4 (Mac16,1), Omarchy, system Mesa Vulkan ICD (the Omarchy package, not the omarchy-mlx private ICD), omarchy-mlx v0.7.49 (wheel bf5631d).
Model: Qwen3.5 2B text (model.safetensors sha256 b0d5de688567bf4a; qwen3_5 text, 24 layers, hidden 2048, 4-bit group 64, 1.0 GB). Greedy, 128 tokens, 22-token prompt, 6 runs per budget (run 0 warm-up, median of runs 1-5). Fresh boot, screen on and unlocked.

| MLX_OMARCHY_BATCH_WORK | decode tok/s | submissions per token | ms per token | prefill s |
|---|---|---|---|---|
| 5000 (fallback) | 36.33 | 34 | 27.66 | 0.456 |
| 15000 | 44.37 | 13 | 22.53 | 0.389 |
| 30000 | 47.35 | 7 | 21.12 | 0.371 |
| 60000 | 49.07 | 4 | 20.39 | 0.418 |
| 120000 | 49.81 | 2 | 20.07 | 0.514 |
| 1000000 | 50.31 | 1 | 19.89 | 0.700 |

The generated token ids are identical at every budget (one digest, 90280d19). Peak memory is unchanged.
Time per token fits 19.52 ms + 0.238 ms per submission across all budgets (residuals under 0.7%). There is no knee, so the row follows the smooth-band rule: 60000 keeps a decode submission at about 4.9 ms of GPU time. 120000 doubles that, and 1000000 also stretches prefill (0.70 s against 0.37 s at 30000).

Submission GPU time (DRM scheduler run to done, release build) at 60000, all 3306 submissions of the run (prefill and decode, not split): p50 6.14 ms, p99 11.86 ms, max 18.44 ms. At 5000: p50 0.71 ms, p99 2.88 ms, max 11.42 ms. No submission reaches the 33.3 ms frame budget.
