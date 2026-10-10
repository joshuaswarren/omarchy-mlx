# M2 Max (96 GB) on Linux: same models and protocol as the "How close to macOS" table

Linux only. There is no macOS measurement for this chip, so no ratio is claimed here. Same wheel stamp (`0.32.4.dev202610090923+a48b7cc5`), installer mlx-lm patches, llama.cpp commit 65840ed built for Vulkan, default `llama-bench` flags (all layers offloaded, 12 threads, batch 2048, ubatch 512, flash attention auto), `-p 512 -n 128 -r 5`. MLX decode is the median of 10 greedy generations of 64 tokens, prefill the median of 5 runs of a 512-token prompt.

| Model | MLX decode | MLX prefill 512 | llama.cpp tg128 | llama.cpp pp512 |
|---|---|---|---|---|
| Qwen3.8-27B | 14.7 | 109.3 | 6.3 (median 6.9, one slow repeat) | 66.8 |
| gemma-4-26B-A4B (MoE) | 10.5 | 37.8 | 29.5 | 321.5 |
| gemma-4-31B | 8.7 | 80.5 | 7.4 | 56.2 |
| Qwen3.5-9B | 38.7 | 447.8 | 22.3 | 219.1 |
| Qwen3.6-35B-A3B (MoE) | 12.8 | 51.2 | 29.2 | 305.7 |
| gemma-4-E4B | fails to run (GPU timeline stall) | fails to run | 27.2 | 373.6 |

The greedy output digests of the five MLX rows that run are identical to the M1 Max rows on the same stack. The Qwen3.8-27B tg128 mean of 6.3 comes from four repeats at 6.9 and one at 3.8; the median is 6.9. Raw per-run JSON is in `mlx/` and `llamacpp/` (llama-bench output reduced to its measurement fields).
