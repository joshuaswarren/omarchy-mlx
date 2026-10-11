# Known issues

Back to the [README](../README.md). Driver differences from the packaged build are in [known-defects.md](known-defects.md). The tables these notes belong to are in [BENCHMARKS.md](BENCHMARKS.md).

## Output differs from macOS

On Qwen3.5-9B the Linux outputs are token-for-token identical to macOS for the 10 test generations. On the four larger models and on gemma-4-E4B the output digests differ. The greedy outputs of the 10 generations differ from macOS in at least one token. No cause has been established yet.

## Models withdrawn

Withdrawn: Ornith-1.5-9B has no row. It does not run cleanly on the 16 GB M1; the system's out-of-memory killer ended processes during the run.

## Models that could not run

Could not run, by size (MLX 4-bit, or the smallest repository): GLM-5.3-Flash 204 GB, GLM-5.3 418 GB, DeepSeek-V4-Flash 151.5 GB, Kimi-K3 (2-bit) 181 GB and Qwen3.8-Flash-Next 111.5 GB. None fits a 94 GB M2 Max. Also not run: Kolibri-1 (custom architecture, no MLX or GGUF conversion) and Xing4.0 (architecture not in mlx-lm). Ternary-Bonsai-2-27B exists only as a GGUF with a PQ2_0 type, and has no MLX conversion. llama.cpp at the pinned commit gave no result. The lab reports that it refuses the file as an invalid ggml type 142 and that the authors' fork is needed.

## Decode on the 13-inch M1 dropped partway through one day

On 2026-10-11, on one boot and one wheel with the v3 driver, Qwen3.5-9B decode read 12.51, 12.45 and 12.45 tokens per second between 06:08Z and 07:53Z. It then read 6.93 at 08:00Z and 6.95 at 08:32Z. Prefill stayed between 96 and 97.5. Later runs on the newer driver read 6.98 for Qwen3.5-9B and 6.61 for Llama-3.1-8B. The v3 driver itself read 6.93 for Qwen3.5-9B, so these results do not show a loss caused by the newer driver. The cause is not found. No decode ratio for these models on this chip is published, and nothing here says the newer driver lost speed on it.
