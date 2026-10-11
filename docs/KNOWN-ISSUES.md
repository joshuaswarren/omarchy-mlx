# Known issues

Back to the [README](../README.md). Driver differences from the packaged build are in [known-defects.md](known-defects.md). The tables these notes belong to are in [BENCHMARKS.md](BENCHMARKS.md).

## Output differs from macOS

On Qwen3.5-9B the Linux outputs are token-for-token identical to macOS for the 10 test generations. On the four larger models and on gemma-4-E4B the output digests differ. The greedy outputs of the 10 generations differ from macOS in at least one token. No cause has been established yet.

## Models withdrawn

Withdrawn: Ornith-1.5-9B has no row. It does not run cleanly on the 16 GB M1; the system's out-of-memory killer ended processes during the run.

## Models that could not run

Could not run, by size (MLX 4-bit, or the smallest repository): GLM-5.3-Flash 204 GB, GLM-5.3 418 GB, DeepSeek-V4-Flash 151.5 GB, Kimi-K3 (2-bit) 181 GB and Qwen3.8-Flash-Next 111.5 GB. None fits a 94 GB M2 Max. Also not run: Kolibri-1 (custom architecture, no MLX or GGUF conversion) and Xing4.0 (architecture not in mlx-lm). Ternary-Bonsai-2-27B exists only as a GGUF with a PQ2_0 type, and has no MLX conversion. llama.cpp at the pinned commit gave no result. The lab reports that it refuses the file as an invalid ggml type 142 and that the authors' fork is needed.

## Decode on the M1-class chips follows the CPU clock

On 2026-10-11, on one boot and one wheel with the v3 driver, Qwen3.5-9B decode fell from 12.45 to 6.93 tokens per second between 07:53Z and 08:00Z (44%). gemma-4-E4B fell from 13.64 at 06:16Z to 8.89 at 08:08Z (35%). Llama-3.1-8B stayed at 11.52 and 11.67 over the same hours. Prefill stayed within 3%. A test later that day, on another boot, read gemma-4-E4B at 10.65, 10.66 and 10.63 tokens per second under the schedutil governor and 15.27, 15.34 and 15.17 under performance (ratio 1.434). On the 16-inch M1 Max, gemma-4-26B-A4B read 10.40 and 10.39 under schedutil and 12.84, 12.79 and 12.80 under performance (ratio 1.231). We have not shown that the governor caused the drop on the morning of the 11th, because no run set it back during the drop. Rows before 22:36Z did not record the governor. See "Set the CPU governor for local AI" in [BENCHMARKS.md](BENCHMARKS.md#set-the-cpu-governor-for-local-ai) for the setting.
