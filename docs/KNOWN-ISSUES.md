# Known issues

Back to the [README](../README.md). Driver differences from the packaged build are in [known-defects.md](known-defects.md). The tables these notes belong to are in [BENCHMARKS.md](BENCHMARKS.md).

## Output differs from macOS

On Qwen3.5-9B the Linux outputs are token-for-token identical to macOS for the 10 test generations. On the four larger models and on gemma-4-E4B the output digests differ. The greedy outputs of the 10 generations differ from macOS in at least one token. No cause has been established yet.

## Models withdrawn

Withdrawn: Ornith-1.5-9B has no row. It does not run cleanly on the 16 GB M1; the system's out-of-memory killer ended processes during the run.

## Models that could not run

Could not run, by size (MLX 4-bit, or the smallest repository): GLM-5.3-Flash 204 GB, GLM-5.3 418 GB, DeepSeek-V4-Flash 151.5 GB, Kimi-K3 (2-bit) 181 GB and Qwen3.8-Flash-Next 111.5 GB. None fits a 94 GB M2 Max. Also not run: Kolibri-1 (custom architecture, no MLX or GGUF conversion) and Xing4.0 (architecture not in mlx-lm). Ternary-Bonsai-2-27B exists only as a GGUF with a PQ2_0 type, and has no MLX conversion. llama.cpp at the pinned commit gave no result. The lab reports that it refuses the file as an invalid ggml type 142 and that the authors' fork is needed.

## Decode on the 13-inch M1 changed with the CPU governor with no ANE work running

On 2026-10-11, on one boot and one wheel with the v3 driver, Qwen3.5-9B decode fell from 12.45 to 6.93 tokens per second between 07:53Z and 08:00Z (44%). gemma-4-E4B fell from 13.64 at 06:16Z to 8.89 at 08:08Z (35%). Llama-3.1-8B stayed at 11.52 and 11.67 over the same hours. Prefill stayed within 3%. A test later that day, on another boot, read gemma-4-E4B at 10.65, 10.66 and 10.63 tokens per second under the schedutil governor and 15.27, 15.34 and 15.17 under performance (ratio 1.434). On the 16-inch M1 Max, gemma-4-26B-A4B read 10.40 and 10.39 under schedutil and 12.84, 12.79 and 12.80 under performance (ratio 1.231). We have not shown that the governor caused the drop on the morning of the 11th, because no run set it back during the drop. With the ANE filler running on the 13-inch M1, gemma-4-E4B read about 9 tokens per second under both governors (9.18 under performance, 8.97 to 9.02 under schedutil). Those rows use a different driver and hour from the filler-off block, so the governor and ANE effects are not separated yet. Rows before 22:36Z did not record the governor. See "Set the CPU governor for local AI" in [BENCHMARKS.md](BENCHMARKS.md#set-the-cpu-governor-for-local-ai) for the setting.

## Non-Apple development drivers: the independent-streams timing floor is not calibrated on NVIDIA

`omarchy_runtime_tests` asserts that interleaving two independent
streams takes no longer than running them one after the other
(`independent streams measure against the serialized sum`, ratio
floor 0.95). The floor is calibrated on the Apple Honeykrisp device
(ratio 1.21 on the M1) and holds on Intel ANV (55/55 runtime tests,
outside run on a UHD 770, PR
[#111](https://github.com/joshuaswarren/omarchy-mlx/pull/111)).

An RTX 5060 Ti (NVIDIA 615.71.09) measured 0.78-0.88 across 5 runs
in the same outside report. The interleaved copies still verify
byte-for-byte, so the loss is a timing characteristic, not a
correctness failure. The host-visible ReBAR allocation every array uses
on a discrete GPU is possibly the cause. That was not tested. The
report's CPU reads over PCIe, about 3 MB/s, are a different path from
the two-stream copy timing. A global serialization lock would sit near
0.5.
The same run shows one `SUBMIT-RECOVER ... resubmitted 1 stalled
batch` line per run (the submit-recovery ladder fired once; not
investigated).

Until NVIDIA numbers are collected here, the floor is asserted on
every driver except NVIDIA proprietary. The test still prints its
ratio for every run, so a per-driver floor can be set from
collected numbers. AMD (RADV) is untested.
