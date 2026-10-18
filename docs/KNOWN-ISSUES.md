# Known issues

Back to the [README](../README.md). Driver differences from the packaged build are in [known-defects.md](known-defects.md). The tables these notes belong to are in [BENCHMARKS.md](BENCHMARKS.md).

## Output differs from macOS

On Qwen3.5-9B the Linux outputs are token-for-token identical to macOS for the 10 test generations. On the four larger models and on gemma-4-E4B the output digests differ. The greedy outputs of the 10 generations differ from macOS in at least one token. No cause has been established yet.

## Models withdrawn

Withdrawn: Ornith-1.5-9B has no row. It does not run cleanly on the 16 GB M1; the system's out-of-memory killer ended processes during the run.

## Models that could not run

Could not run, by size (MLX 4-bit, or the smallest repository): GLM-5.3-Flash 204 GB, GLM-5.3 418 GB, DeepSeek-V4-Flash 151.5 GB, Kimi-K3 (2-bit) 181 GB and Qwen3.8-Flash-Next 111.5 GB. None fits a 94 GB M2 Max. Also not run: Kolibri-1 (custom architecture, no MLX or GGUF conversion) and Xing4.0 (architecture not in mlx-lm). Ternary-Bonsai-2-27B exists only as a GGUF with a PQ2_0 type, and has no MLX conversion. llama.cpp at the pinned commit gave no result. The lab reports that it refuses the file as an invalid ggml type 142 and that the authors' fork is needed.

## Decode on the 13-inch M1 is 44 percent lower while the ANE is busy

On the 13-inch M1, a busy ANE cuts the speed of the GPU. On 2026-10-16, on one boot with the performance governor, Qwen3.5-9B decode read 12.52 tokens per second with the ANE idle and 7.00 with a Parakeet encoder loop running on the ANE (100 percent busy): 44.1 percent lower. Prefill read 97.94 and 94.60 (3.4 percent lower). The generated tokens were identical in both states (output digest 80274aa790426468). The macOS reference for this model is 12.9 tokens per second, so the idle-ANE row is 97 percent of it and the busy-ANE row is 54 percent. gemma-4-E4B shows the same loss in an earlier block: 15.67 and 15.71 with the ANE idle, 9.19 with it busy (41 percent lower).

The loss follows the wide-load read bandwidth the GPU gets. A Vulkan streaming read (`tools/vkbw`, 16-byte loads, 512 MiB) read 58 to 61 GB/s with the ANE idle and 33 GB/s with it busy (a loss of about 27 GB/s), a ratio of 0.54 to 0.57 in two runs in opposite orders. Reads with 2-byte, 4-byte and 8-byte loads are not a filler result: their idle-ANE values moved by up to 80 percent between the two runs, so we do not quote a ratio for them. We did not check which load width the decode kernels use. The cost of one small GPU operation rose by 7 to 12 percent in two runs. The ANE loop itself streams at most 3.3 GB/s, so the size of the loss is not explained by its own traffic. We have not found the cause. The loss was not seen on the 14-inch M2 Max in one interleaved block (gemma-4-26B-A4B, 13.96 with the ANE idle and 13.19 to 14.26 with it busy). It was not measured on the 16-inch M1 Max.

On 2026-10-11, on one boot and one wheel with the v3 driver, Qwen3.5-9B decode fell from 12.45 to 6.93 tokens per second between 07:53Z and 08:00Z (44 percent), gemma-4-E4B fell from 13.64 at 06:16Z to 8.89 at 08:08Z (35 percent), and Llama-3.1-8B stayed at 11.52 and 11.67. A busy ANE loop explains a drop of this size and was started again at 08:01Z, but we have not shown that it caused this drop: the times may be the start or the end of the rows, and the sources were not re-read.

The CPU governor also changes decode on this chip. With the ANE idle, gemma-4-E4B read 10.65, 10.66 and 10.63 tokens per second under the schedutil governor and 15.27, 15.34 and 15.17 under performance (ratio 1.434, one block). The size of this gap depends on the wheel. On 2026-10-17, with the ANE idle, the development wheel 0.32.4.dev202610102357+e57cb301 (main at e57cb301) read 10.69, 10.64 and 10.59 under schedutil and 15.52, 15.49 and 15.47 under schedutil with a per-process CPU utilization clamp (uclampset -m 1024; this is not the performance governor). The same wheel read 15.67 and 15.71 under performance on 2026-10-13, so the schedutil-to-performance pair on that wheel is 10.64 to about 15.7 (about +47 percent, two different days). The wheel 0.32.4.dev202610141543+7604d92c read 13.27 under schedutil against 15.07 under performance (medians of three, +13.6 percent), and Qwen3.5-9B on that wheel read +2.0 percent under performance (median of three); the 9B was not measured on the e57cb301 wheel. Name the wheel next to any governor ratio. On the 16-inch M1 Max, gemma-4-26B-A4B read 10.40 and 10.39 under schedutil and 12.84, 12.79 and 12.80 under performance (ratio 1.231). With the ANE busy, gemma-4-E4B read about 9 tokens per second under both governors (9.18 under performance, 8.97 to 9.02 under schedutil), on a different driver and at a different hour from the idle block. See "Set the CPU governor for local AI" in [BENCHMARKS.md](BENCHMARKS.md#set-the-cpu-governor-for-local-ai) for the setting.

Limits: one Qwen3.5-9B decode pair in each order on one boot (ANE idle first: 12.52 idle, 7.00 busy; ANE busy first: 6.98 busy, 12.64 idle, read after a hot busy arm), so the loss was 44 percent in both orders, but the temperature at the start of each row was not matched and there is no ABBA run; the vkbw figures are two pairs in opposite orders; the cause of the bandwidth loss is open.

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

## Sum over a long row: row-reduce route, on by default

A float32, float16 or bfloat16 Sum over a row-contiguous suffix of at least 256 elements runs one 256-wide workgroup per output row with a fixed shared-memory tree (`shaders/reduce_row.comp`). `MLX_OMARCHY_ROW_REDUCE=0` restores the serial kernels, which gave one invocation a whole row. Max and Min are not affected.

The sum associates differently, so its bits can differ from the serial kernel in the last places. A model whose graph computes a mean or a sum over a long float row outside a fused kernel produces different bits. Muse Spark Open is such a model. Its norm takes the mean of a (1, 1, 8192) float32 tensor. The model code of gemma3n, kimi_k3 and minimax also computes a sum or mean over the last axis of a float tensor.

The route is the default. `MLX_OMARCHY_ROW_REDUCE=0` restores the serial kernels.

Models that change output are those whose graph reduces a float row of 256 or more elements with a Sum that is not inside a fused kernel. The whole-process dispatch profiles of Qwen3.5-9B and Qwen3.8-27B on an M2 Max, prefill and decode, contain no Reduce dispatch of any kind, so the route is not reached. The sums in the gemma-4 text generation path, qwen3-next, qwen3-moe, gpt-oss and deepseek model code run over the expert axis, over the top-k width (fewer than 256 elements) or over integers. The route needs the trailing axes and a float type, so those sums do not take it. The embedding-norm check in gemma-4 that runs when input embeddings are supplied sums over the last axis.

On Muse Spark Open, a teacher-forced run over three passages of 255 tokens each gave these mean differences in negative log-likelihood per token (route on minus route off, in nats).

| Passage | Off | On | Difference | Standard error |
| --- | --- | --- | --- | --- |
| GPL v3 | 0.13535 | 0.13309 | -0.00226 | 0.0055 |
| Apache 2 | 0.25747 | 0.24504 | -0.01244 | 0.0108 |
| README | 2.86670 | 2.87443 | +0.00773 | 0.0133 |

On Apache 2 the difference of -0.01244 is outside the registered bound of 0.01. A second run over eight passages of technical prose written after the training data gave a mean difference of +0.00484 nats per token. The interval of the mean plus or minus 2.365 standard errors over passages is [-0.00960, +0.01928], so it includes 0 and ends at +0.019.

The route's unit test, `omarchy_reduce_row_tests` (6 cases, 216 assertions), passes on the 13-inch M1, the 16-inch M1 Max and the 14-inch M2 Max. It checks float32, float16 and bfloat16 rows of 256 to 70001 elements against a long double sum, and checks that Max and Min keep the serial route. The measurements, commands and run procedure are in `receipts/2026-10-17-row-reduce-default/`.

## gather_qmm word-per-lane route: developer switch, numerics verdict REJECT

A word-per-lane decode kernel for `gather_qmm` (bf16, 4-bit, group 64, one row per index, 32-lane subgroup) is built into the backend and stays off. `MLX_OMARCHY_GATHER_QMM_WORD=1` is a developer switch and is not a supported setting.

Speed, one Apple M2 Max, one tree, three paired rounds: decode 23.91 tok/s with the switch off and 34.59 with it on on gemma-4-26B-A4B 4-bit (+44.7 percent), and 27.70 against 40.99 on Qwen3.6-35B-A3B 4-bit (+48.0 percent). Prefill is unchanged. The output token streams differ between the two settings.

Numerics, teacher-forced negative log-likelihood on gemma-4-26B-A4B, word route minus the 8-output element kernel, windows 0 to 127 (8192 positions): paired mean +0.00906 nats, cluster standard error 0.00757, interval [-0.00608, +0.02420]. The preregistered rule accepts only an interval wholly inside plus or minus 0.02, so the verdict is REJECT. The point estimate is slightly worse, and the interval contains 0. The route picks a different top token at 7.6 percent of positions. Each output equals the dense `quantized_matmul` on the selected expert bit for bit, which is a statement about the kernel arithmetic, not about model quality. No README row comes from this route.
