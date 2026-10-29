# Hoisted nibble conversions in the batch-shared 4-bit qmv kernel

Result: the batch-shared bf16 GEMV that serves 2 to 4 row decode batches (`MLX_OMARCHY_QMV_BATCH`, on by default) now computes the eight nibble-to-float conversions of each weight word and column once and reuses them for every row. Every row keeps the same values in the same `fma` chain, and the output is bit-identical to the previous kernel in every check below. On a 16-inch M1 Max, a batch-4 decode step of Qwen3.8-2B (4-bit) takes 34.1 ms instead of 36.9 ms (1.083x), batch 3 takes 30.2 ms instead of 31.4 ms (1.041x), batch 2 is unchanged (29.8 ms).

## What changed

`shaders/qmm_vec.comp`, macro `Q4_ROWS_BATCH`: `Q4_NIB` (shift, and, or, subtract) used to run inside the row loop, once per row, so each added row cost about one weight stream of ALU work. A per-word table `na[ROWS_PER_SLOT][8]` (and `nb` for the second word of a 512-wide step) is filled before the row loop and read by every row. The shipped macro is deleted, not kept behind a flag. The single-row kernels and every other quantization format are untouched.

## Exactness (all held)

- Kernel level, 16-inch M1 Max: 3 shapes (2048x6144, 2048x2048, 2048x16384) x T = 1, 2, 4, 6, 8, 10, 12, 14, 16, random bf16 activations (fixed seed), shipped 4-row kernel against the hoisted kernel (and against 8- and 16-row hoisted builds): maximum absolute difference 0 in all 27 cells (`microbench-16-inch-M1-Max.log`, columns `|b4-b4h|`, `|b4-b8h|`, `|b4-b16h|`).
- Decode level, 16-inch M1 Max: the generated token ids of every row, hashed per batch size, are equal in all six runs of each batch size (three runs per arm): batch 2 `14e55e5f88bb975a`, batch 3 `d02eb182bb95ef5b`, batch 4 `87366d53d3d009e8` (`decode-ab-16-inch-M1-Max.log`).
- SPIR-V: compiled on the development machine with the same flags, the 4-row module of this branch is byte-identical to the hoisted 4-row module of the spike branch the kernel-level measurement used. The wheels were built on the 16-inch M1 Max with its own shader compiler; their modules were not compared.
- The qmv batch doctests (`omarchy_qmv_batch_tests`) skip all four cases on the development machine's software Vulkan device (the subgroup GEMV route does not engage there), so that machine exercises none of them. Before the probe fix in #302 the short-pass case failed there with identical mismatch counts with and without this change.

## Speed

Decode, 16-inch M1 Max, kernel 7.1.12-2-20261029.2-lab-ARCH, `mlx_lm` 0.31.3 `BatchGenerator`, greedy, 64 new tokens per row, four short different prompts (left padded), median step time after the first step, 5 repetitions per run, 3 interleaved pairs per batch size (the order alternates), each run its own process. Arm A is the wheel built at `f91094a1` (shipped kernel on the decode path), arm B the wheel built at `c1422078` (this change).

| Batch | A, ms per step (3 runs) | B, ms per step (3 runs) | A / B (medians) |
| --- | --- | --- | --- |
| 2 | 29.69, 29.82, 29.88 | 29.79, 29.81, 30.21 | 1.000 |
| 3 | 31.41, 31.43, 31.49 | 30.11, 30.21, 30.24 | 1.041 |
| 4 | 36.92, 36.90, 36.85 | 34.09, 34.03, 34.08 | 1.083 |

Kernel level (same machine, one process, tile route / shipped 4-row / hoisted 4-row, 3 rounds of 30 calls, per call including eval and synchronise): the hoisted 4-row kernel is 1.19x (2048x6144), 1.02x (2048x2048) and 1.25x (2048x16384) faster than the shipped one at 4 rows, and 1.24x, 1.05x and 1.36x at 8 rows. At 1 and 2 rows it is unchanged within noise (0.96x to 1.07x). A hoisted 8-row pass is within 6 percent of the hoisted 4-row passes it replaces, and a hoisted 16-row pass is up to 13 percent slower (2048x16384 at 12 rows), so the 4-row kernel stays the one in use.

## Not run, not recorded

- The 13-inch M1 and the 14-inch M2 Max have not run the hoisted kernel. `microbench-13-inch-M1-before-hoist.log` is the 13-inch M1 measurement that located the cost (about 0.09 ms per added row on 2048x6144 at every block width, in proportion to the weight bytes); the hoist was written from it, not measured there.
- The qmv batch doctests have not run on Apple hardware for this change.
- These logs do not record the Mesa build, the firmware, or the dispatch trace. The decode path's use of the 4-row kernel at batch 2 to 4 is from the code (`primitives.cpp`, `batch >= 2 && batch <= 4`), not from a trace.
- The kernel-level exactness check compares the hoisted kernel with the shipped one through the `MLX_OMARCHY_QMV_SMALL_T` route (T = 2 to 16), which dispatches the same kernels; the decode check goes through the default batch route.
- Model: `Qwen3.8-2B-mlx-4Bit`, `model.safetensors` sha256 `b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863`. Absolute milliseconds are not comparable across boots; only the in-process A/B ratios are used.

## Files

- `PREREG.md`: the written plan with its stop rule and two amendments, each dated before the measurement it precedes.
- `microbench-16-inch-M1-Max.log`, `decode-ab-16-inch-M1-Max.log`, `microbench-13-inch-M1-before-hoist.log`: raw output (host names replaced by machine names).
- `qmm16c-bench.py`, `bg_decode_ab.py`: the two measurement scripts.
