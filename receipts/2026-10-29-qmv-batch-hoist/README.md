# Hoisted nibble conversions in the batch-shared 4-bit qmv kernel

Result: the batch-shared bf16 GEMV that serves 2 to 4 row decode batches (`MLX_OMARCHY_QMV_BATCH`, on by default) now computes the eight nibble-to-float conversions of each weight word and column once and reuses them for every row. Every row keeps the same values in the same `fma` chain, and the output is bit-identical to the previous kernel in every check below. A batch-4 decode step of Qwen3.8-2B (4-bit) is faster on all three chips: 1.203x on a 13-inch M1, 1.152x on a 14-inch M2 Max, 1.083x on a 16-inch M1 Max. Batch 3 is 1.163x, 1.102x and 1.041x. Batch 2 is 1.081x on the 13-inch M1, unchanged on the 16-inch M1 Max, and **a 5.6 percent longer step on the 14-inch M2 Max (0.947x speed, 29.1 ms instead of 27.6 ms)**. That regression is real and is not fixed here; see "Batch 2 on the 14-inch M2 Max".

## What changed

`shaders/qmm_vec.comp`, macro `Q4_ROWS_BATCH`: `Q4_NIB` (shift, and, or, subtract) used to run inside the row loop, once per row, so each added row cost about one weight stream of ALU work. A per-word table `na[ROWS_PER_SLOT][8]` (and `nb` for the second word of a 512-wide step) is filled before the row loop and read by every row. The shipped macro is deleted, not kept behind a flag. The single-row kernels and every other quantization format are untouched.

## Exactness (all held)

- Kernel level, all three chips: 3 shapes (2048x6144, 2048x2048, 2048x16384) x T = 1, 2, 4, 6, 8, 10, 12, 14, 16, random bf16 activations (fixed seed), shipped 4-row kernel against the hoisted kernel: maximum absolute difference 0 in all 27 cells on each chip (column `|b4-b4h|` of `microbench-16-inch-M1-Max.log`, `microbench-13-inch-M1.log`, `microbench-14-inch-M2-Max.log`).
- Decode level, all three chips: the generated token ids of every row, hashed per batch size, are equal in every run of both arms: batch 2 `14e55e5f88bb975a`, batch 3 `d02eb182bb95ef5b`, batch 4 `87366d53d3d009e8` (`decode-ab-*.log`, 3 runs per arm per batch size per chip).
- SPIR-V: compiled on the development machine with the same flags, the 4-row module of this branch is byte-identical to the hoisted 4-row module of the spike branch the kernel-level measurement used. The wheels were built on the 16-inch M1 Max with its own shader compiler; their modules were not compared.
- The qmv batch doctests (`omarchy_qmv_batch_tests`) skip all four cases on the development machine's software Vulkan device (the subgroup GEMV route does not engage there), so that machine exercises none of them. They have not run on Apple hardware for this change; the hardware checks above (27 kernel cells per chip, token ids per decode run) stand in for them.

## Speed

Decode: kernel 7.1.12-2-20261029.2-lab-ARCH (16-inch M1 Max), 7.1.12-2-20261013.18-sep-ARCH (13-inch M1), 7.2.11-lab (14-inch M2 Max); `mlx_lm` 0.31.3 `BatchGenerator`, greedy, 64 new tokens per row, four short different prompts (left padded), median step time after the first step, 5 repetitions per run, 3 interleaved pairs per batch size (the order alternates), each run its own process. Arm A is the wheel built at `f91094a1` (shipped kernel on the decode path), arm B the wheel built at `c1422078` (this change). Absolute milliseconds are not comparable between chips or boots; only the in-process A/B ratios are used.

| Chip | Batch | A, ms per step (3 runs) | B, ms per step (3 runs) | A / B (medians) |
| --- | --- | --- | --- | --- |
| 16-inch M1 Max | 2 | 29.69, 29.82, 29.88 | 29.79, 29.81, 30.21 | 1.000 |
| 16-inch M1 Max | 3 | 31.41, 31.43, 31.49 | 30.11, 30.21, 30.24 | 1.041 |
| 16-inch M1 Max | 4 | 36.92, 36.90, 36.85 | 34.09, 34.03, 34.08 | 1.083 |
| 13-inch M1 | 2 | 69.49, 69.56, 69.45 | 64.26, 64.33, 64.27 | 1.081 |
| 13-inch M1 | 3 | 89.42, 89.52, 89.65 | 77.05, 76.97, 76.92 | 1.163 |
| 13-inch M1 | 4 | 116.45, 116.57, 116.43 | 96.78, 96.81, 96.69 | 1.203 |
| 14-inch M2 Max | 2 | 27.12, 28.13, 27.57 | 28.38, 29.12, 29.11 | 0.947 |
| 14-inch M2 Max | 3 | 17.35, 17.38, 17.39 | 15.81, 15.77, 15.71 | 1.102 |
| 14-inch M2 Max | 4 | 19.99, 20.05, 20.02 | 17.35, 17.38, 17.37 | 1.152 |

Kernel level (one process, tile route / shipped 4-row / hoisted 4-row, 3 rounds of 30 calls, per call including eval and synchronise), hoisted against shipped at 4 rows on 2048x6144 / 2048x2048 / 2048x16384: 1.10x / 1.03x / 1.22x on the 14-inch M2 Max, 1.19x / 1.02x / 1.25x on the 16-inch M1 Max, 1.34x / 1.23x / 1.48x on the 13-inch M1. At 1 and 2 rows it is unchanged within noise on the 16-inch M1 Max (0.96x to 1.07x); on the 14-inch M2 Max at 2 rows it is 1.03x, 0.99x and 1.08x. A hoisted 8-row pass is within 6 percent of the hoisted 4-row passes it replaces on the 16-inch M1 Max, and a hoisted 16-row pass is up to 13 percent slower there (2048x16384 at 12 rows; 1 to 3 percent slower at 12 rows on the 13-inch M1), so the 4-row kernel stays the one in use. No 8- or 16-row kernel is in this change.

## Dispatch trace

`dispatch-trace-13-inch-M1.log` and `dispatch-trace-14-inch-M2-Max.log`, both wheels: with the batch route on, the batch kernel (kernel id 547 in that process, found as the one id that appears with the route on and not with it off) runs 210.38 times per decode step at batch 2, 3 and 4, identically in both wheels, and no other batch kernel runs. So every decode difference above comes from that one kernel. The 16-inch M1 Max trace was not recorded (the run was truncated).

## Batch 2 on the 14-inch M2 Max

`batch2-route-diag-14-inch-M2-Max.log` (kernel 7.2.11-lab, 3 interleaved pairs, median ms per step): shipped wheel, route on 28.02 and route off 28.03; this change, route on 29.06 and route off 28.13. With the route off the two wheels are equal, so the loss comes from the hoisted batch kernel and from nothing else in the wheel. On the 13-inch M1 the same four cells (`batch2-route-diag-13-inch-M1.log`) read 69.45 and 71.53 for the shipped wheel and 64.16 and 71.47 for this change: the route is worth 3 percent shipped and 11 percent hoisted there.

The cause is not isolated. In isolation the hoisted kernel is not slower on the 14-inch M2 Max at 2 rows (above), and the shipped batch route gives no gain at batch 2 on that chip (28.02 against 28.03). The M2 batch-2 step also dispatches 1522 kernels against 883 at batch 3, with a different kernel mix, so it runs a different graph than batches 3 and 4.

### A per-quad variant did not fix it

Written before any measurement (`PREREG.md`, Amendment 3), the hypothesis was that the whole-word form keeps 32 conversion floats live across the row loop and costs occupancy on the 14-inch M2 Max. The variant (`38ce0a37`) converts per quad, 8 conversion floats live, and runs the row loop inside the quad loop. It stays bit-identical (`per-quad-exactness-14-inch-M2-Max.log`: 0 of 144 datasets differ from the per-row kernel in all three wheels, same combined hash). On the 14-inch M2 Max (kernel 7.2.11-lab, `per-quad-decode-14-inch-M2-Max.log`, 3 interleaved triples per batch size, median ms per step, ids hash equal in all 27 runs):

| Batch | shipped | this change | per-quad | this change / shipped | per-quad / shipped |
| --- | --- | --- | --- | --- | --- |
| 2 | 27.95 | 28.78 | 28.81 | 0.971x | 0.970x |
| 3 | 17.40 | 15.80 | 17.62 | 1.101x | 0.987x |
| 4 | 20.06 | 17.40 | 18.64 | 1.153x | 1.076x |

It does not remove the batch-2 loss (0.970x), it loses the batch-3 gain (0.987x), and it keeps about half of the batch-4 gain. The register-pressure hypothesis predicted the opposite; it was wrong. The per-quad variant is not in this change and was not run on the other two chips.

## Not run, not recorded

- The qmv batch doctests have not run on Apple hardware for this change.
- The 16-inch M1 Max dispatch trace and batch-2 route diagnosis were not recorded.
- These logs do not record the Mesa build or the firmware.
- The kernel-level exactness check compares the hoisted kernel with the shipped one through the `MLX_OMARCHY_QMV_SMALL_T` route (T = 2 to 16), which dispatches the same kernels; the decode check goes through the default batch route. `exact_fingerprint.py` checks the default route against the per-row kernel for the per-quad variant and does not prove the batch kernel ran in every cell.
- Model: `Qwen3.8-2B-mlx-4Bit`, `model.safetensors` sha256 `b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863`.

## Files

- `PREREG.md`: the written plan with its stop rule and three amendments, each dated before the measurement it precedes.
- `microbench-*.log`, `decode-ab-*.log`: raw kernel-level and decode output per chip (host names replaced by machine names).
- `dispatch-trace-*.log`, `batch2-route-diag-*.log`: trace and route on/off runs on the 13-inch M1 and 14-inch M2 Max.
- `per-quad-*.log`: the per-quad variant on the 14-inch M2 Max.
- `qmm16c-bench.py`, `bg_decode_ab.py`, `exact_fingerprint.py`, `summarize_decode3.py`: the measurement scripts.
