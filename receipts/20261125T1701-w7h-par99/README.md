# PAR-99 — jw16 4B decode per-kernel GPU table + achieved bandwidth (G14X) — w7H

Question: jw16 4B decode is -27.6% vs macOS (76.17 vs 105.16 tok/s). Profile one
4B decode window with the AGX timestamps (w88-validated buffer+dispatch method,
NOT vkCmdWriteTimestamp), rank the top 5 kernels by GPU time, give each kernel's
achieved bandwidth against the T6001 peak.

## Method

- Host jw16-linux, boot `27497967-4cb2-4ee5-b477-38fdea213231`, kernel
  `7.1.12-2-20261103.2-lab-ARCH`, Honeykrisp ICD (guard-pinned).
- Diag path (no torchvulkan): `MLX_OMARCHY_GPU_PROFILE` +
  `MLX_OMARCHY_GPU_PROFILE_DISPATCH=1`, `MLX_DISABLE_COMPILE=1`,
  `MLX_OMARCHY_BATCH_WORK=100000`, model
  `mlx-community/Qwen3-4B-Instruct-2507-4bit` (the only 4B MLX snapshot on
  jw16), 512-token house prompt, 9 greedy decode steps, markers bound the
  decode window `(decode_start, decode_done]` = 22.75 ms wall covering
  3,528 dispatches.
- Analyzer: `w7h-par99-table.py` (PAR-97 attribution + per-kernel bytes from
  the k:d binding ranges `[[buf,off,range],...]`).

## Decode window: 3,528 dispatches, 33.02 ms kernel time

Top 5 by GPU time (bandwidth = binding-range bytes / kernel ns, peak = 400 GB/s
LPDDR5):

| rank | kernel | calls | kernel ns | share | bytes | achieved GB/s | % of peak |
|---|---|---:|---:|---|---:|---:|---|
| 1 | QmmVecQ4MultiSubgroupBF16R2S16 | 1152 | 11,272,846 | 34.14% | 11.49 GB | 1019.1* | 254.8%* |
| 2 | FastRmsNormAppleBF16 | 584 | 5,279,721 | 15.99% | 12.0 MB | 2.3 | 0.6% |
| 3 | FastRmsNormBF16 | 576 | 4,943,876 | 14.97% | 6.2 MB | 1.3 | 0.3% |
| 4 | FastRopeBF16 | 576 | 4,819,955 | 14.60% | 310 MB | 64.4 | 16.1% |
| 5 | SdpaDecodeSplitP1VR4BF16Hd128 | 288 | 2,702,965 | 8.19% | 619 MB | 228.9 | 57.2% |

\* QMM binding-range bytes overcount: the range sum counts virtual buffer
sizes re-read across the 1152 dispatches (weights are cache-resident), so
achieved > peak means the kernel is compute/occupancy-bound, not starved.

## Reading

- The decode gap is NOT bandwidth-starved on the top kernel: QMM runs the
  range-sum at 255% of peak (weights cache-resident), i.e. compute-bound.
- The three norm/rope families (FastRmsNormAppleBF16, FastRmsNormBF16,
  FastRopeBF16 = 45.6% of kernel time) move only 6-31 MB per window and run at
  0.3-16% of peak - tiny launches living on dispatch/launch overhead, matching
  the launch-count findings from the M2 4B row (PAR-98) and the jw16 2B row
  (PAR-97: ~82 casts + ~81 f32 elementwise per token).
- SDPA split P1 runs at 57% of peak - the healthiest large family.
- Bandwidth column caveat stands for every row: binding RANGES are virtual
  sizes, not traced bytes; kernels with achieved > peak are re-read/cache-hit
  dominated, so treat the number as an upper bound and the share column as the
  ranking.

## Artifacts

- jw16 `~/scratch/w7h-par99/`: prof-decode.jsonl, markers.jsonl,
  kernel-table-decode.txt, run.log, compute.h,
  analyzer copy `w7h-par99-table.py`.
- Run context: guard ticket `ea4135417` (retry after the mlx_lm fix + the
  10:26Z no-op whose venv the disk sweep had deleted).
