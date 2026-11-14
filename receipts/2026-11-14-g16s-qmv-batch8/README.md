# G16S wider batch-shared decode GEMV (batch 8)

Short-pass verify of 5 to 8 rows was issued as two 4-row dispatches, so the
weight stream was read twice. `QmmVecQ4WordSubgroupBatch8BF16` runs the same
per-row decode chain in one 8-row dispatch, reading the weight words once.
Each output row stays bit-identical to the single-row GEMV and to the batch-4
kernel. Default on where the device short-pass cap is 8 (M4 Pro G16S);
`MLX_OMARCHY_QMV_BATCH8=0` restores 4-row blocks.

## Verify-pass cost, device "Apple M4 Pro (G16S B1)", SMALL_T 8

| rows T | batch4 (=0) med ms | batch8 (=1) med ms | delta |
|---|---|---|---|
| 1 | 29.55 | 29.29 | ~0 |
| 2 | 36.72 | 36.70 | ~0 |
| 3 | 43.74 | 43.69 | ~0 |
| 4 | 51.93 | 52.15 | ~0 |
| 5 | 76.08 | 59.70 | -21.5% |
| 6 | 83.49 | 68.91 | -17.5% |
| 7 | 91.47 | 79.93 | -12.6% |
| 8 | 98.69 | 87.27 | -11.6% |

T<=4 is one block in both builds, so they match. At T>=5 batch8 is one
dispatch instead of two, which removes one full weight read.

## End-to-end prompt-lookup, Qwen3-8B 4-bit, k=7, batch8 on

All outputs bit-identical to greedy ("identical": true):
- code  : base 33.9 -> 29.1 tok/s (0.86x, low repetition)
- quote : base 33.5 -> 90.4 tok/s (2.70x)
- chat  : base 34.9 -> 22.2 tok/s (0.64x, no draft hits)

Prompt-lookup wins on repetitive spans; the batch8 kernel lowers verify cost
whenever a pass carries 5 to 8 rows, so it can only help or tie the greedy-exact
speculative path. tcost.py and spec.py harness in receipts/2026-11-14-g16s-speculative-lookup.
