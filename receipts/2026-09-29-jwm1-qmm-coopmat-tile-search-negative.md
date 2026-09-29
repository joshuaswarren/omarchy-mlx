# 2026-09-29 jwm1 (M1, T8103): qmm coopmat tile-shape search for 512-token prefill — negative results

Do not repeat these. Baseline: the shipped `qmm_coopmat.comp` X32 route (32 rows x 32 columns per workgroup, 64 threads, K step 16, 8 accumulators per
subgroup). Prefill-512 Linux 242.5 tok/s vs macOS 345.3 on the same M1 (receipts in ane-linux-experiments 2026-09-28-jwm1-macos-parity-legs); the qmm
coopmat kernel is 76.7% of the timed leg (~0.94 TFLOP/s by parameter-count estimate, ~36% of the 2.6 TFLOP/s spec peak).

All variants were built as env-gated kernels in one wheel each (default behavior unchanged), same-wheel interleaved A/B, n=15 per arm, one process per sample
(`qwen38-mlx-bench.py --limit 1 --new-tokens 1 --prefill-tokens 512 --warmup 2 --passes 1`), loadavg < ~0.4 at start except where noted, digests
`ccb601895581d89f` identical in EVERY arm and run (outputs are bit-identical; only speed differs):

| variant | env | prefill-512 vs shipped | note |
|---|---|---:|---|
| 64-row tile, 2 subgroups (16 accumulators/subgroup) | `MLX_OMARCHY_QMM_COOPMAT_ROWS=64` | 0.939x (twice: 0.939, 0.939) | early pairs of the first run started at load 1.9 |
| K step 32 | `MLX_OMARCHY_QMM_COOPMAT_STEPK=32` | 0.897x | fewer barriers, more per-step dequant |
| K step 64 | `MLX_OMARCHY_QMM_COOPMAT_STEPK=64` | 0.880x | 8 KiB shared |
| 64-row tile, 4 subgroups (8 accumulators/subgroup) | `MLX_OMARCHY_QMM_COOPMAT_ROWS=64sg4` | 0.978x | same per-thread footprint as shipped |

Reading: weight-tile reuse (rows) and barrier count (K step) are not what limits this kernel; holding registers per subgroup fixed recovers ~4 points of the 64-row loss
but yields no gain. Untested: A-operand global-load traffic, coopmat lowering quality in the compiler, narrower operand types (would change outputs), a wider N tile.
No hardware counters are available on this stack, so the mechanisms are inferences.

Code lives on the pushed branch `agent/qmm-coopmat-tile64` (5eff55057, 49b75c28b, cba19e702), not merged. Private notebook entries jwm1-parity H6/H7/H13.
