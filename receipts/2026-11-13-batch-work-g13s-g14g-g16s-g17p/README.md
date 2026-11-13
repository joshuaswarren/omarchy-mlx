# Submission budgets for G13S, G14G, G16S and G17P

Date: 2026-11-13. This receipt backs the four new rows of `kBatchWorkCalibration` in `overlay/mlx/backend/omarchy/encoder.h`: M1 Pro (G13S) 20000, M2 (G14G) 10000, M4 Pro (G16S) 10000, A18 Pro (G17P) 5000. Before this change all four boards took the 5000 fallback.

## Rule

The rule is the one in `receipts/2026-10-20-g13g-batch-work-8000`. Each board gets the largest swept budget that keeps the submission p50 of both models at or under 6 ms.

## Machines and stack

| Board | Machine | Memory | OS | Kernel | Vulkan driver |
|---|---|---|---|---|---|
| G13S | MacBook Pro 16-inch 2021, M1 Pro (`apple,j316s` / `apple,t6000`) | 16 GB | Fedora 45 | 7.3.0, aurora-silicon/linux `e6611926` | Mesa 26.3.0-devel Borealis, aurora-silicon/mesa `be147e3625`, device `Apple M1 Pro (G13S C0)` |
| G14G | MacBook Air 13-inch 2022, M2 (`apple,j413` / `apple,t8112`) | 8 GB | Fedora 44 | 7.3.0, `e6611926` | Mesa `be147e3625`, device `Apple M2 (G14G B0)` |
| G16S | MacBook Pro 16-inch, M4 Pro (`apple,j616s` / `apple,t6040`) | 24 GB | Fedora 45 | 7.3.0, `c6c15acf` | Mesa `c208afc352`, device `Apple M4 Pro (G16S B1)` |
| G17P | MacBook Neo, A18 Pro (`apple,j700` / `apple,t8140`) | 8 GB | Fedora 45 | 7.3.0, `1de767ba` | Mesa `c208afc352`, device `Apple A18 Pro (G17P B1)` |

The driver reports driver ID 35 (`VK_DRIVER_ID_AURORA_BOREALIS`, admitted since #381) and Vulkan 1.4.363. The ICD is the system `asahi_icd.aarch64.json`, and `omarchy-mlx-vulkan` is not installed. Every run used Python 3.14, mlx-lm 0.31.3 with the `install.sh --patch-venv` series, the `schedutil` governor, and a desktop session with no other GPU work.

## Submission sizes

Diagnostics build of `v0.7.41` (`8380f002`) with `-DMLX_OMARCHY_GPU_PROFILING=ON` (`tools/build_diag.sh`). It was built on each board's OS: the M1 Pro build served the M1 Pro, M4 Pro and A18 Pro, and the M2 built its own because Fedora 44 has an older glibc. `scripts/subcap_calibrate.py` (unchanged) ran 64 decode tokens after a 512-token prefill, one process per budget (`tools/run_subcap.sh`). Raw output is in `subcap/`.

Submission p50 / p95 in ms:

| Budget | M1 Pro 2B | M1 Pro 4B | M2 2B | M2 4B | M4 Pro 2B | M4 Pro 4B | A18 Pro 2B | A18 Pro 4B |
|---:|---|---|---|---|---|---|---|---|
| 5000 | 1.13 / 2.94 | 1.27 / 1.56 | 2.01 / 5.75 | 2.41 / 3.94 | 2.62 / 3.54 | 2.63 / 2.82 | **4.06 / 7.69** | **4.35 / 4.69** |
| 8000 | 1.68 / 4.04 | 1.95 / 2.17 | 3.04 / 7.38 | 3.63 / 3.81 | 1.38 / 5.93 | 4.01 / 4.33 | 5.07 / 11.24 | 6.56 / 7.00 |
| 10000 | 2.09 / 3.72 | 2.50 / 2.71 | **3.94 / 6.44** | **4.80 / 4.87** | **1.60 / 6.31** | **5.40 / 5.63** | 5.07 / 9.55 | 8.84 / 9.46 |
| 15000 | 1.33 / 4.68 | 3.71 / 3.97 | 2.29 / 8.35 | 7.18 / 7.29 | 1.35 / 10.09 | 8.14 / 8.57 | 4.35 / 14.75 | 13.40 / 14.40 |
| 20000 | **1.56 / 5.82** | **4.70 / 5.36** | 2.68 / 10.72 | 9.00 / 11.21 | 1.35 / 14.33 | 10.37 / 11.58 | 5.00 / 21.18 | 17.44 / 19.72 |
| 40000 | 3.11 / 8.91 | 9.31 / 9.56 | 5.81 / 15.92 | 18.00 / 18.28 | 2.72 / 30.63 | 27.91 / 29.52 | 7.10 / 42.47 | 41.83 / 43.82 |

2B is `SiddhJagani/Qwen3.8-2B-mlx-4Bit` and 4B is `mlx-community/Qwen3-4B-Instruct-2507-4bit`. Bold marks the chosen budget. The 4B rows rise steadily with the budget and set the limit on every board. The 2B p50 does not rise steadily, and it stays at or under 6 ms at every chosen budget.

## Decode at the chosen budget

Release wheel `v0.7.41` (`0.32.4.dev202611111431+8380f002`, RECORD hashes match). `receipts/2026-10-09-linux-vs-macos/score_bench.py` runs 10 greedy generations of 64 tokens and 5 prefills of 512 tokens, taking medians. Each board ran the default first and then `MLX_OMARCHY_BATCH_WORK=<budget>`, in one sitting (`tools/ab_bw.sh`). Records are in `ab/`.

| Board | Model | Decode, 5000 | Decode, new budget | Ratio | Prefill 512, 5000 | Prefill 512, new budget | Tokens |
|---|---|---:|---:|---:|---:|---:|---|
| M1 Pro, 20000 | Qwen3.8-2B | 61.42 | 77.92 | 1.269 | 279.9 | 284.7 | identical (`8134db97`) |
| M1 Pro, 20000 | Qwen3-4B | 31.27 | 44.20 | 1.413 | 133.5 | 134.9 | identical (`ee938320`) |
| M2, 10000 | Qwen3.8-2B | 38.54 | 40.10 | 1.040 | 168.5 | 169.6 | identical (`aa592095`) |
| M2, 10000 | Qwen3-4B | 18.48 | 20.62 | 1.115 | 79.0 | 78.8 | identical (`ee938320`) |
| M4 Pro, 10000 | Qwen3.8-2B | 105.68 | 127.87 | 1.210 | 511.8 | 513.3 | identical (`912ebb46`) |
| M4 Pro, 10000 | Qwen3-4B | 49.47 | 56.85 | 1.149 | 240.1 | 241.1 | identical (`67c191ee`) |

The A18 Pro row is 5000, the value the fallback already gave, so its default does not change. The M2 is passively cooled, and its 10000 arm started at 90-99 C (load1 1.0-1.3) against 43-70 C for the default arm. That bias works against the new budget. Each cell is one process. The spread across its 10 decode samples is in the JSON (`decode_tok_s.min` / `max`).

## Tests

`overlay/tests/omarchy/test_runtime.cpp`, case "submission-cap default resolves per calibrated board", pins the four rows with the exact device names these drivers report. It also checks that the G14G row does not shadow G14C.
