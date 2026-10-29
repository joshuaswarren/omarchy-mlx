# Short-pass Q4 qmm through the Batch4 GEMV (MLX_OMARCHY_QMV_SMALL_T) on G13S: route and numerics

`MLX_OMARCHY_QMV_SMALL_T=<T max>` (opt-in, default off) sends a batch-1 bf16 `[1, T, K]` pass with 2 <= T <= T max, against one transposed 2D 4-bit/group-64 weight, to `QmmVecQ4WordSubgroupBatch4BF16` as one dispatch of up to four rows (T max is capped at 4). Each row is bit-identical to the single-row GEMV, not to the coopmat chain, so this is the numerics decision `receipts/2026-09-29-jwm1-qmm-small-m-negative.md` names. This receipt is route and numerics only. It carries no speed claim: the timing runs on this host did not meet the idle gate (load1 2.9, PSI cpu some avg10 2.1), so a timing receipt will follow on a quiet host with `scripts/mlx_provenance.py`.

## What ran

- Chip: 14-inch M1 Pro (T6000, G13S), 10 cores online (present 0-9). Boot `c94404cc`, kernel `7.2.11-elementary-aurora72-wip4` (elementary OS rather than Omarchy), schedutil governor. Firmware: system 27.0, OS 13.5, iBoot-8422.141.2, mBoot-20457.1.29.
- Driver: Honeykrisp from `joshuaswarren/mesa-1` at `9323171c12`, private ICD via `VK_DRIVER_FILES` (`libvulkan_asahi.so` sha256 prefix `2e83122b98f43f62`). The system Mesa is 26.0.8 and was not used.
- Candidate: commit `0e4d0105` on `5dbc64dda` (main). Wheel `mlx_omarchy-0.32.4.dev202610281820+0e4d010`, sha256 `2827ad9c5c0b835808225dcd28efb6d87c3a3e03ace74eedcb16cb2c881a4d41`, built with `scripts/build-wheel.sh` and installed in a copy of the live venv. Loaded libmlx sha256 prefix `4eaa7a692ef2a935`. The branch was then rebased onto `9959a9a8` without conflicts; the route code is unchanged, and #293's edits to `primitives.cpp` are elsewhere in the file.
- Deployed arm: the same wheel with `MLX_OMARCHY_QMV_SMALL_T` unset. Candidate arm: `MLX_OMARCHY_QMV_SMALL_T=6`. The flag is read live, so one build serves both arms.
- Model: `SiddhJagani/Qwen3.8-2B-mlx-4Bit@0867d98bfb174b042d88461c0e7c97b86b34b381` (docs/parity-contract.md). `model-sha256.txt` holds the weight, tokenizer and config hashes; they match the `model_sha256`, `tokenizer_sha256` and `config_sha256` of `receipts-work/vocab-prune/out/contract-ctl-r1.json`.
- Prompts: the ten contract prompts, taken as token ids from that record (pass 0). The prompt text is not in the tree.
- Teacher: 512 free-running greedy tokens per prompt on the deployed route (`teacher.json`, 5,120 positions). 8 of 10 prompts reproduce the record's 32 output ids exactly; prompts 1 and 4 diverge at positions 14 and 19 (G13G record from an older wheel).
- Reference: stock upstream MLX 0.32.3 (`mlx` and `mlx-cpu` from PyPI) with stock mlx-lm 0.31.3, CPU stream, fp32 (`model.set_dtype(float32)`; the CPU quantized matmul dequantizes in fp32), whole text in 64-token chunks. The omarchy CPU backend lacks `rms_norm`, `rope` and `sdpa`, so it could not serve as the oracle.
- Every arm runs with `MLX_OMARCHY_FULL_LOGITS=1` so the last-logits patch keeps every position.

## Rule fixed before the run

From docs/numerics-gate.md and docs/parity-contract.md section 4.2:

- The route only runs on passes of 2 to T max rows, and a one-pass prefill has none, so both GPU arms read the teacher-forced text in 4-token chunks (five runs each) and in 6-token chunks (one run each, a four-row and a two-row block).
- Check 1: per-op relative-L2 error against an fp64 numpy contraction of the dequantized weight and the bf16 x is no worse than the deployed route (`perop_error.py`: six shapes, T 2 to 8, 24 datasets each).
- Check 2: candidate teacher-forced top-1 agreement with the reference is at least the lower bound of the two-sided 95% Wilson interval on the baseline. A flip whose reference top-2 margin is at most 2 bf16 ULP at the reference top logit (`tools/dfuse/bf16_ulp.py`) is a near tie; raw counts are kept.
- Check 3: median candidate PPL deviation `abs(log(PPL / PPL_ref))` is at most the median baseline deviation plus the baseline noise (max minus min over five baseline runs).
- Harness check: S=1 teacher-forced PPL of the deployed route matches one-pass prefill PPL of the same text.
- Route: the trace must show the Batch4 kernel at T 2 to T max with the flag set, and the shipped kernels without it.

## Result

Route (`trace-off.txt`, `trace-on.txt`, kernel ids from `ComputeKernel`):

| rows | flag unset | `MLX_OMARCHY_QMV_SMALL_T=6` |
| ---: | --- | --- |
| 1 | `QmmVecQ4WordSubgroupBF16` (410) | 410 |
| 2, 4 | `CastBF16F32` (9), `QmmPrefillCoopmatM16BF16X32FullN` (449) | `QmmVecQ4WordSubgroupBatch4BF16` (547) |
| 6 | 9, 449 | 547 four rows, 547 two rows |
| 7 | 9, 449 | 9, 449 (above T max) |

Tests: `omarchy_qmv_batch_tests` on this device, 4 test cases and 34 assertions pass, 0 skipped (`doctest-qmv-batch.txt`, rerun after the cap below). The two new cases check every row against the single-row kernel bit for bit at T 2 to 4 over 24 datasets per cell, and the trace-proven engagement rule.

Check 1 (`perop-error.json`): the candidate mean error is no worse in 40 of 42 cells. In the other two (K 448, N 130, T 3 and T 7) it is higher by 1.5e-12 and 1.3e-12 on a mean of 1.7e-3, with the same maximum. The median candidate/deployed ratio over all cells is 0.99999997.

Checks 2 and 3 (`gate.txt`, `gate-summary.json`), 5,120 positions:

| arm | chunk | runs | top-1 vs reference | flips | near ties | PPL | mean abs dlogprob |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deployed | 4 | 5 | 99.141% | 44 | 44 | 1.38878 | 0.00899 |
| candidate | 4 | 5 | 99.082% | 47 | 47 | 1.38775 | 0.00885 |
| deployed | 6 | 1 | 99.141% | 44 | 44 | 1.38877 | 0.00899 |
| candidate | 6 | 1 | 99.082% | 47 | 47 | 1.38775 | 0.00885 |
| reference (CPU fp32) | 64 | 1 | | | | 1.38817 | |

- Check 2 passes: every flip in both arms is a near tie, so agreement with near ties is 100% for both; the baseline Wilson lower bound is 99.925%. Raw exact agreement is 3 positions lower for the candidate.
- Check 3 passes: PPL deviation 0.000307 (candidate) against 0.000434 (baseline); baseline noise 0, since the five runs of each arm are identical.
- Harness check (`validate.json`, prompts 0 and 1): S=1 PPL 1.42504, one-pass PPL 1.42546, relative difference 2.9e-4.
- Old against new (`compare_tf9.txt`, information only): 5,094 of 5,120 top-1 the same (99.49%); 26 differences, largest gap 2.0 bf16 ULP, mean 0.73.

## Revision: cap at 4

The G13G A/B on PR #296 found the route faster at 2 to 4 rows and slower from 5, where a second GEMV block costs more than the coopmat tile. The flag now accepts 2 to 4 (larger values count as 4), and a pass runs as one Batch4 dispatch. The gate above still covers the capped route: every pass in the 4-token chunk runs is at most 4 rows, so those runs dispatch the same kernel with the same operands under the cap. A 6-row pass now takes the coopmat tile, so the 6-token chunk runs no longer exercise the route; they are kept as measured. The same holds for the T 5 to 8 cells of check 1.

## Files

- `run-gate.sh`: the gate as run (paths are the operator's layout). `qwen38_chunked_gate.py` (teacher, chunked measure, harness check), `analyze_gate.py`, `perop_error.py`, `trace_small_t.py`.
- Results: `gate.txt`, `gate-summary.json`, `perop-error.json`, `validate.json`, `compare_tf9.txt`, `trace-off.txt`, `trace-on.txt`, `doctest-qmv-batch.txt`, `model-sha256.txt`, `teacher.json`.
- The per-arm records (about 240 KB each) and the reference record are kept by the operator, not committed.
