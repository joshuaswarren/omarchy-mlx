# M4 Pro (G16S): submission budget 15000

`kBatchWorkCalibration` gains a G16S row of 15000. Before this, G16S had no row and took the 5000 fallback. No other board changes.

## Why the M4 Pro needs a row

The budget counts dispatch work groups. G16S decodes with the 2x4 Qmm column geometry, which launches 4x the work groups of 2x16. At 5000 a Qwen3-8B token splits into about 48 submissions (`MLX_OMARCHY_TRACE_DISPATCH`, tokens 10..40); at 30000 into about 10 (Qwen3.5-4B: 47.5 and 10.3). Each boundary costs a completion-to-launch round trip.

## Decode

`decode/ttft2.py` (128 greedy tokens, best of 2 per process), 5 interleaved rounds per arm, median, stock system ICD (Mesa 26.3.0-devel Borealis `05ddb00f5c`), this tree built as the Python core. Raw: `decode/ab-bw.txt`; `python3 decode/summ.py decode/ab-bw.txt b5`.

| budget | Qwen3-4B-Instruct-2507 | Qwen3-8B | granite-4.0-1b |
|---:|---:|---:|---:|
| 5000 | 66.09 | 42.87 | 113.70 |
| 10000 | 73.29 (+10.9%) | 45.02 (+5.0%) | 120.61 (+6.1%) |
| **15000** | **74.72 (+13.1%)** | **46.18 (+7.7%)** | **133.85 (+17.7%)** |
| 20000 | 76.12 (+15.2%) | 46.27 (+7.9%) | 137.92 (+21.3%) |
| 30000 | 76.39 (+15.6%) | 46.47 (+8.4%) | 136.71 (+20.2%) |

Greedy ids (3 prompts x 256 tokens, `decode/toks2.py`) are identical at 5000, 10000 and 30000 on all three models (`decode/tb-*.json`, `decode/toks-bw.txt`).

## Submission time on a release build

The profiling build inflates small dispatches (the per-dispatch timestamp mode adds a barrier per dispatch; its decode runs at 10-12 tok/s against 46-76 tok/s released). So the submission times here come from a release build, using the kernel DRM scheduler tracepoints `gpu_scheduler:drm_sched_job_{queue,run,done}` (`real/run-real.sh`). Each run is a ~500-token prefill plus 128 decode tokens (`real/reald.py`). vkcube presents at 120 Hz under the Wayland compositor (Hyprland) throughout. `busy` is `done - max(run, previous done of the same process)`, the time the submission held the compute ring. Raw traces: `real/*.trace.xz`; `python3 real/parse2.py` reproduces `real/summary.txt`.

All submissions, prefill included, busy ms p50 / p99 / max:

| budget | Qwen3-4B-2507 | Qwen3-8B |
|---:|---|---|
| 5000 | 0.83 / 27.2 / 28.0 | 1.05 / 52.6 / 54.2 |
| 10000 | 1.64 / 43.4 / 44.7 | 1.91 / 70.7 / 72.1 |
| **15000** | **2.46 / 27.8 / 28.5** | **2.94 / 53.9 / 54.7** |
| 20000 | 3.13 / 49.1 / 49.7 | 3.83 / 54.0 / 54.4 |
| 30000 | 4.73 / 49.4 / 49.8 | 5.91 / 93.3 / 94.6 |

Decode submissions only, p50 / p99 / max: at 15000 they are 2.45 / 2.89 / 2.91 ms (4B) and 2.92 / 4.45 / 4.51 ms (8B).

The prefill tail at 5000 is a single large dispatch (it runs whole, whatever the budget), so 5000 is the floor. 15000 is the only budget above 5000 whose tail stays at that floor on both models: 4B p99 27.8 ms against 27.2, 8B max 54.7 ms against 54.2. 20000 raised the 4B tail to 49.7 ms, and 30000 put an 8B submission at 94.6 ms, past the ~78 ms stutter of #19. 10000 merged two prefill dispatches into one 72.1 ms submission on 8B. Every budget is one run, so the exact split points are one sample.

Compositor side, same runs. vkcube frame interval (done to done) has p50 8.33 ms at every budget, max 11.4-17.7 ms, and no frame over 33.3 ms in any run. Frames over 16.7 ms: 3 at 5000 (8B), 2 at 10000 (8B), 0 elsewhere. Hyprland render jobs wait at most 6.6-11.0 ms from queue to done (idle 0.7-2.7 ms).

## Profiling harness, for comparison

`scripts/subcap_calibrate.py` unchanged, on a `-DMLX_OMARCHY_GPU_PROFILING=ON` build of this tree, 64 tokens, one process per budget (`harness/`). On Qwen3-4B-2507 it reads p50 / p99 / max 4.74 / 24.7 / 29.2 ms at 10000, 7.25 / 39.8 / 41.3 at 15000 and 14.6 / 65.7 / 85.2 at 30000. Its per-dispatch timestamps inflate every submission, which is why the row above is chosen on the release-build numbers.

## Geometry

`geometry/ab-c2.txt`: with the budget raised, 2x4 and 2x2 decode tie (Qwen3-8B +8.9% both, granite +11.8 / +12.9%, Qwen3.5-4B +6.3 / +6.0% against 2x4 at 5000), and 2x16 at 5000 is within 0.4-3.2%. So the geometry stays 2x4 (the current default) and only the budget changes.

## Stack

Apple M4 Pro (`Apple M4 Pro (G16S B1)`), Linux 7.3 (aurora-silicon/linux), Mesa 26.3.0-devel Borealis `05ddb00f5c` system ICD, Python 3.14, mlx-lm with the `--patch-venv` series.

## Tests

`overlay/tests/omarchy/test_runtime.cpp` pins the G16S row to 15000 by the exact device name. It also checks that `Apple M4 (G16G B1)` does not pick up the row.
