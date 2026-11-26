# M6 (G18G) submission budget: 5000 to 40000

Date: 2026-11-13. Board: Apple M6 (G18G B1), Linux kernel `7.1.12-aurora-m6usb-ARCH #23`, GPU driver module sha256 prefix `3f192f53`, boot id `c8ea6c57-5d61-408b-8722-81e179f21a4d`, one boot for every run in this directory. Kernel command line carries `apple_mailbox.ascwrap_v8_cpu_control=1`. The GPU firmware is the one the kernel module drives in its G18 app-job profile; no separate firmware file exists on the board to hash.

## Verdict

`kBatchWorkCalibration` gets the row `{"G18G", 40000}`. Before the row, `Apple M6 (G18G B1)` matched no table entry and took the 5000 fallback (`kBatchWorkFallback`).

| Model | 5000 (fallback) | 20000 | 40000 | 40000 against 5000 | 40000 against 20000 |
|---|---|---|---|---|---|
| Qwen3.8-2B 4-bit | 3.934 tok/s | 5.011 tok/s | 5.083 tok/s | +29.2% | +1.4% |
| Qwen3-4B 4-bit | 2.712 tok/s | 3.605 tok/s | 4.031 tok/s | +48.6% | +11.8% |

Each cell is the mean of three leg means (legs interleaved, one fresh process per leg). 20000 is +27.4% (2B) and +32.9% (4B) against 5000.

Output tokens are equal in every arm: token hash `59811541b917f1a8` on every rep of every 2B leg, `0ebd8e39ba53323a` on every rep of every 4B leg (hash of the 128 greedy token ids, 512-token prompt).

## Cause

`batch_work_budget()` in `overlay/mlx/backend/omarchy/encoder.h` looks the device name up in `kBatchWorkCalibration`. The table had rows G13S, G13C, G13G and G14C. `Apple M6 (G18G B1)` contains no row key, so the lookup returned 0 and the budget became the 5000 fallback. Per decode step on this board, with the same dispatch count:

| Per decode step | 2B at 5000 | 2B at 40000 | 4B at 5000 | 4B at 40000 |
|---|---|---|---|---|
| submits | 12 | 2 | 20 | 3 |
| compute dispatches | 235 | 235 | 333 | 333 |
| barriers emitted | 232 | 223 | 315 | 298 |
| barriers skipped | 14 | 14 | 38 | 38 |

Run `runs/bcwk6.out` (six processes per model, counters from `scripts/barcount.py`). The G14C, which has the 40000 row, shows 2 (2B) and 3 (4B) submits per decode step.

## Decode rate, no frame probe (`runs/mlxwk2b.out`, `runs/mlxwk4b.out`, `runs/json/wk-*`)

Command: `scripts/m6-mlxdecode-wk.sh 2b 70` and `scripts/m6-mlxdecode-wk.sh 4b 60`. Legs run in the order default, 20000, 40000, repeated three times. Each leg is one new Python process: model load, warm-up, then repeated 128-token greedy decodes from a 512-token prompt for the leg's time budget. The rate is 128 divided by the time after the first token (prefill excluded). `MLX_OMARCHY_BATCH_WORK` is unset for the default arm and set to 20000 or 40000 for the others. The 2B legs held 2 (default) and 3 (20000, 40000) reps, the 4B legs held 2 reps each.

| Model | Arm | Leg 1 | Leg 2 | Leg 3 | Mean |
|---|---|---|---|---|---|
| 2B | default | 3.945 | 3.937 | 3.920 | 3.934 |
| 2B | 20000 | 4.995 | 5.024 | 5.015 | 5.011 |
| 2B | 40000 | 5.097 | 5.061 | 5.091 | 5.083 |
| 4B | default | 2.714 | 2.704 | 2.719 | 2.712 |
| 4B | 20000 | 3.592 | 3.623 | 3.601 | 3.605 |
| 4B | 40000 | 4.028 | 4.022 | 4.042 | 4.031 |

Standard deviation inside a leg is 0.003 to 0.037 tok/s. The first 2B default leg overlapped a 10 second read of the 2B weight file for hashing (CPU and disk only); its reps read 3.935 and 3.955, in line with legs 2 and 3. In `runs/mlxwk2b.out` and `runs/mlxwk4b.out`, the first `rep` line of the first default leg is overwritten by the queue's timeout-counter line; `runs/json/` holds the complete per-rep rates and hashes.

## Frame probe beside decode (`runs/mlxab2b.out`, `runs/mlxab4b.out`)

`scripts/m6-mlxdecode-ab.sh`: the 60 Hz stand-in client (`scripts/submit-latency.c` (in this repository)) submits one 256-byte fill command buffer every 16.6 ms and times the latency from submit to fence, while decode runs. Two legs per arm; the probe runs 90 seconds, starting 5 seconds into each 120 second leg. Probe alone for 20 seconds (1196 frames) gives p50 1.00 to 1.12 ms, p99 1.25 to 1.28 ms, max 4.2 to 4.6 ms, 0 frames over 33.3 ms.

| Model | Arm | Frames over 33.3 ms (leg 1 / leg 2) | p50 ms | p99 ms | max ms |
|---|---|---|---|---|---|
| 2B | default | 51 / 58 of about 5150 | 7.26 / 7.12 | 32.7 / 35.7 | 80 / 45 |
| 2B | 20000 | 50 / 57 | 7.65 / 7.96 | 32.3 / 34.8 | 94 / 84 |
| 2B | 40000 | 53 / 53 | 7.96 / 7.73 | 36.9 / 36.2 | 133 / 102 |
| 4B | default | 93 / 94 of about 5030 | 6.07 / 5.96 | 78.7 / 79.1 | 113 / 136 |
| 4B | 20000 | 45 / 45 | 7.58 / 7.56 | 27.5 / 27.6 | 164 / 178 |
| 4B | 40000 | 45 / 45 | 7.23 / 7.66 | 27.0 / 27.3 | 213 / 165 |

About 1% of probe frames exceed 33.3 ms in every 2B arm, including 5000. On 4B the 5000 arm has twice the frames over 33.3 ms and three times the p99 of the larger budgets. The maximum latency grows with the budget (2B: 80 / 45 ms at 5000, 133 / 102 ms at 40000; 4B: 113 / 136 ms at 5000, 213 / 165 ms at 40000). The maximum is one sample per leg.

The probe also slows decode: with the probe running, 2B decode is about 9% slower at 5000 and about 17% slower at 40000. The tok/s table above therefore uses the probe-free runs only. This board has no per-submission timestamp wheel, so the 2 to 6 ms submission band used by the other rows was not measured; the row follows decode rate.

## Provenance

| Item | Value |
|---|---|
| Wheel | `mlx_omarchy-0.32.4.dev202611131122+nocpu.pr328dtc-cp314-cp314-linux_aarch64.whl`, sha256 `99423bca436bfd2270caaed34019d7faa4aaa33ca6cb9ab42c9a416406f57df9` |
| Vulkan device | `Apple M6 (G18G B1)`, driver Honeykrisp, Mesa 26.1.4, Vulkan 1.4.354 |
| ICD library | `libvulkan_asahi.so` sha256 `ee5fbac9500bfbf71c09a0d20153e9cad66712a0a2337580b85de2bac4fb2b6d` |
| Qwen3.8-2B 4-bit | `SiddhJagani/Qwen3.8-2B-mlx-4Bit` revision `0867d98bfb174b042d88461c0e7c97b86b34b381`, `model.safetensors` sha256 `b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863` |
| Qwen3-4B 4-bit | `mlx-community/Qwen3-4B-Instruct-2507-4bit` revision `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`, `model.safetensors` sha256 `2a73c6c248601ab904e035548abd8e6abb65ea27dcb5f342fb0a8910eb44173f` |
| Device reopen | every leg opens the device in a new process and exits 0; zero new GPU timeout, watchdog, recovery or failed-submit kernel messages across `mlxwk2b`, `mlxwk4b`, `mlxab2b`, `mlxab4b` and `bcwk6` (`fail_lines_new=0` in each log) |
| Thermal | idle between rows, no temperature log; legs interleaved so drift lands on every arm |

Wheel source commit: `3e0868b5f8cf7fa67091699c697b6a087b65c13e` (`nocpu` wheel recipe). `encoder.h` at that commit has the same `kBatchWorkCalibration` table as the base of this change (`4fb694ed`), so the 5000 arm is the unchanged table behaviour on this board.

## Test

`overlay/tests/omarchy/test_runtime.cpp` checks that `calibrated_batch_work("Apple M6 (G18G B1)")` and `batch_work_budget("Apple M6 (G18G B1)")` return 40000 and differ from `batch_work_fallback()`. Before the row the lookup returned 0 for this name. The other boards and llvmpipe keep their values (checked with the table and lookup extracted from `encoder.h`).
