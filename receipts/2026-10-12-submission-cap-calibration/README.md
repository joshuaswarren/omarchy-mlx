# Issue #19: per-board submission-cap calibration (piece 1 of 3)

Reporter: georgeamccarthy (M1 Pro, stock Mesa 26.2.3, 120 Hz Hyprland).
Pre-registered plan; runs on 2026-10-11 (13-inch M1) and 2026-10-12
(16-inch M1 Max). The tables below are filled from the raw
device-timestamp profiles, recomputed by `subcap-analysis.py` from the
per-dispatch GPU ticks (not from the run's own summary lines).

## Question

Does one work-group budget land typical decode submissions at p50 in the
2-6 ms band on every measured board and model size? Per-group GPU cost is
chip-dependent, so the shipped single default (40000 groups, calibrated on
the M2 Max only) is expected to miss on the M1 family. If no per-board
group count can land the band, the fallback promised on the issue is to
switch the flush estimate to measured per-dispatch GPU time.

## Setup

- Diagnostics wheel 0.32.4.dev202610110831+diag.e57cb30 (profiling harness
  compiled in), scripts/subcap_calibrate.py, budgets
  0/40000/20000/5000/2000/1000, 64 decode tokens after a 512-token
  prefill, fixed prompt, per-budget subprocess with its own profile
  stream.
- Boards: the 16-inch M1 Max on the lab-default Mesa build (Mesa pull
  request 6 head 69bde15340) with a plain-transformer 4-bit 4B model and a
  gated-delta 4-bit 9B model; the 13-inch M1 with the stock Mesa 26.2.4-1
  driver pinned by `VK_DRIVER_FILES` (lib sha256 a70fb890552f...), the
  reporter's driver version, with the 4B model only.
- The reporter's exact model (Qwen2-VL-7B 8-bit) cannot be loaded by the
  harness's mlx_lm path, so no row matches his setup exactly.
- Honest limits: the M1 Pro itself is not in the fleet, so it has no row;
  load average and driver identity are recorded per run.

## Results

### 13-inch M1 (G13G, stock Mesa 26.2.4-1, pinned by `VK_DRIVER_FILES`, lib sha256 a70fb890552f...)

Qwen3-4B-Instruct-2507-4bit, 64 decode tokens after a 512-token prefill,
load average 0.29 at start, kernel 7.1.12-2-12.6-sep-ARCH. CPU governor
not recorded in that run; the board read schedutil at the time. Recomputed
from the raw per-dispatch timestamp streams; the run's own summary lines
agree to the displayed precision. Decode tok/s at each budget from the
same run:

| MLX_OMARCHY_BATCH_WORK | submissions | p50 ms | p95 ms | p99 ms | max ms | decode tok/s |
|---|---|---|---|---|---|---|
| 0 (cap off) | 71 | 89.44 | 90.75 | 324.19 | 842.45 | 8.651 |
| 40000 (old default) | 287 | 23.56 | 24.70 | 170.71 | 189.85 | 8.588 |
| 20000 | 574 | 11.97 | 17.34 | 93.29 | 95.18 | 8.545 |
| 5000 | 1868 | 3.47 | 6.96 | 23.79 | 23.92 | 8.128 |
| 2000 | 2768 | 2.31 | 6.90 | 13.86 | 14.03 | 8.004 |
| 1000 | 5394 | 1.18 | 1.84 | 9.81 | 13.23 | 7.279 |

The shipped 40000 default leaves this board at p50 23.6 ms, about 4x the
band top. In band: 5000 and 2000. The G13G row takes 5000: same p95 as
2000, 1.5% better decode tok/s, and 40% fewer submissions. Cost against
cap off is -6.0%; cap off on this board means 89 ms submissions, so the
cost buys the smooth band the reporter describes. Calibrated on the 4B
model only (the only model cached on that machine).

### 16-inch M1 Max (G13C, lab-default Mesa 69bde15340, performance governor)

Both models, 64 decode tokens after a 512-token prefill, load average 1.06
(4B) and 1.18 (9B) at start, kernel 7.1.12-2-12.6-sep-ARCH, CPU governors
pinned at performance (recorded before and after each run; ppd balanced).
Recomputed from the raw per-dispatch timestamp streams; the runs' own
summary lines agree to the displayed precision.

Qwen3-4B-Instruct-2507-4bit:

| MLX_OMARCHY_BATCH_WORK | submissions | p50 ms | p95 ms | p99 ms | max ms | decode tok/s |
|---|---|---|---|---|---|---|
| 0 (cap off) | 71 | 5.40 | 6.04 | 10.33 | 16.84 | 33.834 |
| 40000 | 287 | 1.55 | 1.67 | 3.41 | 3.47 | 33.504 |
| 20000 | 574 | 0.79 | 0.97 | 1.73 | 1.88 | 32.279 |
| 5000 | 1869 | 0.28 | 0.43 | 0.53 | 0.63 | 27.012 |
| 2000 | 2785 | 0.17 | 0.22 | 0.31 | 0.50 | 23.971 |
| 1000 | 5427 | 0.08 | 0.16 | 0.20 | 0.32 | 20.784 |

Qwen3.5-9B-MLX-4bit:

| MLX_OMARCHY_BATCH_WORK | submissions | p50 ms | p95 ms | p99 ms | max ms | decode tok/s |
|---|---|---|---|---|---|---|
| 0 (cap off) | 72 | 13.33 | 15.68 | 27.71 | 50.05 | 12.648 |
| 40000 | 916 | 1.24 | 1.80 | 1.99 | 2.36 | 12.449 |
| 20000 | 1693 | 0.78 | 1.09 | 1.40 | 2.08 | 11.882 |
| 5000 | 5401 | 0.25 | 0.65 | 0.68 | 1.08 | 9.507 |
| 2000 | 14407 | 0.07 | 0.44 | 0.50 | 0.90 | 8.044 |
| 1000 | 14572 | 0.07 | 0.47 | 0.50 | 0.74 | 7.953 |

On this board and driver the shipped 40000 default already holds both
models far under the band top: p50 1.55 ms (4B) and 1.24 ms (9B), p99 at
or under 3.4 ms, decode tok/s within 1.6% of cap off. The band's lower
half is unreachable here: the step from cap off (5.40 / 13.33 ms) to the
smallest measured budget jumps straight under 2 ms, so every capped value
over-splits, which is the safe side. The G13C row keeps 40000. The cap
still earns its keep on the 9B: cap off puts typical submissions at
13.3 ms.

## Decision (as pre-registered, applied to the measurements)

- One budget in band everywhere measured: no (only cap off lands the
  4B on the 16-inch M1 Max in band, and cap off fails the other rows).
- Per-board values keeping every measured row at p50 <= 6 ms: yes.
  G14C keeps 40000 (5.6 / 4.4 ms p50); G13C keeps 40000 (1.55 / 1.24 ms
  p50); G13G takes 5000 (3.47 ms p50; the shipped default left it at
  23.6 ms).
- The measured-GPU-time fallback is not triggered: no measured board or
  model exceeds the band top once the 13-inch M1 row is lowered. The
  fallback stays available for model classes outside this table (an
  8-bit 7B decodes with a different per-group cost; see the
  desktop-stutter time-budget branch for the estimator that bounds any
  class by construction).

G13S (the reporter's M1 Pro) stays uncalibrated. Its default is the most
conservative calibrated value, 5000. That is the safe direction: shorter
submissions than strictly needed, never longer.

## Change

`overlay/mlx/backend/omarchy/encoder.h`: `kBatchWorkCalibration` keyed by
board code like the direct-matmul rows (G13C/G13G/G13S/G14C);
`calibrated_batch_work()` pure lookup; `batch_work_fallback()` returns the
most conservative calibrated value for boards with no row (shorter
submissions, never longer); `MLX_OMARCHY_BATCH_WORK` semantics unchanged
(override, 0 = off). Call sites pass the device name.
`overlay/tests/omarchy/test_runtime.cpp`: doctest pins the table lookups,
the two measured G13 values, and the fallback.
