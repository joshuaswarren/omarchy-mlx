# Issue #19: per-board submission-cap calibration (piece 1 of 3)

Reporter: georgeamccarthy (M1 Pro, stock Mesa 26.2.3, 120 Hz Hyprland).
Pre-registered plan, run 2026-10-11 on two lab machines; the tables below
are filled from the raw device-timestamp profiles, recomputed by
`subcap-analysis.py` from the per-dispatch GPU ticks (not from the run's
own summary lines).

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

### 16-inch M1 Max (G13C, lab-default Mesa 69bde15340)

(FILLED FROM RECEIPTS)

### 13-inch M1 (G13G, stock Mesa 26.2.4-1, pinned)

(FILLED FROM RECEIPTS)

## Decision rule (written before the run)

- One budget in band everywhere measured: the single default stays.
- Per-board values in band but different: this per-board table is the
  default (`kBatchWorkCalibration` in the encoder).
- No per-board value lands the band for some board+model: switch the
  flush estimate to measured GPU time (the issue's stated fallback).

## Change

`overlay/mlx/backend/omarchy/encoder.h`: `kBatchWorkCalibration` keyed by
board code like the direct-matmul rows (G13C/G13G/G13S/G14C);
`calibrated_batch_work()` pure lookup; `batch_work_fallback()` returns the
most conservative calibrated value for boards with no row (shorter
submissions, never longer); `MLX_OMARCHY_BATCH_WORK` semantics unchanged
(override, 0 = off). Call sites pass the device name.
`overlay/tests/omarchy/test_runtime.cpp`: doctest pins the table lookups
and fails until the G13C/G13G rows carry their measured values
(failing-first).
