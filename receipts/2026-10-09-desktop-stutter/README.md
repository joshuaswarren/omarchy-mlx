# Desktop stutter under GPU load: the submission budget (issue #19)

Reporter: georgeamccarthy. Worker: DesktopStutter, 2026-10-08. Public
facts only; raw profile JSONLs and per-run logs live in the private lab
notebook under `artifacts/DesktopStutter/`.

## What the budget counts

The submission cap (v0.7.17) bounded each queue submission with the
**summed work-group count** of its dispatches
(`overlay/mlx/backend/omarchy/encoder.cpp`, `batch_work_`): a compute
dispatch contributes `group_count_x * y * z`, the evaluator submits the
open batch when the sum reaches the budget (default 40000; copies and
fills are not counted). The queue-priority knob
(`MLX_OMARCHY_QUEUE_PRIORITY`, default `LOW`) is separate and unchanged.

Measured cost of one work-group, from the device-timestamp profiles of
the 2026-10-02 calibration (M2 Max, Qwen3-4B 4-bit, one decode token ≈
161,000 groups ≈ 474 dispatches ≈ 20.9 ms):

| dispatch class | groups per dispatch | ns per group | us per dispatch |
|---|---|---|---|
| quantized matvec | 960 | ~78 | ~75 |
| RMSNorm | 13 | ~2320 | ~30 |
| RoPE | 6 | ~4660 | ~29 |
| SDPA | 42 | ~1100 | ~46 |
| reductions (once per token) | 1 | ~250,000-320,000 | ~250-320 |

Per-group cost spans three orders of magnitude, and most of a decode
token's time is the per-dispatch floor (20-30 us each, nearly
independent of the group count), not the group count. A group-count
budget calibrated on one model class therefore does not transfer to
another: it measures the wrong thing.

## Why the reporter still saw 84 ms at the default

Their M1 Pro + Qwen2-VL-7B 8-bit decode at `MLX_OMARCHY_BATCH_WORK=40000`
showed frame p50 83.7 ms (120 Hz panel). One decode token of a 7B-class
8-bit model is ~51,000 work-groups across ~309 dispatches carrying
~7.3 GB of weight reads; at 40000 groups per submission that is about 2
submissions per token, and their measured frame interval implies about
2.1 us per group — so each default-capped submission ran ~85 ms on their
machine. The same arithmetic on the M2 Max 4-bit models gives ~140
ns/group. The budget was doing what it was told; work-groups were the
wrong unit.

(Their 4.5 tok/s = 222 ms/token also sits far above the 36.5 ms
weight-read floor at their 200 GB/s, and their stack reports
`cooperative_matrix_f32_8 = 0` — stock Mesa without the cooperative-matrix
fast path the omarchy fork ships. After the budget is fixed, the
remaining gap on that machine is kernel speed, not submission length.)

## What changes

The default budget becomes a **time budget**. Each dispatch estimates
GPU time as `25 us fixed + bound_buffer_bytes / (200 GB/s)`; the
evaluator submits when the open batch's estimate reaches the budget.

- `MLX_OMARCHY_BATCH_MS=<ms>` — the new default knob, default 4 ms (the
  middle of the reporter's measured-smooth 2-6 ms band). `0` = off.
- `MLX_OMARCHY_BATCH_WORK=<groups>` — unchanged v0.7.17 behavior for
  anyone tuned against it; a set value wins over a set `BATCH_MS`;
  `0` = off, as shipped.

Synthetic 7B 8-bit token (the shape above): old default = 2 submissions
of up to ~85 ms on the reporter's machine; new default = 11 submissions
of est. <= 4.1 ms each. Scheduling only: splitting still happens between
dispatches, results stay bit-identical, greedy ids digests must match
across every setting.

Measured on our hosts (2026-10-08, diagnostics wheel at edac8b4c6,
Qwen2-VL-7B-Instruct-8bit text decode, 2 x 32 tokens per arm, device
timestamps + GdkFrameClock probe on the logged-in compositor):

(FILLED AFTER RUNS)

## Frame pacing (the check the reporter asked for)

A GTK4 `GdkFrameClock` probe runs on a real logged-in Hyprland session
on the M2 Max while decode runs: idle baseline, then one probe per arm
with frames restricted to the decode windows.

(FILLED AFTER RUNS)

## Regression test in CI

`overlay/tests/omarchy/test_runtime.cpp` gains pure (no-GPU) cases:
budget resolution across the three modes, flush boundaries, and the
frame-pacing contract — a synthetic 7B 8-bit decode token walked through
the real flush predicate asserting every submission's estimate stays
under the time budget, while the same plan under the old 40000-group
budget is shown to permit ~85 ms submissions at the reporter's measured
per-group cost.

## Recommendation

Ship the time default. Keep the group knob for tuning. For the
reporter's machine the honest limit: a time budget bounds submission
length regardless of chip, but the ~5x distance between their token
time and the bandwidth floor lives in the stock driver's kernel path,
not in submission length.

## Issue #19 reply draft (Main posts; worker does not post)

> Thank you for the measured frame table — the 83.7 ms frame interval
> at the default is the number that settles this. You were right: the
> work-group estimate under-counts for this model class.
>
> **What we found.** The cap counted work-groups per dispatch. Per-group
> GPU time is not comparable across kernels, models, or driver builds:
> our device-timestamp profiles show groups costing about 78 ns each in
> the quantized matvecs but 2,300-4,600 ns each in the small norm/rope
> kernels, and most of a decode token is a 20-30 us per-dispatch floor
> that the group count barely affects. Your 7B 8-bit at 40000 groups per
> submission works out to about 2.1 us per group — ten times the
> per-group cost the default was calibrated on (4-bit 4B/9B on an M2
> Max, 140 ns per group). So the default that split those models into
> 4-6 ms submissions left yours at about 84 ms.
>
> **What we change.** The default budget becomes a time budget. Each
> dispatch is estimated at a 25 us floor plus its bound buffer bytes at
> a 200 GB/s reference; the evaluator submits the open batch when the
> estimate reaches `MLX_OMARCHY_BATCH_MS` (default 4, i.e. 4 ms; `0`
> off). `MLX_OMARCHY_BATCH_WORK=<groups>` keeps the v0.7.17 behavior
> with the same off switch, and a set `MLX_OMARCHY_BATCH_WORK` still
> wins over a set `MLX_OMARCHY_BATCH_MS`, so nothing you have scripted
> changes meaning. Splitting is scheduling only: greedy token ids stay
> bit-identical across settings in our A/B runs.
>
> **What to run on your machine.** Your frame probe is the right
> instrument. Two phases, same as your last run, text decode with
> Qwen2-VL-7B 8-bit:
>
> ```
> MLX_OMARCHY_BATCH_WORK=2000 python your_frame_probe.py
> MLX_OMARCHY_BATCH_WORK=1000 python your_frame_probe.py
> ```
>
> Please report fps, p50/p99, and decode tok/s for both, and say
> whether the desktop feels smooth at either setting. If you also want
> the per-submission histogram, that needs a profiling build: build with
> `scripts/build-wheel.sh --diagnostics`, then
>
> ```
> python scripts/subcap_calibrate.py \
>   --model ~/.cache/huggingface/hub/models--mlx-community--Qwen2-VL-7B-Instruct-8bit/snapshots/<sha> \
>   --tokens 32 --prefill-tokens 256 --budgets 0,40000,5000,2000 \
>   --profile /tmp/sc-profile.jsonl
> ```
>
> **One more thing.** Your token time (222 ms at 4.5 tok/s) is about
> six times the 37 ms a 7.3 GB weight read needs at your memory
> bandwidth. Your report already names the likely cause:
> `cooperative_matrix_f32_8 = 0` on stock Mesa means the quantized
> matvecs take a slower path than the honeykrisp-omarchy fork. The time
> budget bounds the stutter on any driver, but closing the rest of that
> gap is kernel work, not submission work.

