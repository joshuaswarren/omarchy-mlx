# Honeykrisp submit poll: the 2000 us default costs CPU and power and buys no speed

The backend set `HK_SUBMIT_POLL_US=2000` at device init (commit c94baca86). With that value Honeykrisp's `vk_drm_syncobj_wait_many` polls `DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT` with a zero timeout. It does that for up to 2 ms before it blocks. A poll that sees the signal resets the back-off. So every wait that ends inside 2 ms polls again at the next wait.

## What the poll does in a decode

An `LD_PRELOAD` shim on `ioctl` (tools in the research repo, `tools/timeline-spy`) on a 13-inch M1, Qwen3.8-2B 4-bit decode with the default, on a wheel built from main plus an unrelated flag:

- 21,000 `SYNCOBJ_TIMELINE_WAIT` calls per decoded token return `ETIME`. The number comes from two runs 128 tokens apart. Load and prefill cancel.
- Every sampled call has a zero deadline. The sample spans the whole run. It holds 9,000 calls and 1,215 distinct waits over timeline points 1 to 1,235.
- A Vulkan-only program that submits work and calls `vkWaitSemaphores` blocks with one ioctl per wait. So the loop is the driver poll, not the application.

## Decode: CPU and power, 13-inch M1

2B 4-bit, 400 tokens, fresh process per run. The arms alternated A B C three times. The generated ids were identical in all nine runs. A is the old default, B is `HK_SUBMIT_POLL_US=0`, C is 200. Power comes from the hwmon Total System, Heatpipe and AC input sensors, sampled every 50 ms over the decode window. Idle Total System Power was 6.92 W.

| Arm | tok/s | spread | process CPU s | Total System W | Heatpipe W |
| --- | --- | --- | --- | --- | --- |
| A, 2000 us | 44.73 | 0.5% | 9.66 | 21.77 | 16.55 |
| B, 0 | 44.75 | 0.1% | 4.28 | 20.73 | 15.78 |
| C, 200 us | 44.65 | 0.1% | 4.98 | 21.12 | 15.67 |

The 2000 us poll costs 5.4 s of process CPU over a 400-token decode. It also costs about 1 W of system power and 0.8 W of package power. Speed is the same within the spread.

## Parakeet transcribe, the workload the poll was tuned for

`mlx-omarchy-parakeet transcribe` on its packaged fixture, installed from the v0.7.37 draft. The number is the `tdt_decode` stage wall from `transcribe-report.json`. Four runs per arm.

13-inch M1, medians: A 120.7 ms, B 119.1 ms, C 118.7 ms. Process CPU 2.56, 2.48 and 2.49 s.

16-inch M1 Max, AGX vote on, two runs of the row. Here `tdt_decode` is bimodal: about 93 ms or 224 ms. The arm does not decide which. Run 1 (unpinned): A 223, 151, 93, 224; B 223, 224, 224, 94; C 95, 224, 224, 224. Run 2 (pinned to cores 2 to 9): A 224, 95, 91, 224; B 224, 224, 98, 92; C 224, 224, 224, 96. Inside one mode the three arms are equal. Fast mode is 91 to 98 ms and slow mode is 223.7 to 224.2 ms. I did not find what picks the mode, and pinning to the P-cores did not remove it.

So the poll does not help this workload on either machine in the product configuration. The earlier 341 to 254 ms figure (mesa-1 commit d597ce2a1145) was taken before the AGX vote existed. It is not reproduced here.

## Change

The backend no longer sets `HK_SUBMIT_POLL_US`. A value in the environment still reaches the driver unchanged. The new test was written first. It failed on llvmpipe with `2000 == (unset)`.

## Not done

- The 16-inch M1 Max decode CPU and power row is queued and has not run. This receipt gets its numbers when it does.
- The Honeykrisp side (a poll that backs off after a hit that took most of the budget) is not changed.
