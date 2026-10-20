# bf16 A_log GatedDeltaUpdate decode: the untiled prefetch kernel is also the default for Hv == 3 * Hk on G14C (2026-10-20)

On a Vulkan device whose name contains G14C (Apple M2 Max), a model with a bf16 A_log, Hk != Hv and Hv == 3 * Hk (Qwen3.8-27B: 16 key heads, 48 value heads) now decodes with GatedDeltaDecodeBF16Pf (the untiled prefetch kernel) instead of the tiled GatedDeltaDecodeBF16. G13C already has this default for Hv == 2 * Hk and Hv == 3 * Hk. G14C at Hv == 2 * Hk was not measured and keeps the tiled default, as does every other part and head ratio. MLX_OMARCHY_GDN_DECODE_TILE=1, MLX_OMARCHY_GDN_PF=0 and MLX_OMARCHY_GDN_GQA_BF16A=1 each keep the earlier kernels.

The rule is `gdn_bf16a_untiled_default(device, hk, hv)` in `device.h`; it replaces the earlier part and shape pair.

The two kernels are not bit-equal. A pre-registered test on this chip (first-ab/PREREG-g14c-27b.md, written before the first valid run) set the bar: faster in at least 4 of 5 rounds on both prompts with a median change of at least 2 percent, and teacher-forced NLL not worse by more than 0.02 nats at 2 SE. Both were met (see "First A/B").

## Machine

14-inch M2 Max, Vulkan device Apple M2 Max (G14C B1), kernel 7.1.12-2-20261013.18-sep-ARCH, Mesa 26.3.0-devel (git-a5b1797446), driver library prefix c499962c742f399a, performance governor. Firmware identity read from the device tree after the runs (vendor prefix of the property names dropped): compatible apple,j414c / apple,t6021; iboot1-version mBoot-20457.1.29; iboot2-version iBoot-8422.141.2; os-fw-version 13.5; system-fw-version 27.0. The runs did not print them.
Model mlx-community/Qwen3.8-27B-4bit snapshot 10c35caafbb80f7dc6a7a432cdd11af10a6d4818, model.safetensors.index.json sha256 13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8 (the same index as on the 16-inch M1 Max; weight files not hashed).

## First A/B on one wheel (first-ab/)

Wheel mlx_omarchy-0.32.4.dev202610200902+f334420 (main f33442078, sha256 d05326b0835cf47b8a21fc4cf9c1f9f22e8813a00feb1963f67eb79bd4933d8c, libmlx prefix 1a2af63453eeb246), default against MLX_OMARCHY_GDN_DECODE_TILE=0, boot 98f40d84, GPU-timeout delta 0 on every ticket (first-ab/*.ticket.out). A first launch used a general serving environment whose mlx-lm has no Omarchy patches; it ran generic ops (census 63511 dispatches, no GatedDeltaDecode kernel) and produced no result. The amendment in the pre-registration records this. Those outputs stayed on the machine that ran them, moved aside under failed-out-* names; they were not copied into this directory.

- Census: default kernel 455 x528 per 10 tokens, TILE=0 kernel 480 x528, totals 7491 and 10283.
- Speed (decode tok/s, 64 tokens, 5 rounds, fresh process each, order alternating; every row verified=match):

| Prompt | Default median | Untiled median | Change | Untiled faster | Per round |
|---|---:|---:|---:|---:|---|
| P1 | 15.06 | 15.98 | +6.11% | 5/5 | +5.84 +6.11 +6.10 +6.39 +5.64 |
| P2 | 15.07 | 15.96 | +5.91% | 4/5 | +5.71 +5.83 +5.70 +5.91 -5.11 |

The P2 round 5 untiled sample read 14.30 tok/s; the other nine untiled P2 rows are 15.91 to 15.97. The rule is met with or without that sample. That round was not rerun.
- NLL: 128 prefixes of a fixed text, 8192 positions per arm. Untiled minus default +0.00107 nats per token, cluster SE 0.00062, interval at 2 SE [-0.00018, +0.00232] (first-ab/nll-cmp.log), below the +0.02 line. All five score arrays are byte-identical (same sha256) to the arrays of the 16-inch M1 Max run (receipts/2026-10-20-gdn-bf16a-ratio3/nll/), made on a different chip and build, so this NLL adds no independent evidence beyond that identity. A repeat of the default arm equals the first.

## This change on the new wheels (pr-check/)

N = mlx_omarchy-0.32.4.dev202610201405+7076d7a (this change at 7076d7a5, sha256 d0d205ed5b4067822ff4ad7d5c42954db6c0555cf3dc89d91abd18c87664641e, libmlx prefix a32e5cdf868fb03e). O = mlx_omarchy-0.32.4.dev202610201359+c163e54 (main c163e54a, sha256 730553fabd04f26dabbe30c374a7037149d2cd12354a344d43b09077dbbbdbcb, libmlx prefix 0c3f6d711e49914d). Test binary sha256 597311bc8ba70416f186208d7410d3635522b2dae2fcda636380514dcb34fc6c. Same boot, mlx-lm with identical patches in both environments (gated_delta.py prefix 0d5cd194449cd36e), no GDN, MLX, HK or AGX variable in the default arms, every row verified=match.

- Device test (pr-check/run-test.log, test-all.out): 8 of 8 cases, 121 of 121 assertions on Apple M2 Max (G14C B1), including the named (part, ratio) checks. The ticket file of this stage was not kept; the GPU-timeout delta is 0 on the other five tickets.
- Census: N default kernel 480 x528 per 10 tokens, N with TILE=1 kernel 455 x528, totals 7491 and 10283.
- Numerics (128 greedy tokens, full log-probability vector of 8 steps, both prompts): N default equals O with TILE=0 bit for bit, and N with TILE=1 equals O default bit for bit (max absolute log-probability difference 0.0, 0 differing elements, KL 0.0).
- Speed, two distinct builds, default environment in both (tools/sum_r3_speed.py pr-check; 64 tokens, 5 rounds, order alternating):

| Prompt | O median | N median | Change | N faster | Per round |
|---|---:|---:|---:|---:|---|
| P1 | 15.070 | 15.980 | +6.04% | 5/5 | +6.32 +6.32 +5.24 +5.97 +6.03 |
| P2 | 15.060 | 15.960 | +5.98% | 5/5 | +5.84 +6.03 +5.86 +5.84 +6.05 |

## Hardware record

- Source commits: 7076d7a5 (N, this change), c163e54a (O, main), f33442078 (first A/B wheel).
- Kernel, Mesa, driver library, Vulkan device, firmware identity and model hashes: "Machine" above. Numerical result and backend dispatch trace: the sections above (census histograms in first-ab/ and pr-check/hist-*.txt).
- Commands: tools/ (verify-g14c.sh with stage test | census | numP1 | numP2 | speedA | speedB, run-m2.sh, wheel-r3.sh).
- Timing and thermal procedure: every run set ran on one boot (98f40d84) as queued tickets of the lab GPU scheduler, with the performance governor, one fresh process per sample, arm order alternated by round, and one warm-up as the first load of each process. No temperature, fan or throttle reading was taken. The GPU-timeout counter was read before and after every ticket of the first A/B (8 tickets) and of the speed, numerics and census stages (5 tickets): delta 0 on all 13. The device-test ticket file was not kept.
- Vulkan device reopen: no dedicated close-and-reopen test was run. Every run is a fresh process that opened the device from the start. All 54 exit codes in the run logs (first-ab/ and pr-check/ run-*.log) are 0, and every timing, census and numerics row prints verified=match.

## Limits

One chip (G14C), one measured model here (Qwen3.8-27B-4bit), one text, a 512-token NLL window, two 20-token speed prompts. G14C at Hv == 2 * Hk and Hv == 4 * Hk are not measured and keep the tiled default. The NLL is a not-worse check, not a quality benchmark. The first A/B and the NLL ran on the wheel from main f33442078, the two-build check on the new wheels; both hold the same kernels.

## Files

- first-ab/: PREREG-g14c-27b.md, run-m2.sh, out-*/ (census, timing and NLL parts, without the *.npy log-probability arrays), nll-cmp.log, ticket files.
- pr-check/: run-*.log per stage, t-*.out and t-*.err timing runs, num-*/, cen-*/, hist-*.txt, test-all.out, test-list.out, ticket files.
- tools/: verify-g14c.sh (stages test, census, numP1, numP2, speedA, speedB), run-m2.sh, wheel-r3.sh, sum_r3_speed.py and the readers. SHA256SUMS lists every file in this directory. The *.npy arrays (about 8 MB each) are not kept; their digests are in the run logs.
