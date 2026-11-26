# bf16 A_log GatedDeltaUpdate decode: the untiled prefetch kernel is also the default for Hv == 3 * Hk on G13C (2026-10-20)

On a Vulkan device whose name contains G13C, a model with a bf16 A_log, Hk != Hv and Hv == 3 * Hk (Qwen3.8-27B: 16 key heads, 48 value heads) now decodes with GatedDeltaDecodeBF16Pf (the untiled prefetch kernel) instead of the tiled GatedDeltaDecodeBF16. PR 243 made this the default for Hv == 2 * Hk (Qwen3.6-35B-A3B). Every other part and every other head ratio (for example 16 and 64) keeps its previous default. MLX_OMARCHY_GDN_DECODE_TILE=1, MLX_OMARCHY_GDN_PF=0 and MLX_OMARCHY_GDN_GQA_BF16A=1 each keep the earlier kernels.

The two kernels are not bit-equal. The default therefore changed only after a paired teacher-forced NLL check passed the not-worse rule (see "Decision").

## Machine and builds

16-inch M1 Max, Vulkan device Apple M1 Max (G13C C0), kernel 7.1.12-2-12.6-sep-ARCH, driver library prefix 40282811c6e14ef9 (Mesa 26.3.0-devel, git-69bde15340). Model mlx-community/Qwen3.8-27B-4bit snapshot 10c35caafbb80f7dc6a7a432cdd11af10a6d4818.

- N (this change): mlx_omarchy-0.32.4.dev202610200429+2939e78, wheel sha256 567c99392954092f308cbd546968a6bb09b193d8def5be2de62260155bceeb06, libmlx prefix 451ccdea64b12f54. Built from commit 2939e781 on the head of PR 243. The pull request commit is the same change rebased on main (PR 243 is merged as 311efc83); the code of both source files that this change touches is the same on both bases.
- O (before this change): mlx_omarchy-0.32.4.dev202610200425+657582a, wheel sha256 f047c15b6dddb64daa6f56cb468f41c894ec69df09963bb136a240615453288c, libmlx prefix 55bd83b2450afce8. Built from 657582a3, the head of PR 243 and the parent of 2939e781. It equals main for this code path.
- Test binary of the pull request head 18b78977: sha256 62055c20fe26298f25644f14bd5193b53b089efec41c924de9e5c5a577a23a83 (built with its own wheel, mlx_omarchy-0.32.4.dev202610200607+18b7897, sha256 6c71aa29b77325a952a4475e649f34b25e1f352e683097ba06d4b84dc7ac10ef; that wheel was not used for timing).

Every timing and numerics row prints its provenance line with verified=match, with the build stamp of its own wheel. Timing ran as queued tickets of the lab GPU scheduler on one boot, with the performance governor; no GDN, MLX, HK or AGX variable was set in the default arms (ab-two-builds/run-*.log, line "env ... present: 0").

## Decision: paired teacher-forced NLL, accept-not-worse (nll/)

nll/PREREG-nll27.md was written before the scoring run (registered 2026-10-20). 128 prefixes of a fixed text, each a 512-token prefill and 64 decode steps, 8192 scored positions per arm. d = NLL(untiled) - NLL(tiled); the arm named default (D) in the logs ran the tiled kernel, because the wheel of this check predates the change; the unit of replication is the prefix. The wheel used for this check is mlx_omarchy-0.32.4.dev202610181348+49e6b3c1 (sha256 7654db6b519b221ab5b221e96e1bda3b28c6317b0969b78de2abf3d101f8fe59), with MLX_OMARCHY_GDN_DECODE_TILE=0 for the untiled arm; it holds the same kernels as N.
The dispatch census of that wheel (nll/census/hist-D-10.txt, hist-U-10.txt) shows kernel 455 for D and kernel 480 for U, 528 dispatches each per 10 tokens. The pre-registration file and the score arrays are in one commit, so their order is not checkable from git; the arrays were written after the pre-registration, the header carries only the date 2026-10-20. On the machine that ran the scoring, the pre-registration file's modification time was 22:01 local time (03:01Z) and the array files' was 23:06 (04:06Z); the first scoring ticket started at 03:09:05Z (nll/tickets/out-D0-1.ticket.out). These are file times, not signed timestamps.
Result (nll/nll-cmp.log, from the stored arrays with tools/nll_cmp.py): D (tiled) mean NLL 0.94625, U mean 0.94732, paired mean +0.00107 nats per token, cluster SE 0.00062, interval at 2 SE [-0.00018, +0.00232]. The upper end is below the +0.02 line, so the rule gives ACCEPT-NOT-WORSE. The untiled arm differs from the default on 7825 of 8192 positions and in 128 of 128 prefixes; 91 of 8192 argmax positions differ. A second default run on prefixes 0 to 31 (D2) equals the first on every position, so the scoring is deterministic.
This is a not-worse check on one text and one model. It is not a quality benchmark.

## This change on the new wheels (ab-two-builds/)

- Device test, pull request head 18b78977 (test2 stage, run-test2.log, test-all.out): 8 of 8 test cases, 115 of 115 assertions on Apple M1 Max (G13C C0). The new case prints "ratio 3: default kernel 480, 3 dispatches". The run-test2.log header line names the test binary of the first stage (dfcd4b86...); the stage ran the binary with sha256 62055c20... named above.
- An earlier run of the test binary built from 2939e781 (run-test.log) reported 7 of 8 cases and 117 of 119 assertions: the case "gdn raw decode f32 A_log: ..." still carried a block that asserted the previous default for the 16/48 bf16 shape (line 431: CHECK_EQ(480, 455); line 434: same(...) false). The stage marker stage-test.done exists although that stage had rc=1, because verify-r3.sh has no -e; the rc is in run-test.log. That block is removed in 18b78977; the dedicated bf16 case covers the shape. The output of that run was overwritten by the second run; the run-test.log line is kept.
- Census (run-census.log, hist-*.txt): N default dispatches GatedDeltaDecodeBF16Pf (id 480) x528 per 10 tokens, N with MLX_OMARCHY_GDN_DECODE_TILE=1 dispatches GatedDeltaDecodeBF16 (id 455) x528; every other histogram line and the totals (7491 and 10283 for 6 and 10 tokens) are equal.
- Numerics (run-numP1.log, run-numP2.log; 128 greedy tokens, full log-probability vector of 8 steps): N default equals O with MLX_OMARCHY_GDN_DECODE_TILE=0 bit for bit, and N with TILE=1 equals O default bit for bit, on both prompts (max absolute log-probability difference 0.0, 0 differing elements, KL 0.0, ids equal).
- Speed, two distinct builds, default environment in both (tools/sum_r3_speed.py ab-two-builds; 64-token runs, 5 rounds, fresh process each, order alternating):

| Prompt | O median tok/s | N median tok/s | Change | N faster | Per round |
|---|---:|---:|---:|---:|---|
| P1 | 14.080 | 14.900 | +5.82% | 5/5 | +5.82 +5.89 +5.83 +5.90 +5.89 |
| P2 | 14.080 | 14.890 | +5.75% | 5/5 | +5.90 +5.68 +5.97 +5.68 +5.68 |

P1 is "Explain in plain words how a bicycle works." P2 is "Write a short story about a lighthouse keeper."
The earlier switch inside one wheel (same-wheel-speed/, TILE=0 against default) gave +5.75% and +5.76%.

## Hardware record

- Source commits: 2939e781 (N wheel), 18b78977 (pull request head, test file fix), comparison build 657582a3.
- Kernel 7.1.12-2-12.6-sep-ARCH; boot 2abdc10a for every run in this directory; Vulkan driver Mesa 26.3.0-devel (git-69bde15340), library sha256 prefix 40282811c6e14ef9; device Apple M1 Max (G13C C0). Firmware identity, read from the device tree on the same machine after the runs (values verbatim, vendor prefix of the property names dropped): compatible apple,j316c / apple,t6001; iboot1-version mBoot-18000.161.10; iboot2-version iBoot-8422.141.2; os-fw-version 13.5; system-fw-version 26.6.2. The runs did not print them.
- Model and quantization: the repo snapshot above (4-bit); model.safetensors.index.json sha256 13b840162b4cb35c66fef7df072f7dbb4717908204364f5e5d9f9655a2758fa8. The three weight files were not hashed.
- Commands: tools/ (verify-r3.sh with stage test | test2 | census | numP1 | numP2 | speedA | speedB, wheel-r3.sh, run27.sh, run-nll27.sh).
- Numerical result and dispatch trace: sections above. The census histograms are in ab-two-builds/hist-*.txt.
- Timing and thermal procedure: one boot, performance governor, fresh process per sample, arm order alternated by round, one warm-up as the first load of each process. No temperature, fan or throttle reading was taken. The GPU-timeout counter was read before and after each ticket: delta 0 on the five NLL tickets (nll/tickets/) and on six A/B tickets (census, numP1, numP2, speedA, speedB, test2; ab-two-builds/tickets/). The ticket file of the first test stage was not kept.
- Vulkan device reopen: no dedicated close-and-reopen test. Every run is a fresh process that opened the device from the start; every logged exit code is 0 except the first test run above.

## Limits

One chip (G13C), two measured models (Qwen3.6-35B-A3B in PR 243 and Qwen3.8-27B-4bit here), one text, a 512-token NLL window, two 20-token speed prompts. G14C (M2 Max) is not measured and keeps the tiled default. The Hv == 4 * Hk ratio is not measured and keeps the tiled default. The NLL arrays and the speed numbers come from the 49e6b3c1 wheel and the N/O wheels respectively; both hold the same kernels. The *.npy full log-probability arrays (several MB each) are not kept; their digests are in the run logs. Names of the machines are not recorded; the build script in tools/ was run on the 16-inch M1 Max.

## Files

- ab-two-builds/: run-*.log per stage, t-*.out and t-*.err per timing run, num-*/ (top.npz, ids, without full.npy), cen-*/, hist-*.txt, test-all.out, test-list.out.
- nll/: census/ (the dispatch census of the NLL wheel), PREREG-nll27.md, the five score arrays (D0, D1, D2, U0, U1 .npz), their .out and .run.log files, nll-cmp.log.
- same-wheel-speed/: the two timing logs of the switch inside one wheel.
- tools/: the run scripts and readers. SHA256SUMS lists every file in this directory.
