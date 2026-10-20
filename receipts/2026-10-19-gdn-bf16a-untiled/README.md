# bf16 A_log GatedDeltaUpdate decode: the untiled prefetch kernel as the default on G13C (2026-10-19)

On a Vulkan device whose name contains G13C, a model with a bf16 A_log, Hk != Hv and Hv == 2 * Hk (Qwen3.6-35B-A3B: 16 key heads, 32 value heads) now decodes with GatedDeltaDecodeBF16Pf (the untiled prefetch kernel, with the two q and k expansion copies per linear layer per token) instead of the tiled GatedDeltaDecodeBF16. Every other part and shape keeps its previous default. MLX_OMARCHY_GDN_DECODE_TILE=1, MLX_OMARCHY_GDN_PF=0 and MLX_OMARCHY_GDN_GQA_BF16A=1 each keep the tiled kernel.

The two kernels are not bit-equal. The default therefore changed only after a paired teacher-forced NLL check passed the not-worse rule (see "Decision").

## Machine and build

16-inch M1 Max, Vulkan device Apple M1 Max (G13C C0), kernel 7.1.12-2-12.6-sep-ARCH, driver library prefix 40282811c6e14ef9. Model mlx-community/Qwen3.6-35B-A3B-4bit snapshot 38740b847e4cb78f352aba30aa41c76e08e6eb46.
Two wheels were used. The speed, numerics and NLL runs of "untiled vs tiled" used the rebased head of PR 231, mlx_omarchy-0.32.4.dev202610181348+49e6b3c1 (sha256 7654db6b519b221ab5b221e96e1bda3b28c6317b0969b78de2abf3d101f8fe59, libmlx prefix 960f0f358a2aba25), with the switch MLX_OMARCHY_GDN_DECODE_TILE=0. The check of this change used the wheel built from the commit of this pull request, mlx_omarchy-0.32.4.dev202610192245+a2b0864 (sha256 a1eb9765672cf21873a032eeb10e3e27dc6c03be908d267585183a4101b55f5c, libmlx prefix 818270b8ca578548) and its test binary (sha256 cb3696d2c6501d3941903b65d46f430b0330acafc8db93ecf94b74d580cbdbc0).
Timing ran under the lab GPU lock, on one boot, with the performance governor. The runs are not guard tickets, so no GPU-timeout counter was read.

## Untiled against tiled (untiled-vs-tiled/)

Route: the decode census (10 tokens minus 6) names GatedDeltaDecodeBF16 (id 455) x330 for the tiled default and GatedDeltaDecodeBF16Pf (id 480) x330 for MLX_OMARCHY_GDN_DECODE_TILE=0. Every other histogram line and the dispatch totals are equal (11992 and 17272 for 6 and 10 tokens).
Speed (decode tok/s, 64-token runs, 5 rounds, fresh process each, order alternating; tools/sum_speed.py on speed-run.log):

| Prompt | Tiled median | Untiled median | Change | Untiled faster | Spread |
|---|---:|---:|---:|---:|---:|
| P1 | 28.900 | 31.250 | +8.13% | 5/5 | 0.35% |
| P2 | 28.900 | 31.230 | +8.06% | 5/5 | 0.52% |

Numerics (128 greedy tokens, full log-probability vector of the first 32 steps; numerics-run.log): the two kernels are not bit-equal. P1 generated ids diverge at step 30 (max absolute log-probability difference 6.0, KL up to 0.0049, top-5 sets equal in 26 of 30 steps). P2 ids are equal over 128 steps with max difference 4.75, KL 0.0228. Two default runs are bit-equal, so the run is deterministic.

## Decision: paired teacher-forced NLL, accept-not-worse (nll/)

nll/PREREG.md was written before the scoring run. 128 prefixes of a fixed text, each a 512-token prefill and 64 decode steps, 8192 scored positions per arm. d = NLL(untiled) - NLL(tiled); the unit of replication is the prefix.
Result (nll/run.log): D mean +0.00082 nats per token, cluster SE 0.00225, interval at 2 SE [-0.00369, +0.00533], at t(127) the upper end is +0.00527 (computed from the logged mean and SE; the log prints the 2 SE interval). Both are below the +0.02 line, so the rule gives ACCEPT-NOT-WORSE. The two default runs (D and D2) are bit-identical on every position; the untiled run differs from the default on 7964 of 8192 positions; 203 of 8192 argmax positions differ.
This is a not-worse check on one text and one model. It is not a quality benchmark.

## This change (verify/, test/)

- New test case "gdn raw decode bf16 A_log: the default kernel by part, the switches and the routes outside it": 14 of 14 assertions on this device; a second case re-runs the binary in a fresh process with MLX_OMARCHY_GDN_PF=0 and checks that the default stays tiled (kernel 455) and TILE=0 runs the scalar untiled kernel (479), 2 of 2 assertions; the whole binary runs 8 of 8 cases and 109 of 109 assertions (test/new-case.out, test/pf0-case.out, test/test-all.out). It names the default kernel by enum (480 here), the switches, a 16/48 head ratio and equal head counts, which keep the previous default.
- Census of the new wheel (verify/run.log): default dispatches kernel 480 (330 per 10 tokens), TILE=1 dispatches kernel 455; dispatch totals 11992 and 17272 in both arms.
- Numerics of the new wheel against the earlier runs: the new default equals the earlier TILE=0 arm bit for bit, and the new TILE=1 equals the earlier default arm bit for bit, on both prompts (max absolute log-probability difference 0.0, KL 0, ids equal; verify/run.log).
- Speed of the new wheel, default against TILE=1 (tools/sum_speed.py verify/run.log T D): P1 28.950 to 31.210 tok/s (+7.81%), P2 28.970 to 31.220 (+7.77%), the default faster in 5 of 5 rounds on both prompts, spread 0.3%.

## Two distinct builds, default environment (two-build-ab/)

The runs above switch the route inside one wheel. This run compares the two builds themselves, both in the default environment, with a verified build stamp on every row: O is mlx_omarchy 0.32.4.dev202610181348+49e6b3c1 (before this change), N is 0.32.4.dev202610192245+a2b0864 (this change). Every row prints provenance ... verified=match with its own build stamp and the libmlx hash of its wheel. 5 rounds, 2 prompts, 64 tokens, order alternating, fresh process each (tools/ab-builds.sh, tools/sum_speed.py two-build-ab/run.log O N):

| Prompt | O median | N median | Change | N faster | Spread |
|---|---:|---:|---:|---:|---:|
| P1 | 28.920 | 31.240 | +8.02% | 5/5 | 0.31% |
| P2 | 28.940 | 31.200 | +7.81% | 5/5 | 0.38% |

## Hardware record

- Source commits: a2b0864c36761b00c4c97b5c78f32e7b6bd841df (this change, on main b7b0aba36); the comparison build is 49e6b3c1, the rebased head of PR 231.
- Kernel 7.1.12-2-12.6-sep-ARCH. Mesa package 26.2.4 (pacman mesa 1:26.2.4-1); the Vulkan driver reports Mesa 26.3.0-devel (git-69bde15340), library sha256 prefix 40282811c6e14ef9.
- Vulkan device Apple M1 Max (G13C C0). Firmware identity, read from the device tree after the runs on the same machine (values verbatim, the vendor prefix of the property names dropped): compatible apple,j316c / apple,t6001; iboot1-version mBoot-18000.161.10; iboot2-version iBoot-8422.141.2; os-fw-version 13.5; system-fw-version 26.6.2. The runs did not print them.
- Model: mlx-community/Qwen3.6-35B-A3B-4bit snapshot 38740b847e4cb78f352aba30aa41c76e08e6eb46; model.safetensors.index.json sha256 0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea (4-bit quantization of the repo; the weight files were not hashed).
- Commands: tools/ (run-untiled-vs-tiled.sh, run-nll35.sh, verify-bf16a.sh, ab-builds.sh) and the test binary command lines in test/*.out.
- Numerical result, backend dispatch trace: sections above. The census histograms are in untiled-vs-tiled/hist-*.txt and verify/hist-*.txt.
- Timing and thermal procedure: all run sets ran on one boot; speed-run.log line 1 records its kernel and the performance governor, and the two-build-ab/run.log start line records only the boot and load. One boot per run set, performance governor, fresh process per sample, arm order alternated, one warm-up as the first load of each process; timing under the lab GPU lock. No temperature, fan or throttle reading was taken.
- Vulkan device reopen: no dedicated close-and-reopen test was run. Every run is a fresh process that opened the Vulkan device from the start. The logs record 72 exit codes (untiled-vs-tiled/speed-run.log 20, numerics-run.log 5, verify/run.log 24, two-build-ab/run.log 20, nll/run.log 3), all 0, and every timing row prints verified=match. The census runs and the two test-binary runs also exited cleanly; the census pipeline does not log its exit code.

## Limits

One model (Hk 16, Hv 32), one chip (G13C), one text, two 20-token speed prompts. G14C (M2 Max) was not measured: no 35B model is on that machine, so it keeps the tiled default. The Qwen3.8 family shapes (16 key heads, 48 value heads) are not in the measured set and keep the tiled default. The speed and NLL runs used the pre-rebase PR head wheel; the check of this change used its own wheel. The NLL check ran the decode path with a 512-token prefill window; a long context was not scored. The cause of the numerical difference between the kernels is not investigated.

## Files

- two-build-ab/: the two-build A/B (run.log and per-run outputs).
- untiled-vs-tiled/: speed-run.log, numerics-run.log, hist-*.txt (decode census).
- nll/: PREREG.md, run.log, D.npz, U.npz, D2.npz, scored with tools/nll_pair.py and tools/nll_cmp.py.
- verify/: the run log, census histograms and per-run outputs of the new wheel.
- test/: the device test outputs.
- tools/: the run scripts and readers (run-nll35.sh, verify-bf16a.sh, run-untiled-vs-tiled.sh, gdn_logits.py, gdn_numcmp.py, nll_pair.py, nll_cmp.py, sum_speed.py). The *.npy log-probability arrays (about 30 MB each) are not kept; their digests are in the run logs.
