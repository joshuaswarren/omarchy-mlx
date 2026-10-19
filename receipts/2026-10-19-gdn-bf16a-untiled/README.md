# bf16 A_log GatedDeltaUpdate decode: the untiled prefetch kernel as the default on G13C (2026-10-19)

On a Vulkan device whose name contains G13C, a model with a bf16 A_log, Hk != Hv and Hv == 2 * Hk (Qwen3.6-35B-A3B: 16 key heads, 32 value heads) now decodes with GatedDeltaDecodeBF16Pf (the untiled prefetch kernel, with the two q and k expansion copies per linear layer per token) instead of the tiled GatedDeltaDecodeBF16. Every other part and shape keeps its previous default. MLX_OMARCHY_GDN_DECODE_TILE=1, MLX_OMARCHY_GDN_PF=0 and MLX_OMARCHY_GDN_GQA_BF16A=1 each keep the tiled kernel.

The two kernels are not bit-equal. The default therefore changed only after a paired teacher-forced NLL check passed the not-worse rule (see "Decision").

## Machine and build

16-inch M1 Max, Vulkan device Apple M1 Max (G13C C0), kernel 7.1.12-2-12.6-sep-ARCH, driver library prefix 40282811c6e14ef9. Model mlx-community/Qwen3.6-35B-A3B-4bit snapshot 38740b847e4cb78f352aba30aa41c76e08e6eb46.
Two wheels were used. The speed, numerics and NLL runs of "untiled vs tiled" used the rebased head of PR 231, mlx_omarchy-0.32.4.dev202610181348+49e6b3c1 (sha256 7654db6b519b221ab5b221e96e1bda3b28c6317b0969b78de2abf3d101f8fe59, libmlx prefix 960f0f358a2aba25), with the switch MLX_OMARCHY_GDN_DECODE_TILE=0. The check of this change used the wheel built from the commit of this pull request, mlx_omarchy-0.32.4.dev202610192245+a2b0864 (sha256 a1eb9765672cf21873a032eeb10e3e27dc6c03be908d267585183a4101b55f5c, libmlx prefix 818270b8ca578548) and its test binary (sha256 9f81bcbaf4d0debe491137069249f1b23d6cb01a4ea74bdb96059324eef7fb05).
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

- New test case "gdn raw decode bf16 A_log: the default kernel by part, the switches and the routes outside it": 14 of 14 assertions on this device; the whole binary runs 7 of 7 cases and 107 of 107 assertions (test/new-case.out, test/test-all.out). It names the default kernel by enum (480 here), the switches, a 16/48 head ratio and equal head counts, which keep the previous default.
- Census of the new wheel (verify/run.log): default dispatches kernel 480 (330 per 10 tokens), TILE=1 dispatches kernel 455; dispatch totals 11992 and 17272 in both arms.
- Numerics of the new wheel against the earlier runs: the new default equals the earlier TILE=0 arm bit for bit, and the new TILE=1 equals the earlier default arm bit for bit, on both prompts (max absolute log-probability difference 0.0, KL 0, ids equal; verify/run.log).
- Speed of the new wheel, default against TILE=1 (tools/sum_speed.py verify/run.log T D): P1 28.950 to 31.210 tok/s (+7.81%), P2 28.970 to 31.220 (+7.77%), the default faster in 5 of 5 rounds on both prompts, spread 0.3%.

## Limits

One model (Hk 16, Hv 32), one chip (G13C), one text, two 20-token speed prompts. G14C (M2 Max) was not measured: no 35B model is on that machine, so it keeps the tiled default. The Qwen3.8 family shapes (16 key heads, 48 value heads) are not in the measured set and keep the tiled default. The speed and NLL runs used the pre-rebase PR head wheel; the check of this change used its own wheel. The NLL check ran the decode path with a 512-token prefill window; a long context was not scored. The cause of the numerical difference between the kernels is not investigated.

## Files

- untiled-vs-tiled/: speed-run.log, numerics-run.log, hist-*.txt (decode census).
- nll/: PREREG.md, run.log, D.npz, U.npz, D2.npz, scored with tools/nll_pair.py and tools/nll_cmp.py.
- verify/: the run log, census histograms and per-run outputs of the new wheel.
- test/: the device test outputs.
- tools/: the run scripts and readers (run-nll35.sh, verify-bf16a.sh, run-untiled-vs-tiled.sh, gdn_logits.py, gdn_numcmp.py, nll_pair.py, nll_cmp.py, sum_speed.py). The *.npy log-probability arrays (about 30 MB each) are not kept; their digests are in the run logs.
