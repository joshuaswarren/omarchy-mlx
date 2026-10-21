# bf16 A_log GatedDeltaUpdate decode at equal head counts on G14C: untiled default and a wide prefetch kernel (2026-10-21)

Qwen3.8-2B 4-bit (24 layers, 16 key heads and 16 value heads, bf16 A_log) decodes 3.74 percent faster on the Apple M2 Max with this change. The change has two parts. Part 1 makes the untiled prefetch kernel `GatedDeltaDecodeBF16Pf` the default at equal head counts on G14C. Part 2 adds `GatedDeltaDecodeBF16PfWide`, which keeps 32 state words in flight per memory round trip instead of 8, and makes it the default at the same shape. The wide kernel equals the plain kernel bit for bit. The untiled and tiled kernels are not bit-equal.

## Machine

14-inch M2 Max, Vulkan device Apple M2 Max (G14C B1), kernel 7.1.12-2-20261013.18-sep-ARCH, Mesa 26.3.0-devel (git-f144c7949b), performance governor, one boot for every run here. Model mlx-community Qwen3.8-2B 4-bit, `model.safetensors` sha256 b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863. Firmware identity read from the device tree (vendor prefix of the property names dropped): compatible apple,j414c / apple,t6021; iboot1-version mBoot-20457.1.29; iboot2-version iBoot-8422.141.2; os-fw-version 13.5; system-fw-version 27.0. Model index `model.safetensors.index.json` sha256 0f3e25e8b0d783a6808ac27d8dca33875bbd9de77a782aa8d44946f062774e24. The step is the coreglass ledger LLM step (`scripts/llmstep.py`): 477 prompt tokens, 128 generated tokens, one process per sample.

## Wheels

Main: mlx_omarchy-0.32.4.dev202610211005+a04423e (main a04423e42), libmlx prefix 0f9e6d6529dae3a4.

This change: mlx_omarchy-0.32.4.dev202610211622+356c2df (356c2df2), sha256 0f5294f585d878bdf15e2ac2fb35411688b509e55e0dd510704fd563e09e87f5, libmlx prefix 1edd35f48ad2baea. No diagnostic stamp, no profiling strings.

The first screen used the wheel at 7a09bac (libmlx prefix eff0e0a66cecc89d). It differs from the final head only in the default selection: the wide kernel was opt-in there.

## Speed on the final head (speed/final-head-356c2df2/)

Five rounds. Every arm ran once per round. The arm order rotated by one place each round, so every arm held every position once. One solo window on a quiet machine (load below 1.3 at start). Digests are the sha of the 128 generated token ids. All 25 equal `c4ed7eeb93e2eda6`. Median tokens per second, then the change against main, then the five rounds:

Main, default: 101.6 tok/s.

This change, default (wide kernel): 105.4 tok/s, +3.74% against main, rounds +4.15 +3.85 +3.74 +3.54 +3.34.

This change, `MLX_OMARCHY_GDN_PF_WIDE=0` (plain kernel): 104.2 tok/s, +2.76% against main, rounds +2.76 +3.16 +2.95 +2.36 +2.26.

Main, `MLX_OMARCHY_GDN_DECODE_TILE=0`: 104.3 tok/s, +2.95% against main, rounds +2.96 +2.96 +2.95 +2.36 +2.94.

This change, `MLX_OMARCHY_GDN_DECODE_TILE=1`: 101.4 tok/s, -0.1% against main, rounds -0.2 0.0 -0.1 -0.1 -0.49.

The change is the median of the per-round paired changes against main. It differs from the ratio of the two median tok/s: 104.2 over 101.6 is +2.56%, and the list says +2.76%. `speed/summarize-arms.py arms.tee` prints this list from the log. The last arm equals main, so the two wheels do not differ by build.

## First screen on 7a09bac (speed/first-screen-7a09bac/)

Same machine, step and design. The wide kernel was selected with `MLX_OMARCHY_GDN_PF_WIDE=1`: +4.23% median over main, rounds +4.65 +4.23 +3.63 +4.34 +3.85; plain +2.67%; main with TILE=0 +2.86%; tiled kernel on the new wheel +0.1%. All digests equal.

## Screens of other switches on main (speed/main-switch-screen-a, b, c)

Eleven, seven and four arms on main a04423e with one environment switch each (arm definitions in `scripts/m2-arms.sh`, `m2-arms2.sh` and `m2-arms3.sh`). The base arm read 101.8 and 101.2 tok/s. `MLX_OMARCHY_GDN_DECODE_TILE=0` gave +2.36%, +2.66% and +2.87%. `MLX_OMARCHY_SDPA_DECODE_NATIVE=0` gave -1.28%. `MLX_OMARCHY_GATED_BARRIERS=0` gave -2.65%. `MLX_OMARCHY_BATCH_WORK=0` gave +0.98% and cost 33 percent of prefill. Chunked argmax and chunked logsumexp gave -0.2% and -0.1%, so the vocabulary tail is not on the critical path. All other arms stayed between -0.6% and 0. Every digest equals `c4ed7eeb93e2eda6`. One arm (`noword`) exited with an error and has no number.

## Numerics (numerics/)

`scripts/armab.sh num` runs a 128-token greedy decode and records the full log-probability vector of 32 steps, with a fresh process per run.

Plain (`MLX_OMARCHY_GDN_PF_WIDE=0`) against the new default, two prompts: ids equal, max absolute difference 0.0, 0 differing elements. A repeat of the first arm equals it. The dispatch census (`hist-D-10.txt`, `hist-X-10.txt`) shows kernel 480 x198 for plain and kernel 638 x198 for the default, so the arms differ.

Tiled (`MLX_OMARCHY_GDN_DECODE_TILE=1`) against the new default: kernel 455 x198 against 638 x198. The greedy ids differ within 128 tokens on both prompts and the largest log-probability difference is 0.5. These two kernels are not bit-equal.

## Reference gate (numerics/reference-gate/)

`docs/numerics-gate.md` judges a kernel change against an independent reference, not against the kernel it replaces. The reference is a float32 CPU forward of the same `model.safetensors` (sha256 above) with upstream mlx 0.32.4 (package mlx-cpu 0.32.4), mlx-lm 0.32.2, numpy 2.4.6, Python 3.11 on an x86-64 Linux machine. It shares no GPU code with the Omarchy wheel. `tfref_lse.py` ran it once more than needed: its target log-probabilities, top-20 ids and top-20 values equal the first run bit for bit on all 4 segments. The Omarchy wheel on the M2 cannot run the CPU path (a CPU run there stops with `NYI`, see `gpu-arms-ticket-run.log`), so the CPU arm of the ticket produced nothing and the reference came from the other machine.

Text: the GPL-3 licence text (`tfref-text.txt`, sha256 3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986), 4 segments of 160 tokens starting at token 0, 1500, 3000 and 4500, a fresh cache per segment, one forward per token (S=1 teacher forcing), so the GDN decode kernel runs at every step. 159 scored steps per segment, 636 in total. The cache and the GDN state span at most 159 positions; the gate says nothing about error growth at the 477 to 605 token context of the timing step.

GPU arms ran in one ticket on the wheel at 7a09bac (`gpu-arms-ticket-run.log`, `m2-pr-gate.sh`): `tile1` is `MLX_OMARCHY_GDN_DECODE_TILE=1` (the old default kernel), `plain` is the untiled prefetch kernel (the default on that wheel), `plain2` repeats it, `wide` is `MLX_OMARCHY_GDN_PF_WIDE=1`. The default of the final head runs the same wide kernel code as `wide`; commit 356c2df2 differs from 7a09bac in the selector only. Results are in `tfcmp.out`, `tfpair.out`, `tfseg.out` and `tfulp.out`:

Reference: mean NLL 1.81889.

Tiled kernel (old default): top-1 agreement 625/636 = 98.27%, 11 flips, mean NLL 1.82124 (+0.129% against the reference), mean absolute log-prob error of the next token 0.02841.

Untiled plain kernel: top-1 agreement 621/636 = 97.64%, 15 flips, mean NLL 1.81995 (+0.058% against the reference), mean absolute log-prob error of the next token 0.03051.

Wide kernel (new default): top-1 agreement 621/636 = 97.64%, 15 flips, mean NLL 1.81995 (+0.058% against the reference), mean absolute log-prob error of the next token 0.03051.

Wide equals plain on all 636 steps, bit for bit (max difference 0.0). `plain2` equals `plain` bit for bit.

Token choices: the new default agrees with the reference on 4 fewer steps than the tiled kernel (0.63 points, inside the 95% binomial band at 636 steps, about +-1.0 point). `tfulp.out` classifies every flip. All 15 flips of the new default are within 2 bf16 ULP of the reference top-2 margin at the top logit (top logits 13 to 21, ULP 0.0625 or 0.125, margins 0.004 to 0.173), so none is a failure under the rule. All 11 flips of the tiled kernel are near ties too.

Perplexity on the teacher-forced text: the new default is +0.058% against the reference NLL and the tiled kernel is +0.129%. The deviation of the new default is smaller.

Log-probability error of the next token: the new default is 0.00210 higher on average (0.03051 against 0.02841). Paired over steps the 95% bootstrap interval is [-0.00087, +0.00525] and the new kernel is closer on 306 steps, the tiled kernel on 316, equal on 14. Resampling by segment (all 256 resamples of 4 segments) gives [-0.00103, +0.00483]; the per-segment differences are +0.00141 +0.00394 +0.00573 -0.00269. The two kernels are not distinguishable in log-probability error at this sample size.

Not run: the per-op fp32/fp64 error of captured operands (check 1 of the gate). The untiled prefetch kernel is the kernel that the G13C and the G14C ratio 3 defaults already use.

Doctests on the final head (`ticket-logs/pr3-ticket.out`): `omarchy_gdn_gqa_bf16a_tests` 9 of 9 cases, 143 of 143 assertions; `omarchy_runtime_tests` 59 of 59 cases, 27070 of 27070 assertions. Exit codes 0.

## Build (ticket-logs/)

`scripts/m2-pr3.sh` clones the branch, resets to 356c2df2, builds the wheel and the two doctest binaries, runs the doctests and the two numerics comparisons. A first run of the same script on commit 631bcf52 failed one doctest assertion. The selector read the head counts after the q and k expansion and took the wide kernel at ratio 3 too. Commit 356c2df2 fixes it. The 631bcf52 results are not part of this directory.

## Timing and thermal procedure

Every run ran on one boot as a queued ticket of the lab scheduler, with the performance governor, one fresh process per sample, the arm order rotated by round, and the machine load below 1.3 at the start. The 25 samples of the final screen ran back to back (ticket 16:44:01Z to 16:45:43Z) with no cool-down interval. No temperature, fan or throttle reading was taken. The GPU timeout counter was read before and after the build and numerics ticket: delta 0 (`ticket-logs/pr3-ticket.out`). The scheduler's counter record for the timing ticket was not copied. No dedicated close-and-reopen test of the Vulkan device was run. Every sample is a fresh process that opened the device from the start, every exit code in the run logs is 0 and every digest matches.

## Limits

One chip (G14C), one model, one text, one thermal state, one boot. The default applies to any bf16 A_log model with equal head counts on a G14C part; Qwen3.8-2B is the only one measured. Ratio 2 on G14C is not measured. The wide kernel is not measured at ratio 3 and is not the default there.

## Files

SHA256SUMS lists every file here.
