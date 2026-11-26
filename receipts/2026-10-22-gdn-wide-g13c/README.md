# Wide prefetch GDN decode kernel as the equal-head default on G13C (2026-10-22)

Qwen3.8-2B 4-bit decodes 3.64 percent faster on the Apple M1 Max (G13C): 102.4 against 98.8 tokens per second, median of 5 interleaved rounds. The change makes the wide prefetch kernel `GatedDeltaDecodeBF16PfWide` the default for bf16 A_log models with equal key and value head counts on G13C. The wide kernel came with PR 252, where it became the default on G14C at the same shape. It equals the plain prefetch kernel bit for bit. The untiled and the tiled kernels are not bit-equal.

The rule now lives in one pure function, `gdn_pf_wide_default(device, hk, hv)` in `device.h`. It holds where the untiled default holds and the head counts are equal. `gdn_bf16a_untiled_default` also gains G13C at equal head counts. G14C behaviour is unchanged: the rule gives the same answer there as the selector of PR 252. Ratios 2 and 3 keep the plain kernel on both parts, because the wide kernel does not help there (see "Other shapes").

## Machine

16-inch M1 Max, Vulkan device Apple M1 Max (G13C), kernel 7.1.12-2-12.6-sep-ARCH, Mesa 26.3.0-devel (git-f144c7949b), performance governor, one boot for every run on it. Firmware identity read from the device tree (vendor prefix of the property names dropped): compatible apple,j316c / apple,t6001; iboot1-version mBoot-18000.161.10; iboot2-version iBoot-8422.141.2; os-fw-version 13.5; system-fw-version 26.6.2. The 14-inch M2 Max that made part of the captures is described in `receipts/2026-10-21-gdn-bf16a-g14c-ratio1`.

Model mlx-community Qwen3.8-2B 4-bit, `model.safetensors` sha256 b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863, index `model.safetensors.index.json` sha256 0f3e25e8b0d783a6808ac27d8dca33875bbd9de77a782aa8d44946f062774e24. The step is the coreglass ledger LLM step (`scripts/llmstep.py`): a repeated-sentence prompt of 477 nominal tokens (443 real), 128 generated tokens, one process per sample.

## Wheels

- Main: mlx_omarchy-0.32.4.dev202610211005+a04423e (main a04423e42), libmlx prefix 0f9e6d6529dae3a4. It is the wheel of the ledger and has the tiled kernel as the default here.
- Main at aaef3b6c (PR 252): libmlx prefix 62cc77d961d3ead6 on this machine, fd536b16c05e08b0 on the M2 Max. Separate builds of one commit.
- This change: mlx_omarchy-0.32.4.dev202610220317+ba12383 (ba12383f), sha256 bee78ccda372306672fab0e756f2cce7c77544a752291610455816af38197700, libmlx prefix a649ee784c9f61f9. No diagnostic stamp, no profiling strings.

## Speed on the PR wheel (speed/m1max-pr-final/)

Five rounds. Every arm ran once per round and the arm order rotated by one place each round. One solo window. All 20 token digests equal `c4ed7eeb93e2eda6`. Median tokens per second, then the change against main, then the five rounds:

Main, default: 98.8 tok/s.

This change, default (wide kernel): 102.4 tok/s, +3.64%, rounds +3.85 +3.43 +3.86 +3.44 +3.64.

This change, `MLX_OMARCHY_GDN_DECODE_TILE=0 MLX_OMARCHY_GDN_PF_WIDE=0` (plain kernel): 101.6 tok/s, +3.04%, rounds +3.34 +2.63 +3.15 +3.04 +2.63.

This change, `MLX_OMARCHY_GDN_DECODE_TILE=1` (tiled kernel): 98.7 tok/s, +0.1%, rounds 0.0 -0.3 +0.2 +0.2 +0.1.

The change is the median of the per-round paired changes against main, as in PR 252. The tiled arm equals main, so the PR wheel and the main wheel do not differ by build. The script gates the load only at its start. The load average read 0.37 to 1.14 in rounds 1 to 3 and 0.87 to 1.6 in rounds 4 and 5. The median of rounds 1 to 3 alone is +3.85%, so the late load does not raise the result. `speed/summarize-arms.py arms.tee` prints this list from the log.

## Earlier screen on the main wheel (speed/m1max-first-screen/)

The same step on main aaef3b6c, before this change existed: `MLX_OMARCHY_GDN_DECODE_TILE=0` +3.23%, the new wheel's default 0.0%, TILE=0 with the plain kernel +3.25%, TILE=0 with `MLX_OMARCHY_GDN_PF_WIDE=1` +3.96% median, rounds +3.96 +3.95 +4.04 +3.54 +4.16. All 25 digests equal `c4ed7eeb93e2eda6`.

## Other shapes (speed/m1max-27b/, speed/m1max-35b/)

Qwen3.8-27B 4-bit (16 key heads, 48 value heads, snapshot 10c35caa, index sha256 13b840162b4cb35c...): the plain kernel and the wide kernel read 14.2 and 14.0 tok/s, so the wide kernel is 1.41 percent slower in every round. Qwen3.6-35B-A3B 4-bit (16 and 32 heads, snapshot 38740b84, index sha256 0b28df60e33753a1...): base 29.3, plain 29.3, wide 29.4 tok/s, +0.34 percent, inside the noise. All digests in each model equal. The wide kernel therefore stays opt-in at ratios 2 and 3. The tests assert it.

## Reference gate (numerics/)

`docs/numerics-gate.md` judges a kernel change against an independent reference. The reference is a float32 CPU forward of the same `model.safetensors` with upstream mlx 0.32.4 (package mlx-cpu), mlx-lm 0.32.2 and numpy 2.4.6 on an x86-64 Linux machine. It shares no GPU code with the Omarchy wheel. The method is the one of `receipts/2026-10-21-gdn-bf16a-g14c-ratio1/numerics/reference-gate`. The text is the GPL-3 licence text (`text/tfref-text.txt`).

Four shapes ran. Short: 4 segments of 160 tokens, 636 scored steps. Long: 2 segments of 600 tokens at text offsets 0 and 1500, 1198 steps. Prompt 1939: the repeated-sentence prompt of nominal 2048 tokens (1939 real) in one prefill pass, then 129 forced tokens of the text. Prompt 3877: the same for nominal 4096. Short and long feed one token per forward from an empty cache. The two prompt shapes feed the forced tokens after a prefill, so the decode kernel works on a cache of 1939 or 3877 positions. The prefill code is the same in every arm.

Arms on this machine: `tile1` is the main wheel with `MLX_OMARCHY_GDN_DECODE_TILE=1` (the tiled kernel, the default before this change), `wide` is the main-at-aaef3b6c wheel with TILE=0 and PF_WIDE=1 (the kernel this change makes the default), `plain` is the same wheel with TILE=0 and PF_WIDE=0.

Short shape: the tiled kernel agrees with the reference on 624 of 636 top-1 choices (98.11%), the wide kernel on 625 (98.27%). Mean NLL against the reference: tiled +0.0525%, wide +0.0042%. Mean absolute log-probability error of the next token: tiled 0.02945, wide 0.02622. All 12 flips of the tiled kernel and all 11 of the wide kernel lie within 2 bf16 ULP of the reference top-2 margin. Paired over steps the wide minus tiled error is -0.00323 with a 95% bootstrap interval of [-0.00631, -0.00025]. Over the four segments the interval is [-0.00695, +0.00050].

Long shape: tiled 1176 of 1198 (98.16%), wide 1178 (98.33%). Mean NLL against the reference: tiled +0.2399%, wide +0.0796%. Mean absolute error: tiled 0.02812, wide 0.02952. The wide minus tiled difference is +0.00140 with an interval over steps of [-0.00155, +0.00431]. Each arm has one flip beyond 2 ULP: the tiled kernel at step 405 of segment 0 (margin 0.4239) and the wide kernel at step 264 of segment 0 (margin 0.2895), both at ULP 0.125. By position, 0 to 159: tiled 97.50%, wide 97.19%. Positions 160 to 599: tiled 98.41%, wide 98.75%. Two segments give no segment-level interval: the two per-segment differences are +0.00108 and +0.00172.

Prompt 1939: tiled 128 of 129 (99.22%), wide 127 (98.45%). Mean NLL against the reference: tiled -0.2651%, wide +0.3771%. Both arms flip near ties only. This shape has 129 steps, so one flip moves the agreement by 0.8 points. The difference is inside the 95% binomial band of this sample.

Prompt 3877: tiled 129 of 129 (100.00%), wide 128 (99.22%), one near-tie flip at margin 0.0254. Mean NLL against the reference: tiled +0.1547%, wide +0.4096%. Mean absolute error: tiled 0.02082, wide 0.01721. The paired difference is -0.00361 with an interval of [-0.00931, +0.00249]. On both prompt shapes the NLL deviation of the wide kernel is larger than the tiled one's, and its per-token error is smaller. Each sample has 129 steps, too few to separate them. The two prompt shapes do not show a failure. They do not show an improvement either.

The rule applied is the one of `docs/numerics-gate.md`. Agreement with the reference is at least the old kernel's within the binomial band, and flips within 2 ULP are not failures. The deviation of the NLL is within the noise of the old kernel's or smaller. The wide kernel meets both on the short and the long shape. On the two prompt shapes it meets the first and the sample is too small for the second. It is not worse than the tiled kernel on the one flip beyond 2 ULP, because the tiled kernel has one as well. The long-shape error per token is 0.0014 nats higher for the wide kernel while its NLL deviation is smaller. Both numbers are reported as measured.

Wide equals plain bit for bit on every step of every shape and on both machines (`numerics/results/*-tfcmp.out`, pair "wide vs plain"). The tiled kernel repeated with the main wheel (`tile1b` in the long-prompt runs) equals its first run bit for bit.

## One chip against the other (numerics/results/chip-comparison.out)

The same captures ran on this machine (G13C) and on the 14-inch M2 Max (G14C) with the same stack. `np.array_equal` on the four arrays of each segment (ids, target_lp, top_i, top_v) returns true for the tiled and the new arm, in the long shape and in both prompt shapes. These decode kernels give identical numbers on the two chips. That is why one CPU reference serves both. The captures of both chips are in `numerics/captures/`.

## Doctests (ticket-logs/)

`m1max-pr-build-and-doctests.out`: on the PR commit, `omarchy_gdn_gqa_bf16a_tests` passes 9 of 9 cases and 151 of 151 assertions on this machine. The ratio 1 case reads default kernel 638, the wide kernel. `omarchy_runtime_tests` passes 59 of 59 cases and 27070 of 27070 assertions. Both exit codes are 0. The GPU timeout counter delta is 0.

`m2-pr-build-and-doctests.out`: the same two binaries built at the same commit on the 14-inch M2 Max (G14C) pass 151 of 151 and 27070 of 27070 assertions, exit codes 0, and the ratio 1 case reads default kernel 638 there too. That run covers the G14C expectations of the changed rule.

## Context screen of PR 252 on the M2 Max (speed/m2-context-screen/, speed/m2-9b-control/)

These two runs belong to PR 252 and are kept here with its follow-up gates. Main a04423e against main at aaef3b6c with the default environment on the 14-inch M2 Max, 5 interleaved rounds per prompt length, repeated-sentence prompts. Real prompt 443 tokens: +3.65%. Real prompt 1939: +3.43%. Real prompt 3877: +2.12%, rounds -0.53 +2.30 +2.12 +2.18 -0.35. The token digests are equal at 443 and 1939 tokens and differ at 3877. The tiled and the untiled kernels are not bit-equal, so a long context can pick another greedy continuation. The gates above measure that difference against the reference. Control: Qwen3.5-9B 4-bit (f32 A_log, ratio 2) at 443 and 1939 tokens reads +0.00% in both and the digests equal 9a142db605d5939d, so PR 252 changed nothing on that route.

## Limits

One model (Qwen3.8-2B), one text, two chips, one boot each. The default applies to any bf16 A_log model with equal head counts on G13C and G14C. The reference gate covers 159 and 599 positions from an empty cache and two prefill lengths. The per-op fp32/fp64 error check (check 1 of the gate) was not run. The G14C speed of this PR is the result of PR 252, because the rule gives the same answer there.

## Timing and thermal procedure

Every run ran as a queued ticket of the lab scheduler with the performance governor, one fresh process per sample, and the arm order rotated by round. No temperature, fan or throttle reading was taken. The 20 samples of the final screen ran back to back (ticket 03:43:10Z to 03:44:40Z) with no cool-down interval. The GPU timeout counter delta was 0 for the build and doctest ticket. The scheduler counter record of the timing tickets was not copied. No dedicated close-and-reopen test of the Vulkan device was run. Every sample is a fresh process that opened the device from the start, every exit code in the run logs is 0, and every digest matches.

## Files

SHA256SUMS lists every file here. `scripts/` holds the scripts that ran, with host paths removed, and the analysis scripts. `numerics/results/` holds the outputs of the analysis. `numerics/captures/` holds the GPU captures of both chips, and `numerics/references/` the CPU references.
