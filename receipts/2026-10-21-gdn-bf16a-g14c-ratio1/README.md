# bf16 A_log GatedDeltaUpdate decode at equal head counts on G14C: untiled default and a wide prefetch kernel (2026-10-21)

Qwen3.8-2B 4-bit (24 layers, 16 key heads and 16 value heads, bf16 A_log) decodes 3.74 percent faster on the Apple M2 Max with this change. The change has two parts. Part 1 makes the untiled prefetch kernel `GatedDeltaDecodeBF16Pf` the default at equal head counts on G14C. Part 2 adds `GatedDeltaDecodeBF16PfWide`, which keeps 32 state words in flight per memory round trip instead of 8, and makes it the default at the same shape. The wide kernel equals the plain kernel bit for bit. The untiled and tiled kernels are not bit-equal.

## Machine

14-inch M2 Max, Vulkan device Apple M2 Max (G14C B1), kernel 7.1.12-2-20261013.18-sep-ARCH, Mesa 26.3.0-devel (git-f144c7949b), performance governor, one boot for every run here. Model mlx-community Qwen3.8-2B 4-bit, `model.safetensors` sha256 b0d5de688567bf4acd5e421027acd410dabcdc255a5bd46fdbf06c75dc2e6863. The step is the coreglass ledger LLM step (`scripts/llmstep.py`): 477 prompt tokens, 128 generated tokens, one process per sample.

## Wheels

- Main: mlx_omarchy-0.32.4.dev202610211005+a04423e (main a04423e42), libmlx prefix 0f9e6d6529dae3a4.
- This change: mlx_omarchy-0.32.4.dev202610211622+356c2df (356c2df2), sha256 0f5294f585d878bdf15e2ac2fb35411688b509e55e0dd510704fd563e09e87f5, libmlx prefix 1edd35f48ad2baea. No diagnostic stamp, no profiling strings.
- The first screen used the wheel at 7a09bac (libmlx prefix eff0e0a66cecc89d). It differs from the final head only in the default selection: the wide kernel was opt-in there.

## Speed on the final head (speed/final-head-356c2df2/)

Five rounds. Every arm ran once per round. The arm order rotated by one place each round, so every arm held every position once. One solo window on a quiet machine (load below 1.3 at start). Digests are the sha of the 128 generated token ids; all 25 equal `c4ed7eeb93e2eda6`.

| Arm | Median tok/s | Change vs main | Per round |
|---|---:|---:|---|
| main, default | 101.6 | 0 | |
| this change, default (wide kernel) | 105.4 | +3.74% | +4.15 +3.85 +3.74 +3.54 +3.34 |
| this change, `MLX_OMARCHY_GDN_PF_WIDE=0` (plain kernel) | 104.2 | +2.76% | +2.76 +3.16 +2.95 +2.36 +2.26 |
| main, `MLX_OMARCHY_GDN_DECODE_TILE=0` | 104.3 | +2.95% | +2.96 +2.96 +2.95 +2.36 +2.94 |
| this change, `MLX_OMARCHY_GDN_DECODE_TILE=1` | 101.4 | -0.1% | -0.2 0.0 -0.1 -0.1 -0.49 |

`speed/summarize-arms.py arms.tee` prints this table from the log. The last row equals main, so the two wheels do not differ by build.

## First screen on 7a09bac (speed/first-screen-7a09bac/)

Same machine, step and design. The wide kernel was selected with `MLX_OMARCHY_GDN_PF_WIDE=1`: +4.23% median over main, rounds +4.65 +4.23 +3.63 +4.34 +3.85; plain +2.67%; main with TILE=0 +2.86%; tiled kernel on the new wheel +0.1%. All digests equal.

## Screens of other switches on main (speed/main-switch-screen-a, b, c)

Eleven, seven and four arms on main a04423e with one environment switch each (arm definitions in `scripts/m2-arms.sh`, `m2-arms2.sh` and `m2-arms3.sh`). The base arm read 101.8 and 101.2 tok/s. `MLX_OMARCHY_GDN_DECODE_TILE=0` gave +2.36%, +2.66% and +2.87%. `MLX_OMARCHY_SDPA_DECODE_NATIVE=0` gave -1.28%. `MLX_OMARCHY_GATED_BARRIERS=0` gave -2.65%. `MLX_OMARCHY_BATCH_WORK=0` gave +0.98% and cost 33 percent of prefill. Chunked argmax and chunked logsumexp gave -0.2% and -0.1%, so the vocabulary tail is not on the critical path. All other arms stayed between -0.6% and 0. Every digest equals `c4ed7eeb93e2eda6`. One arm (`noword`) exited with an error and has no number.

## Numerics (numerics/)

`scripts/armab.sh num` runs a 128-token greedy decode and records the full log-probability vector of 32 steps, with a fresh process per run.

- Plain (`MLX_OMARCHY_GDN_PF_WIDE=0`) against the new default, two prompts: ids equal, max absolute difference 0.0, 0 differing elements. A repeat of the first arm equals it. The dispatch census (`hist-D-10.txt`, `hist-X-10.txt`) shows kernel 480 x198 for plain and kernel 638 x198 for the default, so the arms differ.
- Tiled (`MLX_OMARCHY_GDN_DECODE_TILE=1`) against the new default: kernel 455 x198 against 638 x198. The greedy ids differ within 128 tokens on both prompts and the largest log-probability difference is 0.5. These two kernels are not bit-equal.
- Doctests on the final head (`build/pr3-ticket.out`): `omarchy_gdn_gqa_bf16a_tests` 9 of 9 cases, 143 of 143 assertions; `omarchy_runtime_tests` 59 of 59 cases, 27070 of 27070 assertions. Exit codes 0.

## Build (build/)

`scripts/m2-pr3.sh` clones the branch, resets to 356c2df2, builds the wheel and the two doctest binaries, runs the doctests and the two numerics comparisons. A first run of the same script on commit 631bcf52 failed one doctest assertion. The selector read the head counts after the q and k expansion and took the wide kernel at ratio 3 too. Commit 356c2df2 fixes it. The 631bcf52 results are not part of this directory.

## Limits

One chip (G14C), one model, one thermal state, one boot. Ratio 2 on G14C is not measured. The wide kernel is not measured at ratio 3 and is not the default there.

## Files

SHA256SUMS lists every file here.
