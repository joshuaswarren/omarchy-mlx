# GDN recur32: maskless prefill defaults to mode 0 on G13G

Claim: on M1 (G13G) the default for a maskless gated-delta prefill is now `MLX_OMARCHY_GDN_RECUR32=0`, the route the M1 Max (G13C) already used. `=2` stays available as an opt-in. A masked prefill on G13 parts keeps mode 2 (unchanged).

Why: with mode 2 as the default, the M1 produced different greedy tokens than the M1 Max for the same model, prompts and wheel. Mode 0 matches the M1 Max, and the reported macOS digest (see below). Main's decision: token parity with macOS ranks above the prefill cost.

## Evidence (measured by the parity lane, H392; raw JSON and SHA-256 sums in the lab notebook, `artifacts/jwm1-parity/h392`)

Qwen3.5-9B 4-bit, mlx-lm 0.32.0 with the vendored patch series, wheel `mlx-omarchy 0.32.4.dev202610090923+a48b7cc5`, greedy decode, `score_bench.py` (5 prefill reps at 512 tokens, 10 decode samples), one run per arm.

| chip | `RECUR32` | decode digest (prefix) | decode tok/s median | prefill 512 tok/s median | load1 at end |
|---|---|---|---|---|---|
| M1 (G13G) | unset (mode 2 then) | `83043aa001f15d1b` | 12.358 | 101.659 | 0.35 |
| M1 (G13G) | 0 | `80274aa790426468` | 12.420 | 98.352 | 0.69 |
| M1 Max (G13C) | unset (mode 0) | `80274aa790426468` | 37.194 | 299.454 | 18.54 |
| M1 Max (G13C) | 2 | `83043aa001f15d1b` | 37.176 | 303.771 | 4.76 |

- On the M1, mode 0 costs 3.3 percent of prefill (98.35 vs 101.66) and decode is equal within noise. Single run per arm.
- The M1 Max rows ran with the machine busy (load1 18.5 and 4.8 at the end), so their tok/s are not timing evidence. Their digests are.
- Digest equality: mode 0 gives the same digest on both chips and mode 2 gives the same digest on both chips. That pairs the digest with the route, not with the chip.
- The parity lane reports that `80274aa790426468` is also the macOS digest for this model and these prompts. That macOS run is theirs; I did not repeat it and it is not in the artifact directory above.

## Change and test

- `GatedDeltaUpdate::eval_gpu`: the default is mode 2 only for a masked prefill on a G13 part, else 0.
- New case `a maskless prefill takes the recur32 kernel only when asked` (`omarchy_gdn_decode_batch_tests`): a maskless T=128 prefill with the environment free must not end on the recur32 kernel by default, and `RECUR32=2` must.

## What was and was not run

- The new test case was not yet run on hardware when this receipt was written. On the M1 Max it should pass before and after the change (its default is already mode 0); only the M1 (G13G) can show it fail before and pass after, and that machine was being rebuilt. The M1 leg is owed.
- Mode 2 is still not bit-identical to the scan (subgroup reduction order), so it stays an opt-in lever. Making it bit-identical is not attempted here.
- Masked padded-batch prefill on G13 parts still follows the recur32 order, so token parity with macOS is not claimed for that case.
