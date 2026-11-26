# Decode `gather_qmm` binds scales, biases and index arrays in place (PR 124, main `18059c80`)

Decode-shape `gather_qmm` (routed-MoE experts, one row) used to copy every expert's whole scales and biases tensors into a staging buffer on each call, 15.9 MB per call at 128 experts. `GatherQmmSubDirect{F16,BF16}` read the tensors where they are. The arithmetic is the same kernel's; `MLX_OMARCHY_GATHER_QMM_DIRECT=0` restores the staged route in the same wheel.

Every figure below names its chip, driver and governor. The lab's measurement notebook holds the raw logs; the repository holds this summary and one bundle hash.

## Conditions

| Rows | Chip | Driver | Governor | Kernel |
|---|---|---|---|---|
| per-call and decode-step rows measured before 2026-10-12 01:23Z | 14-inch M2 Max (G14C), 16-inch M1 Max (G13C), 13-inch M1 (G13G) | packaged Honeykrisp library `40282811c6e14ef9` (Mesa git `69bde15340`) | `schedutil` (the fleet baseline before the switch to `performance`; the tickets did not read it back) | 7.1.12-2 12.6 on all three chips (read in the ticket logs) |
| M2 Max ablation and the same-block block | 14-inch M2 Max (G14C) | same library `40282811c6e14ef9` | `performance` (read back) | 7.1.12-2 `20261011.15` |

Wheel under test for the per-call and decode-step rows: `0.32.4.dev202610111829+d2a3067a` (PR 124 library code before the test and docs commit), libmlx.so `80b7e6dd1fbb8c9d`. Wheel for the same-block block: `0.32.4.dev202610120329+18059c80` (sha256 `2c124f9cff3837c8b221ba49a5583406ace2c10a035722ccc4f1bb396c1541f7`).

## Bit identity

50 cases per chip, 0 differing, the same wheel with the route toggled: bf16 and f16; gemma-4-26B's up/gate and down shapes, 64-expert and 8-expert tensors; 1, 8 and 32 indices; weight, scales, biases and index views at a non-zero offset; explicit lhs and rhs indices. The final-step logits of gemma-4-26B-A4B (4 arms) and Qwen3.6-35B-A3B (8 arms) are identical between the routes. `omarchy_matmul_family_tests` at head `5f711f32`: 35 of 35 cases, 82,942,968 assertions, 0 failed, on the 14-inch M2 Max (kernel `20261011.15`, governor `performance`).

## Per call (8 of 128 experts, gemma's shapes, 3 blocks of staged, direct, direct, staged; `schedutil`)

| chip | up/gate staged to direct | down staged to direct |
|---|---|---|
| M2 Max | 496.6 to 290.7 us (1.71x) | 598.3 to 385.7 us (1.55x) |
| M1 Max | 531.5 to 348.2 us (1.53x) | 696.6 to 502.4 us (1.39x) |
| M1 | 2570.1 to 1280.6 us (2.01x) | 3074.9 to 1766.4 us (1.74x) |

## Decode, M2 Max

Two kinds of number, not interchangeable:

1. One decode step after a 512-token prefix, the median over 160 steps, arms ordered staged, direct, direct, staged (`schedutil`, kernel 12.6):

| model | staged | direct | ratio |
|---|---|---|---|
| gemma-4-26B-A4B 4-bit | 90.59 ms | 71.85 ms | 1.26x |
| Qwen3.6-35B-A3B 4-bit | 82.96 ms | 49.70 ms | 1.67x |

2. The same arms measured as decode tokens per second over greedy generations, three interleaved rounds, with the filler load on the Neural Engine running (`performance`, kernel `20261011.15`, wheel `18059c80`): gemma-4-26B-A4B, staged route 14.27 / 14.28 / 14.28, direct route 19.77 / 19.71 / 19.72, old wheel (main `e57cb301`) 14.23 / 14.24 / 14.01. Direct over staged 1.381 (control spread 0.1 percent); staged over the old wheel 1.004. Prefill staged 227.70 / 227.40 / 225.68, direct 226.72 / 227.85 / 225.42 (0.997). That block's raw logs are summarized in a bundle whose `SHA256SUMS` file hashes to `6ec23d073210bbabd91bab880d658a7e2659abb4efbc709aa72ac009453646f5`.

The saving is the stable quantity: 18.7 ms per token in (1), 18.4 ms in the ablation below, 19.3 ms in (2) (70.03 ms against 50.71 ms per token, from the medians 14.28 and 19.72). The ratio grows as the rest of the step shrinks.

## Where the step goes after this change (M2 Max, gemma-4-26B-A4B, `performance`, kernel `20261011.15`)

Ablation, 24 steps after a 512-token prefix, direct route: full step 59.3 ms; experts about 26 ms (44.7 and 42.4 percent in two arms); router 9 to 11 percent; the remaining 46 percent (27.5 ms) is attention, the dense branch, norms and the head. Staged route in the same harness: step 77.7 ms, experts 57.7 percent.

## Command

`MLX_OMARCHY_GATHER_QMM_DIRECT=0` (staged) or unset (direct), one process per arm, `mlx_lm` model loaded from the HF cache, decode-step harness: 512-token prefix of fixed text, fixed next-token ids, logits evaluated each step, the median per prefix, 5 prefixes of 32 steps (decode-step rows) or 3 prefixes of 24 steps (ablation). The harness script is not in this repository.

## Not covered

- The M1 Max and M1 decode steps: no model was run on either chip.
- A word-per-lane kernel for the same shape (4.59x per call and 1.45x on the decode step on the M2 Max) was built and stopped: its model output differs from the shipped kernels' by more than the registered threshold, and its likelihood over 1,024 teacher-forced positions is 0.025 nats worse than the shipped kernel with a standard error of 0.014. It is not in main.
- Any model other than gemma-4-26B-A4B and Qwen3.6-35B-A3B.
