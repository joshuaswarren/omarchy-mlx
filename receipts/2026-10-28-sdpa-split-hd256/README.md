# Split-K decode attention for head width 256 (2026-10-28)

Decode attention for bf16, head width 256 and one query token now takes the two-dispatch split-K route that head width 128 got on 2026-10-23. The Qwen3.5 and Qwen3.8 full-attention layers have head width 256. Their decode attention read the KV cache at 3.1 GB/s on the 13-inch M1. The route reads it at 41.8 GB/s at 8192 keys and 50.4 GB/s at 32768 keys (copy bandwidth on the same machine: 59.1 GB/s).

Qwen3.5-9B 4-bit decodes 2.17 times as fast after an 8192-token prompt. Qwen3.8-2B 4-bit decodes 2.46 times as fast after 8192 tokens and 4.71 times as fast after 32768 tokens. Prefill is unchanged. Short contexts cost 1 to 2 percent on the 2B (6 extra dispatches per token) and nothing measurable on the 9B.

`MLX_OMARCHY_SDPA_DECODE_SPLIT=0` restores the previous route at run time. The route is the default from the same per-part key counts as head width 128 (`sdpa_split_min_keys`: 128 on the Apple M1, 320 on the M1 Max and M2 Max).

## What changed

`sdpa_decode_split_p1.comp` and `sdpa_decode_split_p2.comp` take `-DHD=256`: each of the 128 threads owns D / 128 output dimensions in the value phase. Head width 128 is the same arithmetic in the same order. Five new kernels (pass 1 for groups of 1, 2, 4 and 8 query heads per key-value head, and pass 2), routed in `primitives.cpp` when `head_dim == 256`. The value-phase variant of pass 1 stays at head width 128.

## Pre-registration

`PREREG-sdpa-hd256-ab-20261028.md` was written before the first run of the new kernel on a device. Its bounds: H1 the 9B at 8192 is at least 1.25 times as fast, H2 the 2B at 32768 is at least 1.5 times as fast, H3 short contexts are at least 0.97 times as fast, tokens agree on at least 44 of 48, mean NLL difference within 0.01 nats per token. All five hold on the 13-inch M1.

## Machine and build

13-inch M1 (G13G B1), 16 GB, Honeykrisp Mesa 26.3.0-devel (git-69bde15340), packaged ICD. MLX built from this branch (head `ff4b52ac`). One build serves both arms: arm A sets `MLX_OMARCHY_SDPA_DECODE_SPLIT=0`, arm B leaves it unset. Arms ran interleaved A, B, A, B, A, B. The first build had no CMAKE_BUILD_TYPE; the Release rebuild gives the same long-context numbers (the cells are GPU-bound) and short-context numbers at parity with the packaged wheel. Models are the 4-bit MLX checkpoints `mlx-community/Qwen3.5-9B-4bit` and `SiddhJagani/Qwen3.8-2B-mlx-4Bit`. The runner prefills N random token ids in chunks of 512, then decodes 128 tokens greedily. Logs are in `logs/` with `SHA256SUMS`.

The PROVENANCE lines in the logs print the version of the installed `mlx-omarchy` distribution (the packaged one, `0.32.4.dev202610050725+5c15fba`); the `mlx file` field on the same line names the module that ran, the branch build.

## Decode tokens per second, Release build, three interleaved pairs

| Cell | A (route off) | B (route on) | B over A |
| --- | --- | --- | --- |
| Qwen3.5-9B, 8192 tokens | 4.70, 4.69, 4.70 | 10.24, 10.19, 10.11 | 2.17 |
| Qwen3.8-2B, 8192 tokens | 13.97, 13.97, 13.96 | 34.34, 34.38, 34.47 | 2.46 |
| Qwen3.8-2B, 32768 tokens | 4.65, 4.65, 4.65 | 21.91, 21.88, 21.91 | 4.71 |

The first build (no Release flag) gave 10.21, 10.12, 10.18 for the 9B, 34.43, 34.31, 34.43 for the 2B at 8192 and 21.90, 21.96, 21.96 for the 2B at 32768. Arm A reproduces the packaged wheel's numbers at these contexts (4.70, 13.96 and 4.65).

## Short context (pre-registered H3), Release build, two interleaved repetitions

| Model, context | A | B | B over A |
| --- | --- | --- | --- |
| Qwen3.8-2B, 128 | 41.65, 41.89 | 40.90, 40.87 | 0.982, 0.976 |
| Qwen3.8-2B, 256 | 41.31, 41.55 | 40.72, 40.79 | 0.986, 0.982 |
| Qwen3.5-9B, 128 | 11.78, 10.98 | 11.75, 10.92 | 0.997, 0.995 |
| Qwen3.5-9B, 256 | 11.64, 11.70 | 11.72, 11.52 | 1.007, 0.985 |

The route adds 6 dispatches per token on the 2B (265 to 271) and 8 on the 9B (399 to 407). The 9B figures scatter by about 5 percent between repetitions (11.78 and 10.98 for the same arm), more than the difference between the arms.

## Attention alone

`sdpa-decode-bench.py`: one query token against a bf16 cache, per call, the shapes of the three models (head width 256 with 16 and 4 heads for the 9B, 8 and 2 for the 2B; head width 128 with 32 and 8 for Qwen3-4B).

| Shape, keys | A: microseconds, GB/s | B: microseconds, GB/s |
| --- | --- | --- |
| 9B, 2048 | 1259, 6.7 | 238, 35.2 |
| 9B, 8192 | 10783, 3.1 | 803, 41.8 |
| 9B, 32768 | 41877, 3.2 | 2662, 50.4 |
| 2B, 8192 | 5411, 3.1 | 487, 34.5 |
| 2B, 32768 | 20914, 3.2 | 1434, 46.8 |
| Qwen3-4B, 8192 | 11297, 3.0 | 768, 43.7 |

The Qwen3-4B rows are the head-width-128 route of 2026-10-23, which no release carries yet.

## Correctness

- `omarchy_sdpa_split_tests` on the M1: 11 of 11 cases, 791 assertions, including head width 256 with groups 1, 2, 4 and 8, strided cache views, large magnitudes and one-hot values, each against a float64 reference and with the route taken (two dispatches). `omarchy_primitive_tests`: 107 of 107, 2,768,377 assertions.
- Greedy decode of a fixed 419-token prompt, 48 new tokens, route off against route on: 48 of 48 token ids equal for the 2B and for the 9B.
- Teacher-forced decode over real text (the repository docs), 3000 prompt tokens, 400 scored decode steps. Mean negative log-likelihood, route on minus route off: -0.0018 nats per token for the 2B (3.0938 to 3.0920) and -0.0004 for the 9B (2.6662 to 2.6658). The largest single-step difference is 0.196 and 0.145 nats. Next-token agreement with the text: 187 and 190 of 400 (2B), 195 and 195 (9B).

The route sums in float32 in a different order from the fused kernel, so the output is not bit-equal to it.

## Against the packaged wheel (same session, three arms interleaved, three repetitions)

P is the packaged wheel (0.32.4.dev202610050725), RA this branch's Release build with the route off, RB the same build with the route on (`logs/relcmp-*.log`). Decode tokens per second:

| Cell | P | RA | RB |
| --- | --- | --- | --- |
| Qwen3.5-9B, 256 | 11.88, 11.94, 11.88 | 11.72, 11.72, 11.71 | 11.69, 11.73, 11.72 |
| Qwen3.5-9B, 8192 | 4.72, 4.72, 4.72 | 4.69, 4.68, 4.69 | 10.07, 10.07, 10.08 |
| Qwen3.8-2B, 256 | 40.45, 40.37, 40.54 | 41.43, 41.70, 41.67 | 40.74, 40.64, 40.80 |
| Qwen3.8-2B, 8192 | 13.79, 13.77, 13.79 | 13.77, 13.77, 13.76 | 33.43, 33.53, 33.36 |

At 256 tokens the 9B on this branch is 1.4 percent below the packaged wheel (it makes 23 submissions per token against 6, the board budget of 3b24ad416 and a04423e42) and the 2B is at or above it. At 8192 tokens RB is 2.13 times (9B) and 2.43 times (2B) the packaged wheel.

## Head width 128 is unchanged

The shaders keep the previous head-width-128 code under `#if HD == 128`. Compiled with the same glslc, pass 1 for groups 1, 2, 4 and 8 and pass 2 are byte-identical SPIR-V to the shaders before this change. The first version of this change shared the loops between the two widths and measured 7 percent slower at 8192 keys at width 128 (`logs/hd128-*.log`: 754 against 704 microseconds per call, three interleaved repetitions); that version is not the one in this branch. The width-256 SPIR-V of this branch is byte-identical to the version the device runs above used.

## Not measured

The 14-inch M2 Max (a run is queued; its numbers will be added in a separate receipt). Contexts above 32768 tokens. Qwen3-4B end to end after this change (its route is unchanged here).
