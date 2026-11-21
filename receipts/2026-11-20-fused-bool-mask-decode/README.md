# Fused masked batch-2 BF16 decode, guarded measurement (2026-11-20)

The measurement behind the v0.7.48 and v0.7.49 notes for PR 521 and PR 531.
Two arms on the same build, alternating so a drift in machine state cannot
favour one arm.

## Machine and build

- Apple M1 Max, a 16-inch MacBook Pro.
- Wheel under test: `mlx_omarchy-0.32.4.dev202611201909+db50a6464d4fd3dcfbf07b7a8897e767c8da3eec-cp314-cp314-linux_aarch64.whl`
  (the PR 521 build).
- Model: `mlx-community/Qwen3-4B-Instruct-2507-4bit`, snapshot
  `50d427756c6b1b2fe0c0a10f67fbda1fc8e82c1b`.
- Vulkan driver: the installed Apple ICD, pinned by path for the run.
- Cache cap 80000, greedy decode.

Arm A is the composed route, arm B the fused route from PR 521. Every run
in a pair must return the same token id digest.

## Two streams

| pair | order | A tok/s | B tok/s |
|---|---|---|---|
| 1 | AB | 53.894555602242654 | 69.11999175928929 |
| 2 | BA | 53.95376146455499 | 69.06079505652245 |
| 3 | AB | 54.03166575171092 | 68.93416849439262 |
| 4 | BA | 53.7116041888747 | 68.77548297778824 |
| 5 | AB | 53.793462391114645 | 68.7179772965099 |

Median A 53.894555602242654 tok/s, median B 68.93416849439262 tok/s.
Every run returned the token id digest
`92cfcf51eaf90a47b8932fd63d96e60ad96de542adb38da503ab3c427fcb0f56`.

## One stream

| pair | order | A tok/s | B tok/s |
|---|---|---|---|
| 1 | AB | 41.09427327975817 | 41.12735194325684 |
| 2 | BA | 41.05177687122749 | 41.025686947073275 |
| 3 | AB | 41.06417544842172 | 41.09719997753698 |
| 4 | BA | 41.04565882765643 | 41.107532881287014 |
| 5 | AB | 41.13012395154725 | 41.1296226554837 |

Median A 41.06417544842172 tok/s, median B 41.107532881287014 tok/s.
Every run returned the token id digest
`c244d9a0640702129b51865c82dddc6de8558a8b019ff0e614009d246793f143`.

## What the run checked

- Both arms returned one identical digest per run.
- GPU timeouts before the run 0, after the run 0, delta 0.
- No other GPU was used during either run (foreign GPU counter 0 at start
  and at end).

The two-stream arm is the one that carries the release claim: the fused
route decodes 53.89 to 68.93 tokens per second against the composed
route, with identical token ids. The one-stream arm moves less than one
percent, which is why the note quotes the two-stream numbers.
