# M4 Pro (G16S): subgroup-tree RMSNorm for 4096-wide rows

Changed contract: `fast::rms_norm` (and the fused scaled and gated norms) on a bf16 row of 4096 elements with at most 64 rows, on G16S, now runs the
subgroup-tree kernel (`FastRmsNormAppleBF16`, 256 threads, one 32-lane subgroup sum plus eight shared partials, three barriers) instead of the
shared-memory pairing tree (`FastRmsNormBF16`, seven barriers). The width-2048 and width-2560 rows already ran that tree; 4096 (the hidden size of
Qwen3-8B and similar models) reads sixteen contiguous values per thread instead of eight.

- `omarchy::apple_norm_shape_selected` admits 4096 (rows <= 64, like 2048).
- `omarchy::apple_norm_4096_allowed`: default on where the device name contains G16S, `MLX_OMARCHY_NORM_APPLE_4096=1` forces it on and `=0` off on any chip
  (read live). Other parts are unchanged.
- `fast_norm_gated.comp` gets the same sixteen-value contiguous path, so the fused scaled and gated norms keep the same reduction order as the composed
  chain at 4096 (the width-2560 rule from receipts/2026-10-03-norm-apple).
- Test: `apple norm width 4096 selection and fused-vs-composed bits` (capability simulation binary), the width-2560 case with 4096.

The output is not bit-equal to the shared-memory tree (a different summation order); the accuracy section measures what that moves.

## Environment

| | |
| --- | --- |
| source | this branch on origin/main c41c6188 |
| device | MacBook Pro 16-inch, Apple M4 Pro (J616s, T6040), `Apple M4 Pro (G16S B1)` |
| kernel | 7.3.0-m4el2.up105 |
| Mesa | aurora-wip f22f0908629 (Mesa 26.3.0-devel, Borealis), private prefix, default environment otherwise |
| model | mlx-community/Qwen3-8B-4bit, snapshot 545dc4251c05440727734bcd94334791f6ab0192 |
| tool | `ctx_decode_cool.py` style harness: prompt of the target length, 128 greedy tokens, 3 interleaved rounds per arm, one process per run, warmed model load |
| arms | same build; base = `MLX_OMARCHY_NORM_APPLE_4096=0`, this change = default |
| temperature | GPU temperature guard 108 C; hottest run 99 C |

## Decode (Qwen3-8B-4bit, greedy, 128 tokens)

| context tokens | base tok/s (3 runs) | this change tok/s (3 runs) | change (median) |
| --- | --- | --- | --- |
| 1024 | 37.13 / 38.89 / 38.94 | 39.68 / 39.73 / 40.12 | +2.2 % (ranges disjoint) |
| 4096 | 24.47 / 25.10 / 25.62 | 26.02 / 26.03 / 26.16 | +3.7 % (ranges disjoint) |

Generated text hash at 4096 tokens is identical in both arms (4bd7d04576f3). At 1024 tokens the hash differs (2d67f6d58505 base, c7788cad9145 this change): the
summation order moves a near-tie. Raw rows: `data/decode-ab.txt`. The absolute rates are low because this build has none of the other open G16S defaults.

## Accuracy (wikitext-2 test, decode mode)

The route serves rows <= 64, so a prefill check would not reach it. `tools/acc_yardstick.py` prefills 64 tokens of each window, then feeds the true next
token one at a time through two KV caches, one per arm, toggling the switch per forward. 8 windows x 960 steps = 7680 predicted tokens.

| comparison | perplexity | argmax agreement | mean KL | max abs logit diff |
| --- | --- | --- | --- | --- |
| width-4096 subgroup tree vs shared-memory tree | 11.9779 vs 11.9839 | 98.70 % | 9.18e-4 | 6.59 |

For comparison, the accepted two-pass qmm route (#466) reports 98.70 % agreement and 9.64e-4 mean KL on the same kind of window. Raw output:
`data/yardstick.txt`.

## Tests

- `omarchy_capability_sim_tests m1-honeykrisp-fork -tc='apple norm width*'` on the device: 2 cases (2560 and 4096), 30 assertions, all pass.
- `omarchy_fast_ops_tests`: 52 of 52 cases pass; four assertions fail inside `sdpa vjp dk dv match finite differences`, which is marked `may_fail` and does
  not run a norm.

## Reproduce

```
WIKITEXT=/path/to/wikitext-2-test.txt python receipts/2026-11-17-m4-pro-g16s-norm-4096/tools/acc_yardstick.py
MLX_OMARCHY_NORM_APPLE_4096=0 <decode harness>   # base arm; unset for this change
```
