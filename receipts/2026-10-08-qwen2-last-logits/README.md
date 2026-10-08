# Last-logits prefill for the dense qwen2 family (qwen2.py): CANDIDATE, NOT LANDED

Branch `agent/Qwen2LastLogits` only. Not on main: the numerics check below does not meet the
landing rule (reordered arithmetic needs equal-or-better fp64-reference error).

`patches/mlx-lm-last-logits-qwen2.patch` (0.31.3 series) and
`patches/mlx-lm-0.32/mlx-lm-last-logits-qwen2.patch` (0.32 series) apply the qwen3 change
(`receipts/2026-10-08-qwen3-last-logits`) to `mlx_lm/models/qwen2.py`: cached prefill computes the
head for the final position only; `cache=None` and `MLX_OMARCHY_FULL_LOGITS=1` keep the full head.

## Speed (M1 Max, aurora 12.3, main wheel, idle-guard gpu-turn ticket)

Qwen2.5-0.5B-Instruct-4bit (tied head, vocab 151936, hidden 896), qwen38 protocol, pf512, 10 prompts x
3 passes, 32 new tokens, load1 before each cell 0.17-0.28, interleaved B A B A B A:

| arm | prefill tok/s | mean |
|---|---|---|
| B (last-logits) | 1642.18 / 1641.49 / 1644.55 | 1642.7 |
| A (full head) | 1274.79 / 1274.63 / 1275.34 | 1274.9 |

+28.9 percent, no range overlap. Digest identical (`c3eab16c8fb2ecca`), decode 126.6-126.8 tok/s in both
arms (unchanged).

## Numerics: the fp64-reference error is NOT equal

Last-position logits, 512-token prompt, fp64 reference = dequantized tied embedding times the same
hidden state:

| seed | full head max / mean error | sliced head max / mean error |
|---|---|---|
| 0 | 0.00383 / 0.000311 | 0.01380 / 0.003337 |
| 1 | 0.00348 / 0.000367 | 0.00936 / 0.001889 |
| 2 | 0.00449 / 0.000438 | 0.01371 / 0.002589 |

The sliced head (1 row) is 3 to 10 times less accurate than the full head (512 rows) on this model:
mean error 0.0019-0.0033 against 0.0003-0.0004. Max difference between the two paths 0.0137, argmax
equal. For the qwen3 patch the two errors were equal (4B). The single-row quantized matmul is the same
kernel decode uses for every generated token, so the new prefill head matches the decode head, but it
is less accurate than the stock prefill head here. Why the 0.5B differs (K = 896 = 14 groups of 64 is
the suspect; not investigated) is open.

Decision needed before landing: accept the larger head error for a +28.9 percent prefill gain on
0.5B-class qwen2 models, or first look at the single-row kernel's accumulation for K = 896.
