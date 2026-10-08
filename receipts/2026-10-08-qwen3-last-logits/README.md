# Last-logits prefill for the dense qwen3 family (qwen3.py)

`patches/mlx-lm-last-logits.patch` already makes cached (incremental) prefill compute the quantized
head for the final position only on `qwen3_5` models. The dense `qwen3` family (for example
Qwen3-4B, `model_type: qwen3`, tied embeddings) was not covered: on the 4B,
`MLX_OMARCHY_FULL_LOGITS=1` changed nothing (163.4 vs 163.4 tok/s), because the 512-row head over a
151,936-token vocabulary ran in both arms.

`patches/mlx-lm-last-logits-qwen3.patch` (0.31.3 series) and
`patches/mlx-lm-0.32/mlx-lm-last-logits-qwen3.patch` (0.32 series, mlx-lm 94cdcae) apply the same
change to `mlx_lm/models/qwen3.py`. Whole-sequence calls (`cache=None`: scoring, training) and
`MLX_OMARCHY_FULL_LOGITS=1` keep the full head. `scripts/apply-mlx-lm-patches.sh` applies it
after `mlx-lm-last-logits.patch`.

## Measurements (M1 Max, T8103 / G13G, aurora 12.3, main wheel fbce5f58, idle-guard gpu-turn ticket)

Qwen3-4B-Instruct-2507-4bit, qwen38 protocol (10 prompts x 3 passes, 32 new tokens, 512-token
prefill, temperature 0, CPU PD hold default). Arm B = patched default, arm A = the same patched copy
with `MLX_OMARCHY_FULL_LOGITS=1`, interleaved B A B A B A, load1 before each cell 0.13-0.58.

| arm | prefill tok/s (3 cells) | mean |
|---|---|---|
| B (last-logits) | 178.39 / 178.23 / 178.16 | 178.26 |
| A (full head) | 163.49 / 163.56 / 163.55 | 163.53 |

Gain +9.0 percent, ranges do not overlap. The token digest is identical in every cell
(`eac9fe326212ffa2` at pf512, `42d27a8cbe93df49` at d64); decode (22.62 / 22.63 tok/s) and TTFT in
the d64 cells do not change.

## Numerics

The head runs a different quantized-matmul kernel for 1 row (vector kernel) than for 512 rows
(cooperative matrix), so the logits are not bit-identical: the last-position logits differ by at
most 0.03125 (one bf16 ULP at magnitude 4 to 8), argmax equal. Against an fp64 reference
(dequantized tied embedding times the same hidden state) the two paths have the same error:

| seed | full head max / mean error | sliced head max / mean error |
|---|---|---|
| 0 | 0.02875 / 0.003858 | 0.02875 / 0.003858 |
| 1 | 0.03786 / 0.004029 | 0.03786 / 0.004030 |
| 2 | 0.03287 / 0.004724 | 0.03287 / 0.004724 |

## Apply check

`scripts/apply-mlx-lm-patches.sh` against a clean mlx-lm 0.31.3 venv: first run applies both
last-logits patches, second run reports both "already applied" (rc 0). The 0.32 patch applies with
fuzz 0 to mlx-lm 94cdcae `qwen3.py` and reverses cleanly.

## Scope and limits

Measured on the 4B on this chip only. The other qwen3-family modules (qwen2, llama, qwen3_moe) are
not patched. macOS stock mlx-lm does not have this change either; the 4B prefill on macOS has not
been measured, so this is a speed change, not a parity claim.
