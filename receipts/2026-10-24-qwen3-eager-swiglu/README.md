# Dense Qwen3 decode folds SwiGLU into the gate and up matvec (2026-10-24)

The mlx-lm patch set now rewrites the dense Qwen3 MLP so that single-token rows compute `gate * sigmoid(gate) * up` as eager operations. The GEMV planner folds that product into the gate and up matvec dispatch, so one dispatch per layer disappears: 36 dispatches per Qwen3-4B decode token, out of the 371 that an earlier decode census of the unpatched path counted (that census is not part of this receipt). Qwen3-4B 4-bit decodes 1.65 percent faster on the Apple M2 Max and 1.24 percent faster on the Apple M1 Max. The token digest is identical in all 20 samples.

The +3 percent bar of the nightly rule is not met. The change ships on the rule written in `prereg.txt` before the run: the output is bit identical, the dispatch count drops, and no round is slower than its base.

## Why the fold did not fire before

The backend already folds `silu(gate) * up` into the gate and up matvec (`swiglu_plans` in `fused_chain.cpp`, switch `MLX_OMARCHY_FUSED_GEMV_SWIGLU`). mlx-lm calls the compiled `swiglu` of `activations.py`. A compiled function reaches the backend as one opaque node, so the planner never matches it and a standalone `SwigluBF16` dispatch ran in every layer. The Qwen3-Next patcher already removes the same barrier for that family. This change adds the dense Qwen3 file `qwen3.py`.

## What the patcher does

`scripts/patch-mlx-lm-qwen3-eager-swiglu.py` has two anchors in `qwen3.py`: the `MLP.__call__` body and the line `class MLP(nn.Module):`. It refuses to patch when either anchor is missing or not unique, and a second run changes nothing. Rows with `x.shape[-2] == 1` take the eager product with the operand order of the compiled function. All other rows call the compiled `swiglu` unchanged. `MLX_OMARCHY_EAGER_SWIGLU=0` keeps the compiled function for decode too. The flag line imports `os` under its own name, so the patch does not depend on another patcher. `scripts/apply-mlx-lm-patches.sh` runs it after the Qwen3-Next patcher.

## Bit check

`scripts/check_eager_swiglu_bits.py --family qwen3` compares the eager product and the whole 4-bit MLP with the compiled path on random bf16 data at hidden size 2560 and intermediate size 9728. It runs 64 draws at scales 1 and 8 and batches 1 and 3. Each chip made 512 comparisons and found 0 mismatching values (`tables/bits-apple-m2-max.tsv`, `tables/bits-apple-m1-max.tsv`).

The same run counts dispatches. The compiled path has 47 and the eager path has 46 on both chips, and kernel 419 (`SwigluBF16`) is the missing one (`tables/dispatch-apple-m2-max.tsv`, `tables/dispatch-apple-m1-max.tsv`). The count confirms that the patched file was in use, so the equal bits are not an identity comparison.

## Decode screen

Each chip ran five interleaved rounds. One round runs both arms once, in a rotated order, one process per sample, 128 generated tokens, greedy. The prompt is the ledger step with nominal 477 tokens, which is 443 real tokens. The base arm uses the installed mlx-lm. The eager arm puts a copy of the package with the patched `qwen3.py` first on the Python path. Both arms use the same wheel, `mlx_omarchy 0.32.4.dev202610241001+1edd56d` with libmlx `b920984dcc4554ec`. The log of each screen names the sha of the installed and the patched `qwen3.py`.

Decode tokens per second, base then eager:

| Round | Apple M2 Max | Gain | Apple M1 Max | Gain |
|---|---|---|---|---|
| 1 | 65.80, 67.60 | +2.74% | 64.50, 65.10 | +0.93% |
| 2 | 66.50, 67.70 | +1.80% | 64.30, 65.00 | +1.09% |
| 3 | 66.50, 67.40 | +1.35% | 64.20, 64.90 | +1.09% |
| 4 | 66.60, 67.50 | +1.35% | 64.30, 65.10 | +1.24% |
| 5 | 66.30, 67.60 | +1.96% | 64.40, 65.10 | +1.09% |
| Median | 66.50, 67.60 | +1.65% | 64.30, 65.10 | +1.24% |

The token digest is `8e68ea1ec2d62fcc` in all 10 samples of each chip. Prefill speed is unchanged. The base reading of round 1 on the Apple M2 Max is the lowest of its arm and has a lower prefill speed than the other base samples, so the +2.74 percent of that round is the least reliable. The other four rounds read +1.35 to +1.96 percent. Tables: `tables/decode-apple-m2-max-477.tsv`, `tables/decode-apple-m1-max-477.tsv`. Logs and one sample output per chip are in `run-logs/`.

The gain is about 7 microseconds per removed dispatch on the Apple M2 Max: 66.5 tokens per second is 15.04 ms per step, 67.6 is 14.79 ms, and the 0.245 ms difference divided over 36 layers is 6.8 microseconds.

## What is not covered

The bit check uses bf16 weights quantized to 4 bits with group size 64. Models in fp16 and other group sizes use the same patch and are not tested here. Only the dense Qwen3 file is patched. Other model families that call the compiled `swiglu` keep their dispatch. The 13-inch Apple M1 was not screened. The boot id is redacted in the logs.
