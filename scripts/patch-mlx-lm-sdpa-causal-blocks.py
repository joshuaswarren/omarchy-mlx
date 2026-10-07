#!/usr/bin/env python3
"""Opt-in block-causal prompt attention (prefill lever, default OFF).

Causal prompt attention never reaches a fused kernel on the omarchy backend:
the flash prefill route serves non-causal hd128 only, so a causal prompt
(Qwen3.8: hd256, 8 query heads) runs the composed route over the FULL
[heads, L, L] f32 score square and then masks its upper half away.

With MLX_OMARCHY_SDPA_CAUSAL_BLOCK=<rows> set, scaled_dot_product_attention
splits a causal prompt into row blocks; each block attends only to its key
prefix through mx.fast.scaled_dot_product_attention(mask="causal"), whose
causal mask is bottom-right aligned (the same alignment chunked prefill
already relies on). Every row sees exactly the keys it saw before, so the
dropped work is the fully masked tail: about half of the QK/PV FLOPs and of
the score memory at long L. Engages only for mask == "causal", no sinks, a
non-quantized cache and q_len > rows; everything else, and the variable
unset or 0, is the upstream call.

Usage: python3 patch-mlx-lm-sdpa-causal-blocks.py /path/to/venv
"""
import ast
import glob
import sys

MARKER = "MLX_OMARCHY_SDPA_CAUSAL_BLOCK"

OLD = """    else:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
"""
NEW = """    elif (
        _SDPA_CAUSAL_BLOCK > 0
        and isinstance(mask, str)
        and mask == "causal"
        and sinks is None
        and queries.shape[2] > _SDPA_CAUSAL_BLOCK
    ):
        return _causal_blocks(queries, keys, values, scale, _SDPA_CAUSAL_BLOCK)
    else:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )


# mlx-omarchy: opt-in block-causal prompt attention. Each row block attends to
# its key prefix only (bottom-right aligned causal mask), skipping the fully
# masked tail of the score square. MLX_OMARCHY_SDPA_CAUSAL_BLOCK=<rows>; unset
# or 0 keeps the single full-square call.
_SDPA_CAUSAL_BLOCK = int(os.environ.get("MLX_OMARCHY_SDPA_CAUSAL_BLOCK", "0") or "0")


def _causal_blocks(queries, keys, values, scale, rows):
    L, offset = queries.shape[2], keys.shape[2] - queries.shape[2]
    blocks = []
    for start in range(0, L, rows):
        end = min(start + rows, L)
        blocks.append(
            mx.fast.scaled_dot_product_attention(
                queries[:, :, start:end],
                keys[:, :, : offset + end],
                values[:, :, : offset + end],
                scale=scale,
                mask="causal",
            )
        )
    return mx.concatenate(blocks, axis=2)
"""

venv = sys.argv[1] if len(sys.argv) > 1 else "."
site = glob.glob(venv.rstrip("/") + "/lib/python3*/site-packages/mlx_lm/models")
if not site:
    sys.exit("mlx_lm/models not found under " + venv)
q = site[0] + "/base.py"
text = open(q).read()
if MARKER in text:
    print("already patched:", q)
    sys.exit(0)
if text.count(OLD) != 1:
    sys.exit("unrecognized scaled_dot_product_attention in " + q + "; refusing to patch")
text = text.replace(OLD, NEW, 1)
if "\nimport os\n" not in text:
    text = text.replace("\nimport ", "\nimport os\nimport ", 1)
ast.parse(text)
open(q, "w").write(text)
print("patched:", q)
