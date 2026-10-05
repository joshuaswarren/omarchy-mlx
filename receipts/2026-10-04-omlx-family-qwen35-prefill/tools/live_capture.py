#!/usr/bin/env python3
"""Capture the sdpa OUTPUTS from a live model forward (one arm per
process via ARM + MLX_OMARCHY_SDPA_PREFILL_FLASH256 env)."""
import os
import sys

import numpy as np

w, model_id = sys.argv[1], sys.argv[2]
mode = os.environ.get("ARM", "flash")

import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402

model, tokenizer = load(model_id)
outs = []
orig = mx.fast.scaled_dot_product_attention


def spy(q, k, v, **kw):
    out = orig(q, k, v, **kw)
    if q.ndim == 4 and q.shape[2] > 1:
        mx.eval(out)
        outs.append(np.array(out.astype(mx.float32)))
    return out


mx.fast.scaled_dot_product_attention = spy
ids = tokenizer.encode("The capital of France is")
logits = model(mx.array([ids]))
mx.eval(logits)
for i, o in enumerate(outs):
    np.save(f"{w}/live.{mode}.call{i}.npy", o)
np.save(f"{w}/live.{mode}.lastlogits.npy",
        np.array(logits[0, -1].astype(mx.float32)))
print(f"[{mode}] saved {len(outs)} live outputs")
