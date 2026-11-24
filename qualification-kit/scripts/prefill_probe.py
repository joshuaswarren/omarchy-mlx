#!/usr/bin/env python3
# prefill_probe.py MODEL_DIR N_TOKENS — one model forward pass at exactly
# N_TOKENS prompt tokens (pure prefill, no decode step). Run under
# MLX_OMARCHY_TRACE_DISPATCH=1; stderr carries the [rtmod] DISPATCH lines.
import sys

import mlx.core as mx
from mlx_lm import load

model_path, n_tokens = sys.argv[1], int(sys.argv[2])
model, tokenizer = load(model_path)

# deterministic prompt: a fixed sentence repeated, tokenized once, truncated.
# pad right if a tokenizer renders it short — the length is exact either way.
seed = "The quick brown fox jumps over the lazy dog while the compiler optimizes the kernel."
ids = tokenizer.encode(seed)
while len(ids) < n_tokens:
    ids = ids + ids
ids = ids[:n_tokens]

tokens = mx.array([ids])
out = model(tokens)
mx.eval(out)
# sanity line on stdout: next-token argmax proves the run reached the head
mx.eval(out)
print(f"prefill_probe model={model_path} tokens={len(ids)} argmax={int(mx.argmax(out[0, -1]))}")
