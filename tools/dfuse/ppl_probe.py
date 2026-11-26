#!/usr/bin/env python3
"""Perplexity probe for the numerics gate: mean NLL per token of a fixed
text under teacher forcing, computed identically in two venvs.
usage: ppl_probe.py <model-dir> <prompts.jsonl>
"""
import glob
import json
import sys

import mlx.core as mx

model_dir = sys.argv[1]
prompts_path = sys.argv[2]

from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load(model_dir)
lines = [json.loads(l) for l in open(prompts_path)][:4]
texts = [l.get("prompt") or l.get("text") for l in lines if (l.get("prompt") or l.get("text"))]

total_nll = 0.0
total_tokens = 0
for t in texts:
    ids = tokenizer.encode(t)[:256]
    if len(ids) < 8:
        continue
    x = mx.array([ids[:-1]])
    y = mx.array(ids[1:])
    cache = make_prompt_cache(model)
    logits = model(x, cache=cache)
    logprobs = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
    nll = -mx.take_along_axis(logprobs, y[None, :, None], axis=-1).sum()
    mx.eval(nll)
    total_nll += float(nll)
    total_tokens += len(ids) - 1

ppl = float(mx.exp(mx.array(total_nll / max(total_tokens, 1))))
print(json.dumps({"mean_nll_per_token": total_nll / max(total_tokens, 1),
                  "ppl_of_mean_nll": ppl,
                  "tokens": total_tokens, "texts": len(texts)}))
