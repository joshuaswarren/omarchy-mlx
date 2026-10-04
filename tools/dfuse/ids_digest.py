#!/usr/bin/env python3
"""Greedy-id digests for non-regression: sha256 of the generated ids per
prompt (plain argmax loop, one decode call per token).

usage: ids_digest.py <model-dir> <tokens> <prompt-index...>
"""
import hashlib
import json
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model_dir = sys.argv[1]
ntoks = int(sys.argv[2])
pidx = [int(p) for p in sys.argv[3:]]
texts = [json.loads(l)["text"] for l in
         open("/var/tmp/Jw16NumericsGpu/qwen38-2b-prompts-10.jsonl") if l.strip()]
model, tokenizer = load(model_dir)
for p in pidx:
    ids = tokenizer.encode(texts[p])
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids], dtype=mx.int32), cache=cache)
    toks = []
    for _ in range(ntoks):
        tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        toks.append(tok)
        logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)
    h = hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]
    print(f"p{p} n={ntoks} sha={h}", flush=True)
