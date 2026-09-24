#!/usr/bin/env python3
"""Teacher-forced logits capture (numpy-stable top-8 per step).

Usage: python logits_gl.py <prompts.jsonl> <out.json> [steps]
Raw greedy re-run via model() + make_prompt_cache, per
receipts/2026-09-22-qwen38-correctness protocol.
"""
import glob
import json
import os
import sys

import numpy as np
import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.models.cache import make_prompt_cache

snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/"
    "snapshots/*"))[0]
model, tok = load(snap)
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 32
prompts = [json.loads(line)["text"] for line in open(sys.argv[1]) if line.strip()][:10]

out = []
for pi, text in enumerate(prompts):
    ids = tok.encode(text)
    cache = make_prompt_cache(model)
    cur = model(mx.array(ids)[None], cache=cache)[:, -1, :]
    rec = {"prompt": pi, "steps": []}
    for _ in range(STEPS):
        l = np.asarray(cur.reshape(-1).astype(mx.float32), dtype=np.float32)
        top = np.argsort(-l)[:8]
        rec["steps"].append({
            "argmax": int(top[0]),
            "top1": float(l[top[0]]),
            "tokens": [int(t) for t in top],
            "logits": [float(l[t]) for t in top],
        })
        cur = model(mx.array([int(top[0])])[None], cache=cache)
    out.append(rec)
    print("prompt", pi, "done", file=sys.stderr)

with open(sys.argv[2], "w") as f:
    json.dump({"model": os.path.basename(snap), "steps": STEPS, "records": out}, f)
print("WROTE", sys.argv[2])
