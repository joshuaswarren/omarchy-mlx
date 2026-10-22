#!/usr/bin/env python3
"""Teacher-forced decode steps after a long prompt. The prompt (the coreglass step's repeated sentence, round(N / 18) repeats) is run through the model in one prefill pass,
then STEPS tokens of a fixed continuation (the GPL-3 text, from token CONT_OFFSET) are fed one per forward, so the GDN decode kernel runs on a cache that holds the whole prompt.
Writes seg0.npz with ids, target_lp (log-probability of the next continuation token), top_i, top_v (top 20) and lse, like tfref_lse.py.
usage: tfctx.py --model DIR --text FILE --device gpu|cpu --dtype native|float32 --prompt-tokens N --steps S --cont-offset K --out DIR"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
ap.add_argument("--dtype", choices=["native", "float32"], default="native")
ap.add_argument("--prompt-tokens", type=int, default=2048)
ap.add_argument("--steps", type=int, default=128)
ap.add_argument("--cont-offset", type=int, default=0)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import mlx.core as mx  # noqa: E402

if a.device == "cpu":
    mx.set_default_device(mx.cpu)
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

model, tok = load(a.model)
if a.dtype == "float32":
    model.set_dtype(mx.float32)
sentence = "Apple Silicon runs local language models on Linux through Vulkan and a reverse engineered GPU driver. "
prompt = tok.encode(sentence * max(1, round(a.prompt_tokens / 18)))
with open(a.text, encoding="utf-8") as f:
    cont = tok.encode(f.read())[a.cont_offset : a.cont_offset + a.steps + 1]
cache = make_prompt_cache(model)
chunk = 512
for i in range(0, len(prompt) - 1, chunk):
    model(mx.array([prompt[i : min(i + chunk, len(prompt) - 1)]], dtype=mx.int32), cache=cache)
    mx.eval([c.state for c in cache if hasattr(c, "state")])
last_prompt_token = prompt[-1]
feed = [last_prompt_token] + cont[:-1]
tl, ti, tv, ls = [], [], [], []
for i, t_in in enumerate(feed):
    out = model(mx.array([[t_in]], dtype=mx.int32), cache=cache)
    lp = out[0, -1].astype(mx.float32)
    lse_v = mx.logsumexp(lp)
    lp = lp - lse_v
    idx = mx.argpartition(-lp, 19)[:20]
    v = lp[idx]
    order = mx.argsort(-v)
    idx, v = idx[order], v[order]
    target = cont[i]
    mx.eval(lp[target], idx, v, lse_v)
    tl.append(float(lp[target]))
    ti.append(np.array(idx))
    tv.append(np.array(v))
    ls.append(float(lse_v))
os.makedirs(a.out, exist_ok=True)
np.savez(os.path.join(a.out, "seg0.npz"), ids=np.array(cont), target_lp=np.array(tl), top_i=np.stack(ti), top_v=np.stack(tv), lse=np.array(ls))
print(f"TFCTX prompt {len(prompt)} steps {len(tl)} device {a.device} dtype {a.dtype} mlx {mx.__version__}", flush=True)
