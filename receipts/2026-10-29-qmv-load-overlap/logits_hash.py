"""Greedy decode of a short prompt for 256 steps, hashing the FULL logit vector of every step (bf16 bytes, exact) and the token ids.

Two builds are identical for this purpose when the final line is equal. Not timed. Usage: logits_hash.py MODEL_DIR LABEL [steps]
"""
import glob
import hashlib
import importlib.metadata as md
import os
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model_dir, label = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 256
base = os.path.dirname(mx.__file__)
libs = {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
        for p in sorted(glob.glob(os.path.join(base, "lib", "libmlx*"))) if os.path.isfile(p)}
print("IDENT", label, "mlx_omarchy", md.version("mlx_omarchy"), libs, flush=True)
model, tok = load(model_dir)
prompt = tok.encode("What is the capital of France? Answer in one word.")
cache = make_prompt_cache(model)
x = mx.array(prompt)[None]
logit_hash = hashlib.sha256()
ids = []
for step in range(steps):
    logits = model(x, cache=cache)[:, -1, :]
    mx.eval(logits)
    logit_hash.update(repr(logits.view(mx.uint16).tolist()).encode())
    nxt = int(mx.argmax(logits, axis=-1).item())
    ids.append(nxt)
    x = mx.array([[nxt]])
print("IDENT", label, "steps", steps, "ids_sha", hashlib.sha256(repr(ids).encode()).hexdigest()[:16],
      "logits_sha", logit_hash.hexdigest()[:16], "first", ids[:6], flush=True)
