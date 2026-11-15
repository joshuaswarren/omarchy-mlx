#!/usr/bin/env python3
"""MLX decode tok/s on the M6 GPU: load once, 512-token prompt, 128 greedy tokens per rep, reps until the time budget ends.
usage: m6-mlxdecode.py MODEL_DIR BUDGET_SECONDS OUT.json   (decode rate = 128 / time after the first token, prefill excluded)
"""
import hashlib
import json
import os
import statistics as st
import sys
import time

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load

model, tok = load(sys.argv[1])
budget = float(sys.argv[2])
ids = tok.encode("The history of France begins in antiquity. ")
PLEN = int(os.environ.get("PLEN", "512"))
prompt = mx.array((ids * (PLEN // len(ids) + 1))[:PLEN])
N = 128


def rep():
    it = generate_step(prompt, model, max_tokens=N, prompt_cache=make_prompt_cache(model))
    toks = [int(next(it)[0])]  # first token ends prefill
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(N - 1):
        toks.append(int(next(it)[0]))
    mx.synchronize()
    return (N - 1) / (time.perf_counter() - t), toks


rep()  # warmup, not recorded
if os.environ.get("FLAGFILE"):
    open(os.environ["FLAGFILE"], "w").close()  # tells the ticket script the timed loop started
end = time.time() + budget
rates = []
hashes = []
while time.time() < end:
    r, toks = rep()
    rates.append(r)
    hashes.append(hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest()[:16])
    print("rep %d %.3f tok/s tokhash %s" % (len(rates), r, hashes[-1]), flush=True)
s = {"n": len(rates), "mean": st.mean(rates), "sd": st.pstdev(rates), "min": min(rates), "max": max(rates),
     "first5": st.mean(rates[:5]), "last5": st.mean(rates[-5:]), "rates": [round(r, 3) for r in rates], "tokhashes": hashes}
json.dump(s, open(sys.argv[3], "w"), indent=1)
print("SUMMARY", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k not in ("rates", "tokhashes")}))
