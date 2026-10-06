"""MatmulGap H6: one Qwen3.8-2B 4-bit cell - pf512 prefill time, 32-token
greedy decode tok/s, and hashes of the prefill last logits and the generated
tokens. usage: lm_cell.py <model dir>"""
import hashlib
import json
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, _ = load(sys.argv[1])
ids = mx.array([[(i * 37) % 20000 + 100 for i in range(512)]])


def run():
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(ids, cache=cache)
    mx.eval(logits)
    t1 = time.perf_counter()
    last = logits[0, -1].astype(mx.float32)
    y = mx.argmax(logits[:, -1], axis=-1)
    toks = []
    t2 = time.perf_counter()
    for _ in range(32):
        out = model(y[:, None], cache=cache)
        y = mx.argmax(out[:, -1], axis=-1)
        mx.eval(y)
        toks.append(int(y.item()))
    t3 = time.perf_counter()
    return (t1 - t0) * 1e3, 32 / (t3 - t2), last, toks


run()  # warm-up: pipelines compile here
pf_ms, tps, last, toks = run()
print(json.dumps({
    "pf512_ms": round(pf_ms, 2), "decode_tok_s": round(tps, 2),
    "logits_sha": hashlib.sha256(np.array(last).tobytes()).hexdigest()[:16],
    "tokens_sha": hashlib.sha256(json.dumps(toks).encode()).hexdigest()[:16]}))
