"""Prefill tok/s at a fixed prompt length in one process.

usage: prefill_bench.py <repo> <prompt_tokens> <runs>
One full forward over a fixed synthetic prompt with a fresh cache: one
warmup, then <runs> timed passes. Prints one JSON line with every pass,
the median and the minimum, peak GPU memory and the argmax of the last
position's logits (a cross-arm sanity check).
"""
import json
import resource
import statistics
import sys
import time

import prov

prov.check()
import mlx.core as mx  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

repo, n, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
model, tok = load(repo)
text = ("The quick brown fox jumps over the lazy dog while the rain falls "
        "on the old stone bridge. ") * 400
ids = mx.array(tok.encode(text)[:n])[None]
assert ids.shape[1] == n


def run():
    cache = make_prompt_cache(model)
    t = time.perf_counter()
    y = model(ids, cache=cache)
    mx.eval(y)
    return time.perf_counter() - t, int(mx.argmax(y[0, -1]).item())


_, arg0 = run()
ts = []
for _ in range(runs):
    t, a = run()
    ts.append(t)
peak = getattr(mx, "get_peak_memory", None)
print(json.dumps({
    "model": repo.split("/")[-1], "prompt_tokens": n, "runs": runs,
    "tok_s": [round(n / t, 1) for t in ts],
    "median_tok_s": round(n / statistics.median(ts), 1),
    "best_tok_s": round(n / min(ts), 1), "last_argmax": arg0,
    "peak_gpu_gib": round(peak() / 2**30, 3) if peak else None,
    "max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
}), flush=True)
