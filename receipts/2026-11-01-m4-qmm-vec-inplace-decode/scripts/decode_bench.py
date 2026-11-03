"""Decode and prefill speed of one model in one process.

usage: decode_bench.py <repo> <max_tokens> <runs>
Chat-templated fixed prompt, greedy sampling, one warmup generation, then
<runs> timed generations. Prints one JSON line: per-run decode and prompt
tok/s, their medians, the generated token ids' sha256 (first 16 hex), peak
GPU memory reported by MLX and the process max RSS.
"""
import hashlib
import json
import resource
import statistics
import sys
import time

import prov

prov.check()
import mlx.core as mx  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

PROMPT = "Explain in plain language how a rainbow forms. Give four distinct sentences."
repo, max_tokens, runs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
greedy = make_sampler(temp=0.0)
t0 = time.perf_counter()
model, tok = load(repo)
load_s = time.perf_counter() - t0
msgs = [{"role": "user", "content": PROMPT}]
try:
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
except TypeError:
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
dec, pre, ids_hash, text = [], [], None, ""
for k in range(runs + 1):
    ids, last = [], None
    for r in stream_generate(model, tok, prompt, max_tokens=max_tokens, sampler=greedy):
        ids.append(int(r.token))
        last = r
    h = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]
    if k == 0:
        ids_hash, text = h, tok.decode(ids)
        continue
    if h != ids_hash:
        print(json.dumps({"error": "token ids changed between runs", "run": k}), flush=True)
        sys.exit(4)
    dec.append(round(last.generation_tps, 2))
    pre.append(round(last.prompt_tps, 1))
peak = getattr(mx, "get_peak_memory", None)
print(json.dumps({
    "model": repo.split("/")[-1], "max_tokens": max_tokens, "runs": runs,
    "prompt_tokens": last.prompt_tokens, "generated": last.generation_tokens,
    "decode_tok_s": dec, "decode_median": statistics.median(dec),
    "prompt_tok_s": pre, "prompt_median": statistics.median(pre),
    "ids_sha": ids_hash, "load_s": round(load_s, 1),
    "peak_gpu_gib": round(peak() / 2**30, 3) if peak else None,
    "max_rss_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3),
    "text": " ".join(text.split())[:160],
}), flush=True)
