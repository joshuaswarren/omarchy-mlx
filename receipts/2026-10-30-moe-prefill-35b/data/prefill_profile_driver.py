"""Prefill driver for an mlx_lm model: one pass of N prompt tokens through the model in chunks of --step tokens, one bracketed window per chunk.

Windows file shape is that of profile_batch_driver.py (a meta line with "prompt" and "start_monotonic", then one {"i","t0","t1"} line per chunk), so
b2prof_summary.py one/wall can read it. The prompt is real English text (the first N tokens of the repository's own docs or any text file), not a repeated
token, so the router sees varied expert choices. Prints the unprofiled-style rate too: tokens / sum of window seconds.

  prefill_profile_driver.py --model PATH --text FILE --tokens 1024 --step 512 --warmup-tokens 512 --out windows.jsonl
The warmup pass runs --warmup-tokens through a throwaway cache (pipelines, shader pipelines, first-touch weights) and is not timed.
"""
import argparse
import json
import sys
import time

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--text", required=True)
p.add_argument("--tokens", type=int, default=1024)
p.add_argument("--step", type=int, default=512)
p.add_argument("--warmup-tokens", type=int, default=512)
p.add_argument("--out", required=True)
a = p.parse_args()

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tok = load(a.model)
with open(a.text, encoding="utf-8") as f:
    ids = tok.encode(f.read())
need = a.tokens + a.warmup_tokens
if len(ids) < need:
    print(f"text too short: {len(ids)} tokens, need {need}", file=sys.stderr)
    sys.exit(2)


def run(tokens, step, windows=None, f=None):
    cache = make_prompt_cache(model)
    x = mx.array(tokens)[None]
    pos = 0
    i = 0
    while pos < x.shape[1]:
        chunk = x[:, pos:pos + step]
        t0 = time.monotonic_ns()
        out = model(chunk, cache=cache)
        mx.eval(out[:, -1, :])
        mx.synchronize()
        t1 = time.monotonic_ns()
        if f is not None:
            f.write(json.dumps({"i": i, "t0": t0, "t1": t1, "n": int(chunk.shape[1])}) + "\n")
            f.flush()
        pos += chunk.shape[1]
        i += 1


run(ids[:a.warmup_tokens], a.step)
mx.synchronize()
t_start = time.monotonic_ns()
with open(a.out, "w") as f:
    f.write(json.dumps({"prompt": f"first {a.tokens} tokens after {a.warmup_tokens} of {a.text}", "prompt_len": a.tokens,
                        "warmup_tokens": a.warmup_tokens, "start_monotonic": t_start, "step": a.step}) + "\n")
    run(ids[a.warmup_tokens:a.warmup_tokens + a.tokens], a.step, f=f)
wins = [json.loads(line) for line in open(a.out)][1:]
secs = sum((w["t1"] - w["t0"]) for w in wins) / 1e9
print(f"prefill {a.tokens} tokens in {secs:.3f} s over {len(wins)} chunks = {a.tokens / secs:.2f} tok/s", flush=True)
