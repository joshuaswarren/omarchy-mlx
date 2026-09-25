#!/usr/bin/env python3
"""One-token decode profile driver for the installed qwen3.8-2B path.

Runs the exact serving-shaped greedy decode loop (mlx_lm 0.31.3
generate_step, one host sync per token via y.item()) under the diagnostics
wheel, printing one JSON line per measured token:

    {"i":<n>,"t0":<monotonic_ns>,"t1":<monotonic_ns>,"tok":<id>}

t0 is taken just before the generator's next() (which blocks in y.item())
and t1 after it returns, so [t0, t1] brackets one full token: the tail of
graph_n submission + the host sync on graph_n. The scheduler-side work for
graph_n (graph build + submit) happened one iteration earlier and is
bracketed by the previous window; steady-state per-token cost is
t1[i] - t1[i-1]. The analyzer uses BOTH segmentations and reports the
steady-state one as primary.

Env expected (set by the caller):
  MLX_OMARCHY_GPU_PROFILE=<ndjson path>
  MLX_OMARCHY_GPU_PROFILE_LABEL=<name>
"""
import argparse
import json
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--prompt", default="The history of France begins in antiquity.")
parser.add_argument("--warmup-tokens", type=int, default=16)
parser.add_argument("--tokens", type=int, default=32)
parser.add_argument("--out", required=True, help="driver windows jsonl")
a = parser.parse_args()

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import generate_step

model, tok = load(a.model)
prompt = a.prompt

# Warmup: full compile/warm path on a throwaway cache.
cache = make_prompt_cache(model)
ids = tok.encode(prompt)
it = generate_step(mx.array(ids), model, prompt_cache=cache)
for n in range(a.warmup_tokens):
    next(it)
# Drain: leave the pipeline quiesced (generator holds no pending eval).
mx.synchronize()

# Measured window: fresh cache, 32 greedy tokens.
cache = make_prompt_cache(model)
it = generate_step(mx.array(ids), model, prompt_cache=cache)
toks = []
with open(a.out, "w") as f:
    meta = {
        "prompt": prompt,
        "prompt_len": len(ids),
        "warmup_tokens": a.warmup_tokens,
        "start_monotonic": time.monotonic_ns(),
    }
    f.write(json.dumps(meta) + "\n")
    for n in range(a.tokens):
        t0 = time.monotonic_ns()
        tokid, _ = next(it)
        t1 = time.monotonic_ns()
        toks.append(tokid)
        f.write(json.dumps({"i": n, "t0": t0, "t1": t1, "tok": int(tokid)}) + "\n")
        f.flush()
mx.synchronize()
print("tokens:", toks, file=sys.stderr)
