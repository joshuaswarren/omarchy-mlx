#!/usr/bin/env python3
"""wAZ: host-stamped mlx-lm decode for an ftrace window: prints DECODE <t_first_ns> <t_last_ns> <tokens> (CLOCK_MONOTONIC).
usage: decode_jobs.py MODEL_DIR [GEN=64]"""
import sys, time
import mlx.core as mx
from mlx_lm import load, stream_generate
model, tok = load(sys.argv[1]); GEN = int(sys.argv[2]) if len(sys.argv) > 2 else 64
prompt = "The history of France begins in antiquity. " * 60
for _ in stream_generate(model, tok, prompt, max_tokens=8): pass
time.sleep(0.2)
tf = tl = None; n = 0
for _ in stream_generate(model, tok, prompt, max_tokens=GEN):
    t = time.monotonic_ns()
    if tf is None: tf = t
    tl = t; n += 1
print(f"DECODE {tf} {tl} {n}", flush=True)
