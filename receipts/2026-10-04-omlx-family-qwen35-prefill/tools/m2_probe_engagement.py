#!/usr/bin/env python3
"""Engagement probe, one arm per process (the omarchy flash-256 gate is
a function-local static): time mx.fast.scaled_dot_product_attention on
a long context. The composed arm materializes B*H*qL*kL*4 bytes of f32
scores; the flash arm does not. Run twice:

    python3 m2_probe_engagement.py --mode composed
    python3 m2_probe_engagement.py --mode flash
"""
import argparse
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["composed", "flash"], required=True)
args = ap.parse_args()
if args.mode == "composed":
    os.environ["MLX_OMARCHY_SDPA_PREFILL_FLASH256"] = "0"

import mlx.core as mx  # noqa: E402  (env must be set before first dispatch)

H, KV, QL, KL, D = 8, 2, 1024, 4096, 256
rng = np.random.default_rng(7)
q = mx.array(rng.uniform(-1, 1, (1, H, QL, D)).astype(np.float32)).astype(mx.bfloat16)
k = mx.array(rng.uniform(-1, 1, (1, KV, KL, D)).astype(np.float32)).astype(mx.bfloat16)
v = mx.array(rng.uniform(-1, 1, (1, KV, KL, D)).astype(np.float32)).astype(mx.bfloat16)
scale = 1.0 / np.sqrt(D)

# warmup (compilation, allocator)
out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
mx.eval(out)

t0 = time.perf_counter()
for _ in range(3):
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
    mx.eval(out)
dt = (time.perf_counter() - t0) / 3
print(f"ENGAGEMENT {args.mode} {dt * 1000:.2f} ms/call")
