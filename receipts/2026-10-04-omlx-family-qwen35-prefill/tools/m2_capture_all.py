#!/usr/bin/env python3
"""Capture EVERY sdpa call in one model forward; then (replay mode) run
one call index through the current arm. Localizes the first divergence
layer."""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--capture", nargs=2, metavar=("MODEL", "OUTPREFIX"))
ap.add_argument("--replay", nargs=3, metavar=("PREFIX", "INDEX", "OUT"))
args = ap.parse_args()

if args.capture:
    import mlx.core as mx
    from mlx_lm import load

    model, tokenizer = load(args.capture[0])
    calls = []
    orig = mx.fast.scaled_dot_product_attention

    def spy(q, k, v, **kw):
        if q.ndim == 4 and q.shape[2] > 1:
            calls.append((
                np.array(q.astype(mx.float32)),
                np.array(k.astype(mx.float32)),
                np.array(v.astype(mx.float32)),
                {kk: str(vv) for kk, vv in kw.items()},
            ))
        return orig(q, k, v, **kw)

    mx.fast.scaled_dot_product_attention = spy
    ids = tokenizer.encode("The capital of France is")
    logits = model(mx.array([ids]))
    mx.eval(logits)
    for i, (q, k, v, kw) in enumerate(calls):
        np.savez_compressed(f"{args.capture[1]}.call{i}.npz", q=q, k=k, v=v)
        if i == 0:
            print("kw of call0:", kw, flush=True)
    print(f"captured {len(calls)} sdpa calls", flush=True)
else:
    import mlx.core as mx

    prefix, index, out = args.replay
    d = np.load(f"{prefix}.call{index}.npz")
    scale = 1.0 / np.sqrt(d["q"].shape[-1])
    qb = mx.array(d["q"]).astype(mx.bfloat16)
    kb = mx.array(d["k"]).astype(mx.bfloat16)
    vb = mx.array(d["v"]).astype(mx.bfloat16)
    out_arr = mx.fast.scaled_dot_product_attention(
        qb, kb, vb, scale=float(scale), mask="causal")
    mx.eval(out_arr)
    np.save(out, np.array(out_arr.astype(mx.float32)))
    print(f"call{index} saved", flush=True)
