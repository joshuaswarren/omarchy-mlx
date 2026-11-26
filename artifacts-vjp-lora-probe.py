#!/usr/bin/env python3
"""VjpKernels LoRA one-step probe: one forward + one backward through the
mlx-lm LoRA-wrapped model, tiled to seq 512, run twice in one process
(composed leg then fused leg) so weights/data/seed are identical between
the legs. Prints one JSON line per leg: loss, eval wall time, peak memory.
"""
import argparse
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seq", type=int, default=512)
    args = ap.parse_args()

    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, tok = load(args.model)
    # Pull names of all Linear-like modules under the last few transformer
    # blocks (covers self_attn + linear_attn projections). The 0.31.3
    # API expects a config dict; keys are matched against module names.
    config = {
        "rank": 8,
        "scale": 2.0,
        "dropout": 0.0,
        "keys": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkvz",
            "linear_attn.in_proj_ba",
            "linear_attn.out_proj",
        ],
    }
    linear_to_lora_layers(model, 8, config)

    enc = tok.encode(
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump. "
        "Sphinx of black quartz, judge my vow."
    )
    ids = np.asarray(enc, dtype=np.int32)
    reps = (args.seq + len(ids) - 1) // len(ids)
    ids = np.tile(ids, reps)[: args.seq][None]
    x = mx.array(ids)

    results = []
    for leg in ("composed", "fused"):
        if leg == "composed":
            os.environ["MLX_OMARCHY_NO_FUSED_VJP"] = "1"
        else:
            os.environ.pop("MLX_OMARCHY_NO_FUSED_VJP", None)
        mx.random.seed(args.seed)
        np.random.seed(args.seed)

        def loss_fn(m):
            out = m(x)
            logits = out.logits if hasattr(out, "logits") else out
            return mx.mean(logits.astype(mx.float32))

        lg = nn.value_and_grad(model, loss_fn)
        loss, grads = lg(model)
        flat = []
        for g in grads.values():
            for v in g.values():
                flat.append(v)
        t0 = time.perf_counter()
        mx.eval(loss, *flat)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        peak = ""
        try:
            peak = mx.metal.get_peak_memory()
        except Exception:
            peak = ""
        row = {
            "leg": leg,
            "loss": float(loss),
            "eval_ms": round(dt_ms, 2),
            "peak_mem": peak,
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    with open(args.out, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
