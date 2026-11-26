#!/usr/bin/env python3
"""Bit-identical column-group widening screen for MatmulVecBF16.

Times eager evals of the decode-shape matmuls (lm_head dominant) on the
base wheel vs the candidate wheel, wall clock, interleaved rounds.
Runs under the GPU lock on a quiet machine. Outputs NDJSON rounds.
"""
import argparse
import json
import time

import mlx.core as mx
import numpy as np

SHAPES = [
    ("lmhead", 1, 896, 151936),
    ("gate", 1, 896, 4864),
    ("down", 1, 4864, 896),
    ("qproj", 1, 896, 896),
    ("kproj", 1, 896, 128),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument("--digest-out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    with open(args.out, "w") as nd, open(args.digest_out, "w") as dj:
        for name, m, k, n in SHAPES:
            x = mx.array(rng.standard_normal((m, k)).astype(np.float16)
                         ).astype(mx.bfloat16)
            w = mx.array(rng.standard_normal((n, k)).astype(np.float16)
                         ).astype(mx.bfloat16)
            wt = w.transpose()
            # warmup
            for _ in range(3):
                mx.eval(x @ wt)
            for r in range(args.rounds):
                t0 = time.perf_counter()
                y = x @ wt
                mx.eval(y)
                dt = time.perf_counter() - t0
                rec = {"label": args.label, "shape": name,
                       "round": r, "wall_s": round(dt * 1e3, 4)}
                nd.write(json.dumps(rec, sort_keys=True) + "\n")
            # digest: greedy ints of the output bits (dtype-stable)
            y = x @ wt
            mx.eval(y)
            dj.write(json.dumps({
                "label": args.label, "shape": name,
                "digest": y.astype(mx.uint16).sum().item(),
                "first16": [float(v) for v in np.asarray(
                    y.astype(mx.float32)).reshape(-1)[:16]],
            }, sort_keys=True) + "\n")
    print("SCREEN_DONE", args.label)


if __name__ == "__main__":
    main()
