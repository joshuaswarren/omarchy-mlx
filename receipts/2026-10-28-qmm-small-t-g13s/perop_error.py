#!/usr/bin/env python3
"""docs/numerics-gate.md check 1: per-op error of the short-pass route against an fp64 reference.

For each cell, the same quantized weight and bf16 x run through mx.quantized_matmul twice in this
process: MLX_OMARCHY_QMV_SMALL_T unset (the deployed coopmat route) and set (the short-pass GEMV).
Both are compared with an fp64 numpy contraction of the dequantized weight and the bf16 x.
The route env var is read live by the backend, so one process holds both arms.
Shapes: Qwen3.5/3.8 projections (K, N) and the existing test cells; T = 2..8; 24 datasets each.
"""
import json
import os
import sys

import numpy as np
import mlx.core as mx

CELLS = [(2560, 9216), (9216, 2560), (2048, 6144), (6144, 2048), (2048, 520), (448, 130)]
DATASETS = 24


def run(x, wq, s, b, small_t):
    if small_t:
        os.environ["MLX_OMARCHY_QMV_SMALL_T"] = small_t
    else:
        os.environ.pop("MLX_OMARCHY_QMV_SMALL_T", None)
    y = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
    mx.eval(y)
    os.environ.pop("MLX_OMARCHY_QMV_SMALL_T", None)
    return np.array(y.astype(mx.float32), dtype=np.float64)


rows = []
worse = 0
for k, n in CELLS:
    for t in range(2, 9):
        err = {"deployed": [], "short_pass": []}
        for seed in range(DATASETS):
            mx.random.seed(7919 * k + 31 * n + 101 * t + seed)
            w = mx.random.normal((n, k)) * 0.05
            wq, s, b = mx.quantize(w, group_size=64, bits=4)
            s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
            x = mx.random.normal((1, t, k)).astype(mx.bfloat16)
            mx.eval(wq, s, b, x)
            wd = np.array(mx.dequantize(wq, s.astype(mx.float32), b.astype(mx.float32),
                                        group_size=64, bits=4), dtype=np.float64)
            ref = np.array(x.astype(mx.float32), dtype=np.float64)[0] @ wd.T
            for arm, env in (("deployed", ""), ("short_pass", "8")):
                got = run(x, wq, s, b, env)[0]
                assert np.isfinite(got).all(), (k, n, t, seed, arm)
                err[arm].append(float(np.linalg.norm(got - ref) / np.linalg.norm(ref)))
        row = {"k": k, "n": n, "t": t,
               "deployed_rel_l2_mean": float(np.mean(err["deployed"])),
               "short_pass_rel_l2_mean": float(np.mean(err["short_pass"])),
               "deployed_rel_l2_max": float(np.max(err["deployed"])),
               "short_pass_rel_l2_max": float(np.max(err["short_pass"]))}
        row["no_worse"] = row["short_pass_rel_l2_mean"] <= row["deployed_rel_l2_mean"]
        worse += not row["no_worse"]
        rows.append(row)
        print(f"K={k:5d} N={n:5d} T={t}  deployed {row['deployed_rel_l2_mean']:.3e}  "
              f"short-pass {row['short_pass_rel_l2_mean']:.3e}  {'ok' if row['no_worse'] else 'WORSE'}",
              flush=True)
print(f"cells where the short-pass mean error is worse than deployed: {worse} of {len(rows)}")
json.dump({"mlx": mx.__version__, "datasets": DATASETS, "rows": rows, "worse_cells": worse},
          open(sys.argv[1] if len(sys.argv) > 1 else "perop-error.json", "w"), indent=1)
