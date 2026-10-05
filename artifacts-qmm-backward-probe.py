#!/usr/bin/env python3
"""QmmBackward probe: backward through a LoRA-wrapped nn.QuantizedLinear
on the omarchy Vulkan backend.

This is the kernel-level twin of the 2B LoRA smoke: one QuantizedLinear
(group-64 4-bit, the fleet checkpoint layout) wrapped by
mlx_lm.tuner LoRALinear.from_base, one forward, one mx.grad backward.
The non-transposed quantized matmul (dy @ W) is the dx leg of that
backward. Prints one JSON line: dx vs a dequantized dense reference
(mx.dequantize + dense matmul), max abs difference, bound, finiteness.

Run on the omarchy backend (MLX_OMARCHY_ALLOW_NON_APPLE=1 on the
development box); pair every number with scripts/mlx_provenance.py
output from the same venv.
"""
import argparse
import json
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=32)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dtype = mx.bfloat16
    x_np = rng.normal(0.0, 0.5, size=(1, args.m, args.k)).astype(np.float32)

    from mlx_lm.tuner.lora import LoRALinear

    w = rng.normal(0.0, 0.02, size=(args.n, args.k)).astype(np.float32)
    w_q, scales, biases = mx.quantize(
        mx.array(w), args.group_size, args.bits)
    base = nn.QuantizedLinear(
        args.k, args.n, bias=False,
        group_size=args.group_size, bits=args.bits)
    base.weight = w_q
    base.scales = scales.astype(dtype)
    base.biases = biases.astype(dtype)
    base.freeze()
    lora = LoRALinear.from_base(base, r=args.rank, scale=2.0)
    lora.freeze()
    lora.unfreeze(keys=["lora_a", "lora_b"])

    def loss_fn(x):
        y = lora(x.astype(dtype))
        return y.astype(mx.float32).square().mean()

    x_mx = mx.array(x_np)
    grads = mx.grad(loss_fn)(x_mx)
    mx.eval(grads)

    w_d = mx.dequantize(
        w_q, base.scales, base.biases,
        group_size=args.group_size, bits=args.bits)
    lora_a = lora.lora_a
    lora_b = lora.lora_b

    def dense_loss_fn(x):
        xd = x.astype(dtype)
        y = xd @ w_d.T + (2.0 * (xd @ lora_a @ lora_b)).astype(dtype)
        return y.astype(mx.float32).square().mean()

    dense_grads = mx.grad(dense_loss_fn)(x_mx)
    mx.eval(dense_grads)

    g = np.asarray(grads, dtype=np.float32)
    r = np.asarray(dense_grads, dtype=np.float32)
    finite = bool(np.isfinite(g).all())
    max_abs = float(np.abs(g).max()) if finite else float("nan")
    max_diff = float(np.abs(g - r).max()) if finite else float("nan")
    ref_max = float(np.abs(r).max())
    bound = max(
        (args.n + 2.0) * max(2.0 * ref_max, 1.0) * 2.0**-23
        + max(2.0 * ref_max, 1.0) * 2.0**-8,
        1e-6)
    print(json.dumps({
        "probe": "qmm-backward-lora-grad",
        "shape": [1, args.m, args.k, args.n],
        "group_size": args.group_size,
        "bits": args.bits,
        "rank": args.rank,
        "grad_finite": finite,
        "grad_max_abs": max_abs,
        "dense_ref_max_abs": ref_max,
        "max_diff_vs_dense": max_diff,
        "bound": bound,
        "pass": bool(finite and max_diff <= bound),
    }))
    return 0 if finite and max_diff <= bound else 1


if __name__ == "__main__":
    sys.exit(main())
