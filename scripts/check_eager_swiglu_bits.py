#!/usr/bin/env python3
"""Bit-for-bit check of the eager silu(gate) * up product against the compiled swiglu, on a device.

Needs an mlx-omarchy wheel on a Vulkan GPU and the patched mlx_lm (patch-mlx-lm-qwen3-next-eager-swiglu.py applied).

Two comparisons, all on random bf16 data at the Qwen3.5-9B decode shapes (hidden 4096, intermediate 12288), with standard normal
inputs and the same inputs scaled by 8:

1. Elementwise. swiglu(gate, up) is the compiled function in mlx_lm/models/activations.py, def swiglu(gate, x): nn.silu(gate) * x.
   gate * sigmoid(gate) * up is the eager expression. Equal bits are required for every element.
2. The whole MLP with real 4-bit group-64 projections. Reference: down_proj(swiglu(gate_proj(x), up_proj(x))). Candidate: the
   patched Qwen3NextMLP.__call__, which for single-token rows runs the eager product and lets the backend fold it into the gate/up
   GEMV dispatch. Equal bits are required for every output element, for batch 1 and batch 3, over many random inputs.

`--only compiled` or `--only eager` evaluates one path once and exits, so a dispatch trace can count the kernels of that path.
Exit status 0 only when every comparison is bit-identical.
Usage: check_eager_swiglu_bits.py [--draws 64] [--only compiled|eager]
"""
import argparse
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3_next import Qwen3NextMLP

ap = argparse.ArgumentParser()
ap.add_argument("--draws", type=int, default=64)
ap.add_argument("--only", choices=("compiled", "eager"))
a = ap.parse_args()

DIM, HIDDEN = 4096, 12288


def same_bits(x, y):
    return bool(mx.array_equal(x.view(mx.uint16), y.view(mx.uint16)).item())


mx.random.seed(20261014)
mlp = Qwen3NextMLP(DIM, HIDDEN)
mlp.set_dtype(mx.bfloat16)
nn.quantize(mlp, group_size=64, bits=4)
mx.eval(mlp.parameters())


def compiled_path(x):
    return mlp.down_proj(swiglu(mlp.gate_proj(x), mlp.up_proj(x)))


if a.only:
    x = mx.random.normal((1, 1, DIM)).astype(mx.bfloat16)
    out = compiled_path(x) if a.only == "compiled" else mlp(x)
    mx.eval(out)
    print("ONLY %s done" % a.only, flush=True)
    sys.exit(0)

bad = 0
total = 0
# Standard normal inputs, and the same inputs scaled by 8 so sigmoid saturates and the bf16 rounding of the intermediate differs most.
for scale in (1.0, 8.0):
    for batch in (1, 3):
        mism = 0
        for _ in range(a.draws):
            gate = (mx.random.normal((batch, 1, HIDDEN)) * scale).astype(mx.bfloat16)
            up = (mx.random.normal((batch, 1, HIDDEN)) * scale).astype(mx.bfloat16)
            ref = swiglu(gate, up)
            got = gate * mx.sigmoid(gate) * up
            mx.eval(ref, got)
            total += 1
            if not same_bits(ref, got):
                mism += 1
        print("ELEMENTWISE scale=%g batch=%d draws=%d elements=%d mismatching_draws=%d" % (scale, batch, a.draws, batch * HIDDEN, mism), flush=True)
        bad += mism
for scale in (1.0, 8.0):
    for batch in (1, 3):
        mism = 0
        for _ in range(a.draws):
            x = (mx.random.normal((batch, 1, DIM)) * scale).astype(mx.bfloat16)
            ref = compiled_path(x)
            got = mlp(x)
            mx.eval(ref, got)
            total += 1
            if not same_bits(ref, got):
                mism += 1
        print("MLP scale=%g batch=%d draws=%d outputs=%d mismatching_draws=%d" % (scale, batch, a.draws, batch * DIM, mism), flush=True)
        bad += mism
print("RESULT %s comparisons=%d mismatching=%d" % ("PASS" if bad == 0 else "FAIL", total, bad), flush=True)
sys.exit(0 if bad == 0 else 1)
