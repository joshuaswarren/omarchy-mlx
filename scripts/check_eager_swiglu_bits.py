#!/usr/bin/env python3
"""Bit-for-bit check of the eager silu(gate) * up product against the compiled swiglu, on a device.

Needs an mlx-omarchy wheel on a Vulkan GPU and the patched mlx_lm (patch-mlx-lm-qwen3-next-eager-swiglu.py applied).

Two comparisons, all on random bf16 data at the Qwen3.5-9B decode shapes (hidden 4096, intermediate 12288), with standard normal
inputs and the same inputs scaled by 8:

1. Elementwise. swiglu(gate, up) is the compiled function in mlx_lm/models/activations.py, def swiglu(gate, x): nn.silu(gate) * x.
   gate * sigmoid(gate) * up is the eager expression. Equal bits are required for every element.
2. The whole MLP with real 4-bit group-64 projections. Reference: down_proj(swiglu(gate_proj(x), up_proj(x))). Candidate: the
   patched Qwen3NextMLP.__call__, which runs the eager product for rows <= MLX_OMARCHY_EAGER_SWIGLU_ROWS and lets the backend
   fold it into the gate/up dispatch. This checker opts into 8 rows so seq>1 cases exercise the eager product; equal bits are
   required for every output element, for batch {1,3} x seq {1,4}, over many random inputs.

`--only compiled` or `--only eager` evaluates one path once and exits, so a dispatch trace can count the kernels of that path.
Exit status 0 only when every comparison is bit-identical.
Usage: check_eager_swiglu_bits.py [--draws 64] [--only compiled|eager] [--family next|qwen3]
(--family qwen3 needs patch-mlx-lm-qwen3-eager-swiglu.py applied; the default family needs the Qwen3-Next patcher.)
"""
import argparse
import os
import sys

# Opt into the multi-token row gate before the patched module reads it at
# import: rows 2..8 then exercise the eager product in the MLP check.
os.environ.setdefault("MLX_OMARCHY_EAGER_SWIGLU_ROWS", "8")

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.qwen3 import MLP as Qwen3MLP
from mlx_lm.models.qwen3_next import Qwen3NextMLP

ap = argparse.ArgumentParser()
ap.add_argument("--draws", type=int, default=64)
ap.add_argument("--only", choices=("compiled", "eager"))
ap.add_argument("--family", choices=("next", "qwen3"), default="next", help="next: Qwen3NextMLP (4096 x 12288); qwen3: dense Qwen3-4B MLP (2560 x 9728)")
a = ap.parse_args()

DIM, HIDDEN = (4096, 12288) if a.family == "next" else (2560, 9728)


def same_bits(x, y):
    return bool(mx.array_equal(x.view(mx.uint16), y.view(mx.uint16)).item())


mx.random.seed(20261014)
mlp = (Qwen3NextMLP if a.family == "next" else Qwen3MLP)(DIM, HIDDEN)
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
    for batch, seq in ((1, 1), (3, 1), (1, 4), (3, 4)):
        mism = 0
        for _ in range(a.draws):
            gate = (mx.random.normal((batch, seq, HIDDEN)) * scale).astype(mx.bfloat16)
            up = (mx.random.normal((batch, seq, HIDDEN)) * scale).astype(mx.bfloat16)
            ref = swiglu(gate, up)
            got = gate * mx.sigmoid(gate) * up
            mx.eval(ref, got)
            total += 1
            if not same_bits(ref, got):
                mism += 1
        print("ELEMENTWISE scale=%g batch=%d seq=%d draws=%d elements=%d mismatching_draws=%d" % (scale, batch, seq, a.draws, batch * seq * HIDDEN, mism), flush=True)
        bad += mism
for scale in (1.0, 8.0):
    for batch, seq in ((1, 1), (3, 1), (1, 4), (3, 4)):
        mism = 0
        for _ in range(a.draws):
            x = (mx.random.normal((batch, seq, DIM)) * scale).astype(mx.bfloat16)
            ref = compiled_path(x)
            got = mlp(x)
            mx.eval(ref, got)
            total += 1
            if not same_bits(ref, got):
                mism += 1
        print("MLP scale=%g batch=%d seq=%d draws=%d outputs=%d mismatching_draws=%d" % (scale, batch, seq, a.draws, batch * seq * DIM, mism), flush=True)
        bad += mism
print("RESULT %s comparisons=%d mismatching=%d" % ("PASS" if bad == 0 else "FAIL", total, bad), flush=True)
sys.exit(0 if bad == 0 else 1)
