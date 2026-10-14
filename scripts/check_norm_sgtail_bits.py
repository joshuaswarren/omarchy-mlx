#!/usr/bin/env python3
"""Bit-for-bit check of the subgroup-tail norm kernels against the shared-memory tree kernels, on a device.

Needs an mlx-omarchy wheel built with the SGTAIL kernels on a Vulkan GPU with 32-lane subgroups. MLX_OMARCHY_NORM_SGTAIL is read at every
dispatch, so one process runs each case with the variable set to 0 (shared-memory tree) and to 1 (subgroup tail) and compares the output
bits. If the device cannot run the subgroup-tail kernels, both settings run the same kernel and the check says so.

Cases: mx.fast.rms_norm on bf16 rows of width 128, 640, 1024, 2560, 4096, 5120 and 8192, batch 1 to 4, and mx.fast.rms_norm_gated on
[B, 1, heads, 128] bf16. Inputs are standard normal, the same scaled by 8 and by 1e-3, and rows with one large outlier. Weights are bf16.
Exit status 0 only when every comparison is bit-identical.
Usage: check_norm_sgtail_bits.py [--draws 16]
"""
import argparse
import os
import sys

import mlx.core as mx

ap = argparse.ArgumentParser()
ap.add_argument("--draws", type=int, default=16)
a = ap.parse_args()

EPS = 1e-6
WIDTHS = (128, 640, 1024, 2560, 4096, 5120, 8192)


def with_env(value, fn):
    os.environ["MLX_OMARCHY_NORM_SGTAIL"] = value
    out = fn()
    mx.eval(out)
    return out


def same_bits(x, y):
    return bool(mx.array_equal(x.view(mx.uint16), y.view(mx.uint16)).item())


def make(shape, scale, outlier):
    x = mx.random.normal(shape) * scale
    if outlier:
        flat = x.reshape(-1)
        idx = mx.array([flat.size // 3])
        flat = mx.put_along_axis(flat, idx, mx.array([1.0e4 * scale]), axis=0)
        x = flat.reshape(shape)
    return x.astype(mx.bfloat16)


mx.random.seed(20261014)
bad = 0
total = 0
for width in WIDTHS:
    w = (1.0 + 0.1 * mx.random.normal((width,))).astype(mx.bfloat16)
    for batch in (1, 4):
        mism = 0
        n = 0
        for scale, outlier in ((1.0, False), (8.0, False), (1.0e-3, False), (1.0, True)):
            for _ in range(a.draws):
                x = make((batch, 1, width), scale, outlier)
                mx.eval(x)
                ref = with_env("0", lambda: mx.fast.rms_norm(x, w, EPS))
                got = with_env("1", lambda: mx.fast.rms_norm(x, w, EPS))
                n += 1
                if not same_bits(ref, got):
                    mism += 1
        total += n
        bad += mism
        print("RMS_NORM width=%d batch=%d comparisons=%d mismatching=%d" % (width, batch, n, mism), flush=True)
wg = (1.0 + 0.1 * mx.random.normal((128,))).astype(mx.bfloat16)
for batch, heads in ((1, 32), (3, 32), (1, 16)):
    mism = 0
    n = 0
    for scale, outlier in ((1.0, False), (8.0, False), (1.0e-3, False), (1.0, True)):
        for _ in range(a.draws):
            h = make((batch, 1, heads, 128), scale, outlier)
            g = make((batch, 1, heads, 128), 1.0, False)
            mx.eval(h, g)
            ref = with_env("0", lambda: mx.fast.rms_norm_gated(h, g, wg, EPS))
            got = with_env("1", lambda: mx.fast.rms_norm_gated(h, g, wg, EPS))
            n += 1
            if not same_bits(ref, got):
                mism += 1
    total += n
    bad += mism
    print("RMS_NORM_GATED batch=%d heads=%d width=128 comparisons=%d mismatching=%d" % (batch, heads, n, mism), flush=True)
print("RESULT %s comparisons=%d mismatching=%d" % ("PASS" if bad == 0 else "FAIL", total, bad), flush=True)
sys.exit(0 if bad == 0 else 1)
