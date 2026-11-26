#!/usr/bin/env python3
"""GDU raw-repeat numerics probe (DispatchFuse): fused raw decode kernel
(via MLX_OMARCHY_GDN_RAW_REPEAT=1) vs the exact composed fallback chain
(gate off), Qwen3.5-9B decode shapes (Hk=16, Hv=32, Dk=Dv=128).

Reports max/mean abs error and differing-bit fraction on out and state.
usage: gdu_raw_repeat_probe.py OUT.json
"""
import json
import os
import sys
import zlib

import mlx.core as mx
import numpy as np

sys.path.insert(0, "")
out_path = sys.argv[1]


def call(repeat_on, q, k, v, a, b, A_log, dt_bias, state, mask):
    if repeat_on:
        os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "1"
    else:
        os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
    import importlib
    gd = importlib.import_module("mlx_lm.models.gated_delta")
    out, st = gd.gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, mask,
                                    use_kernel=True)
    mx.eval(out, st)
    return out, st


rng = np.random.default_rng(zlib.crc32("gdu-probe".encode()))
B, Hk, Hv, Dk, Dv = 1, 16, 32, 128, 128
q = mx.array(rng.standard_normal((B, 1, Hk, Dk)) * 0.5).astype(mx.bfloat16)
k = mx.array(rng.standard_normal((B, 1, Hk, Dk)) * 0.5).astype(mx.bfloat16)
v = mx.array(rng.standard_normal((B, 1, Hv, Dv)) * 0.5).astype(mx.bfloat16)
a = mx.array(rng.standard_normal((B, 1, Hv)) * 0.5).astype(mx.bfloat16)
b = mx.array(rng.standard_normal((B, 1, Hv)) * 0.5).astype(mx.bfloat16)
A_log = mx.array(rng.standard_normal(Hv) * 0.1 + 1.0).astype(mx.bfloat16)
dt_bias = mx.array(rng.standard_normal(Hv) * 0.1).astype(mx.float32)
state = mx.array(rng.standard_normal((B, Hv, Dv, Dk)) * 0.1).astype(mx.float32)
mask = mx.ones((B, 1), mx.bool_)

out_ref, st_ref = call(False, q, k, v, a, b, A_log, dt_bias, state, mask)
out_test, st_test = call(True, q, k, v, a, b, A_log, dt_bias, state, mask)


def diff(x, y):
    xf = np.array(x.astype(mx.float32))
    yf = np.array(y.astype(mx.float32))
    d = np.abs(xf - yf)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "bit_diff_frac": float((np.array(x.view(mx.uint16)) != np.array(y.view(mx.uint16))).mean())
        if x.dtype == mx.bfloat16
        else float((np.array(x.view(mx.uint32)) != np.array(y.view(mx.uint32))).mean()),
    }


res = {
    "meta": {"mx": mx.__version__, "shapes": f"Hk={Hk} Hv={Hv} Dk={Dk} Dv={Dv}"},
    "out": diff(out_ref, out_test),
    "state": diff(st_ref, st_test),
}
with open(out_path, "w") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps(res, indent=1))
