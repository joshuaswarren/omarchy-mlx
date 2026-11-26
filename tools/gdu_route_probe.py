#!/usr/bin/env python3
"""Route probe v2 (Qwen3.5-9B production dtypes): A_log f32, dt_bias bf16,
q/k optionally repeated to Hv. Checks the raw route dispatches fused and is
bit-identical to the composed chain.
usage: gdu_route_probe.py OUTDIR
"""
import json
import os
import sys

import mlx.core as mx
import numpy as np

out_dir = sys.argv[1]
rng = np.random.default_rng(7)


def build(Hk, a_log_f32=True):
    q = mx.array(rng.standard_normal((1, 1, Hk, 128)) * 0.5).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((1, 1, Hk, 128)) * 0.5).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((1, 1, 32, 128)) * 0.5).astype(mx.bfloat16)
    a = mx.array(rng.standard_normal((1, 1, 32)) * 0.5).astype(mx.bfloat16)
    b = mx.array(rng.standard_normal((1, 1, 32)) * 0.5).astype(mx.bfloat16)
    A_log = mx.array(rng.standard_normal(32) * 0.1 + 1.0)
    A_log = A_log.astype(mx.float32 if a_log_f32 else mx.bfloat16)
    dt = mx.array(rng.standard_normal(32) * 0.1).astype(mx.bfloat16)
    st = mx.array(rng.standard_normal((1, 32, 128, 128)) * 0.1).astype(mx.float32)
    mask = mx.ones((1, 1), mx.bool_)
    return q, k, v, a, b, A_log, dt, st, mask


def bits(x):
    if x.dtype == mx.bfloat16:
        return np.array(x.view(mx.uint16)).tobytes()
    return np.array(x.view(mx.uint32)).tobytes()


results = {}
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"

# Case 1: direct raw call at Hk=32 (repeated), 9B dtypes -> fused expected
q, k, v, a, b, A_log, dt, st, mask = build(32)
out_f, st_f = mx.fast.gated_delta_update_raw(q, k, v, a, b, A_log, dt, st, mask)
mx.eval(out_f, st_f)
results["direct_hk32_alogf32"] = {"out_shape": list(out_f.shape)}

# Case 2: dispatcher composed (gate off, Hk=16) -> the reference chain
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
from mlx_lm.models.gated_delta import gated_delta_update  # noqa: E402
q2, k2, v2, a2, b2, A_log2, dt2, st2, mask2 = build(16)
out_c, st_c = gated_delta_update(q2, k2, v2, a2, b2, A_log2, dt2, st2, mask2,
                                 use_kernel=True)
mx.eval(out_c, st_c)

# Case 3: dispatcher with the gate on (repeat + fused) -> compare bits
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "1"
q3, k3, v3, a3, b3, A_log3, dt3, st3, mask3 = build(16)
out_r, st_r = gated_delta_update(q3, k3, v3, a3, b3, A_log3, dt3, st3, mask3,
                                 use_kernel=True)
mx.eval(out_r, st_r)
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"


def diff(x, y):
    xf, yf = np.array(x.astype(mx.float32)), np.array(y.astype(mx.float32))
    d = np.abs(xf - yf)
    return {"max_abs": float(d.max()), "bit_identical": bits(x) == bits(y)}


results["fused_vs_composed"] = {"out": diff(out_c, out_r), "state": diff(st_c, st_r)}
with open(f"{out_dir}/gdu-route-probe2.json", "w") as fh:
    json.dump(results, fh, indent=1)
print(json.dumps(results, indent=1))
