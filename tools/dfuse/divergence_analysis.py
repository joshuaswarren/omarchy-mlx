#!/usr/bin/env python3
"""Analyze the first diverging captured operand set: where composed vs fused
(perrow_pf) differ, and what the input values look like at those sites.
usage: divergence_analysis.py DIVNPZ OUT.json
"""
import json
import os
import sys

import mlx.core as mx
import numpy as np

div_npz, out_path = sys.argv[1], sys.argv[2]
cap = np.load(div_npz)

def to_mlx_f32(name):
    return mx.array(cap[name])

q = to_mlx_f32("q").astype(mx.bfloat16)
k = to_mlx_f32("k").astype(mx.bfloat16)
v = to_mlx_f32("v").astype(mx.bfloat16)
a = to_mlx_f32("a").astype(mx.bfloat16)
b = to_mlx_f32("b").astype(mx.bfloat16)
A_log = mx.array(cap["A_log"])  # f32
if A_log.ndim == 1:
    A_log = A_log.reshape(1, 1, -1)
dt = mx.array(cap["dt"]).astype(mx.bfloat16)
if dt.ndim == 1:
    dt = dt.reshape(1, 1, -1)
state = mx.array(cap["state"])  # f32
mask = mx.ones((q.shape[0], q.shape[1]), mx.bool_) if "mask" not in cap else mx.array(cap["mask"])

Hk, Hv = q.shape[-2], v.shape[-2]
set_route_none = None
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
import importlib
gd = importlib.import_module("mlx_lm.models.gated_delta")
out_c, st_c = gd.gated_delta_update(q, k, v, a, b, A_log, dt, state, mask, use_kernel=True)
mx.eval(out_c, st_c)
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "1"
qr = mx.repeat(q, Hv // Hk, -2)
kr = mx.repeat(k, Hv // Hk, -2)
out_f, st_f = mx.fast.gated_delta_update_raw(qr, kr, v, a, b, A_log, dt, state, mask)
mx.eval(out_f, st_f)
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"

oc = np.array(out_c.astype(mx.float32))
of = np.array(out_f.astype(mx.float32))
sc = np.array(st_c.astype(mx.float32))
sf = np.array(st_f.astype(mx.float32))
d_out = np.abs(oc - of)
d_st = np.abs(sc - sf)
res = {
    "out_max_abs": float(d_out.max()),
    "state_max_abs": float(d_st.max()),
    "state_diff_positions": [[int(x) for x in p] for p in np.argwhere(d_st > 0)][:10],
    "state_diff_count": int((d_st > 0).sum()),
    "out_diff_count": int((d_out > 0).sum()),
    "state_sample_at_first_diff": None,
}
pos = res["state_diff_positions"]
if pos:
    h, vv, i = pos[0][0], pos[0][1], pos[0][2]
    res["state_sample_at_first_diff"] = {
        "head": h, "dv": vv, "dk": i,
        "state_composed": float(sc[0, h, vv, i]), "state_fused": float(sf[0, h, vv, i]),
        "g_composed_f32": float(np.exp(-np.exp(np.float64(cap["A_log"].reshape(-1)[h])) * 1.0)) if False else None,
        "A_log_head_f32": float(cap["A_log"].reshape(-1)[h]),
        "k_head_i": float(np.array(k.astype(mx.float32)).reshape(Hv, -1)[h, i]),
        "state_in": float(cap["state"].reshape(Hv, -1)[h, vv * 128 + i] if cap["state"].size >= Hv * 128 * 128 else 0.0),
    }
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res, indent=1))
