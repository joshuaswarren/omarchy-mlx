#!/usr/bin/env python3
"""DecodeGap4 conv+silu bit check: SHA-256 of raw output bits.
Rows per case name:
  convsilu.refnn.<case>   composed: gdn_conv_update(state,x,w) then mlx.nn.silu(out)
                          (the exact serving-stream silu kernel, old stream)
  convsilu.ref.<case>     composed: gdn_conv_update(state,x,w) then out*sigmoid(out)
  convsilu.fused.<case>   gdn_conv_update(state,x,w,activate=True); "ABSENT" on a
                          wheel without the kwarg
  convsilu.state.<case>   the carried conv state bits (raw uint16 path)
Run in the serving venv (fused rows = ABSENT) and the candidate venv (both).
Gates: ref/refnn identical across venvs; fused == ref in the candidate.
"""
import hashlib
import json
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

out_path = sys.argv[1]
rows = {}


def h(*arrs):
    m = hashlib.sha256()
    for a in arrs:
        mx.eval(a)
        if a.dtype in (mx.float32, mx.int32, mx.uint32):
            v = np.array(a.view(mx.uint32))
        elif a.dtype in (mx.bfloat16, mx.float16, mx.int16, mx.uint16):
            v = np.array(a.view(mx.uint16))
        else:
            v = np.array(a)
        m.update(v.tobytes())
    return m.hexdigest()[:16]


def bf16(x):
    return mx.array(np.asarray(x, np.float32)).astype(mx.bfloat16)


rng = np.random.default_rng(20260929)
B, C, K = 1, 2048, 4
state = bf16(rng.standard_normal((B, K - 1, C)) * 0.7)
x = bf16(rng.standard_normal((B, 1, C)) * 0.7)
w = bf16(rng.standard_normal((C, K, 1)) * 0.4)
gdn = mx.fast.gdn_conv_update

cases = {"rand": (state, x, w)}

x_np = np.array(x.astype(mx.float32))
x_np[0, 0, 0] = np.inf
x_np[0, 0, 1] = -np.inf
x_np[0, 0, 2] = np.nan
x_np[0, 0, 3] = 0.0
x_np[0, 0, 4] = -0.0
x_np[0, 0, 5] = 88.0
x_np[0, 0, 6] = -88.0
cases["special"] = (state, bf16(x_np), w)

st_np = np.array(state.astype(mx.float32))
st_np[0, 0, 5] = 1e30
st_np[0, 0, 6] = -1e30
cases["bigstate"] = (bf16(st_np), x, w)

w_np = np.array(w.astype(mx.float32))
w_np[7, 0, 0] = 0.0
w_np[7, 1, 0] = np.nan
cases["specialw"] = (state, x, bf16(w_np))

for name, (st, xv, wv) in cases.items():
    out, s = gdn(st, xv, wv)
    rows[f"convsilu.refnn.{name}"] = h(nn.silu(out), s)
    rows[f"convsilu.ref.{name}"] = h(out * mx.sigmoid(out), s)
    rows[f"convsilu.state.{name}"] = h(s)
    try:
        fout, fs = gdn(st, xv, wv, True)
        rows[f"convsilu.fused.{name}"] = h(fout, fs)
    except TypeError:
        rows[f"convsilu.fused.{name}"] = "ABSENT"

json.dump(rows, open(out_path, "w"), indent=1, sort_keys=True)
print("wrote", out_path, "cases:", len(cases))
