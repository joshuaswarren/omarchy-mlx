#!/usr/bin/env python3
"""Why do k/v/down ride the tile kernel in-model while their shapes pass
the vec guard in isolation? Load the REAL model weights, run each decode
matmul under the GPU profiler, then repeat with contiguous copies of the
weights (fresh, aligned allocations). If the contiguous variant flips to
MatmulVecBF16, the exclusion is weight-buffer-offset alignment."""
import json
import os
import re
import sys

import mlx.core as mx
import numpy as np

MODEL = os.path.expanduser("~/models/Qwen2.5-0.5B-Instruct-bf16-mlx")
PROF = os.environ.get("VEC_PROBE_OUT", "/tmp/vec_excl.jsonl")
os.environ["MLX_OMARCHY_GPU_PROFILE"] = PROF
os.environ["MLX_OMARCHY_GPU_PROFILE_LABEL"] = "vec-exclusion"

names, inside = [], False
for line in open(os.path.expanduser(
        "~/src/mlx-Bf16DecodeAttribution/overlay/mlx/backend/omarchy/compute.h")):
    if "enum class ComputeKernel" in line:
        inside = True
        continue
    if inside:
        if "}" in line:
            break
        m = re.match(r"^\s*([A-Za-z0-9_]+)\s*,?\s*$", line)
        if m and m.group(1) != "Count":
            names.append(m.group(1))

import glob
import mlx.nn as nn
weights = {}
for f in sorted(glob.glob(MODEL + "/*.safetensors")):
    weights.update(mx.load(f))
L = {k.split("layers.0.")[1]: v for k, v in weights.items() if "layers.0." in k}
rng = np.random.default_rng(0)
x896 = mx.array(rng.standard_normal((1, 896)).astype(np.float16)).astype(mx.bfloat16)
x4864 = mx.array(rng.standard_normal((1, 4864)).astype(np.float16)).astype(mx.bfloat16)

# biases exist for qkv (Qwen2.5 attention_bias): run both the raw
# matmul form and the Linear-with-bias form for k/v.
cases = [
    ("q_real", x896, L["self_attn.q_proj.weight"], None),
    ("k_real", x896, L["self_attn.k_proj.weight"],
     L["self_attn.k_proj.bias"]),
    ("v_real", x896, L["self_attn.v_proj.weight"],
     L["self_attn.v_proj.bias"]),
    ("o_real", x896, L["self_attn.o_proj.weight"], None),
    ("gate_real", x896, L["mlp.gate_proj.weight"], None),
    ("up_real", x896, L["mlp.up_proj.weight"], None),
    ("down_real", x4864, L["mlp.down_proj.weight"], None),
    ("lmhead_real", x896, weights.get("lm_head.weight",
        weights["model.embed_tokens.weight"]), None),
]
for name, x, w, b in cases:
    mx.eval(x @ w.transpose() if b is None else x @ w.transpose() + b)
for name, x, w, b in cases:
    wc = mx.contiguous(w)
    bc = mx.contiguous(b) if b is not None else None
    mx.eval(x @ wc.transpose() if bc is None else x @ wc.transpose() + bc)
mx.eval(mx.array([0]))

# attribute kernels: order of d events vs case order is unreliable across
# allocators; instead rely on counts: 14 cases, each exactly 1 matmul.
kernels = []
for line in open(PROF):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("k") == "d" and names[d["e"]] in (
            "MatmulVecBF16", "MatmulBF16", "MatmulBF16Coopmat",
            "MatmulF32", "MatmulF32Coopmat"):
        kernels.append(names[d["e"]])
print("case order: q,k(bias),v(bias),o,gate,up,down,lmhead x"
      " {real x7 then contiguous x7}")
for i, kk in enumerate(kernels):
    print(i, kk)
