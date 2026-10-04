#!/usr/bin/env python3
"""Per-op accuracy: captured GDU operands -> composed f32 path vs fused
kernel vs a float64 numpy reference of the same recursion. The fused kernel
must be no worse than the composed path against fp64.
Phase 1 (capture): run with --capture; monkeypatches the dispatcher to dump
the first decode call's operands.
Phase 2 (compare): --compare CAPTURE_NPZ
"""
import glob
import json
import os
import sys

import mlx.core as mx
import numpy as np

model_dir = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/"))[0]

if sys.argv[1] == "--capture":
    import mlx_lm.models.gated_delta as gd
    orig = gd.gated_delta_update
    captured = {}

    def spy(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
        if not captured and q.shape[1] == 1:
            captured["q"] = q
            captured["k"] = k
            captured["v"] = v
            captured["a"] = a
            captured["b"] = b
            captured["A_log"] = A_log
            captured["dt_bias"] = dt_bias
            captured["state"] = state
            captured["mask"] = mask
            print("captured operands", q.shape, k.shape, v.shape, flush=True)
        return orig(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)

    gd.gated_delta_update = spy
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    model, tokenizer = load(model_dir)
    ids = tokenizer.encode("Explain why seasons change on Earth to a student.")
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids], dtype=mx.int32), cache=cache)
    for _ in range(3):
        tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    np.savez(sys.argv[2], **{k: np.array(v.astype(mx.float32) if v is not None and v.dtype == mx.bfloat16 else np.array(v))
                             for k, v in captured.items() if v is not None},
             **{k + "_raw": np.array(v.view(mx.uint16)) for k, v in captured.items()
                if v is not None and v.dtype == mx.bfloat16})
    print("saved", sys.argv[2])
    sys.exit(0)

# --compare
cap = np.load(sys.argv[2])

def to_mlx(name):
    if name + "_raw" in cap:
        raw = cap[name + "_raw"]
        return mx.array(np.frombuffer(raw.tobytes(), dtype=np.uint16).reshape(raw.shape)).view(mx.bfloat16)
    return mx.array(cap[name])

q = to_mlx("q"); k = to_mlx("k"); v = to_mlx("v")
a = to_mlx("a"); b = to_mlx("b")
A_log = to_mlx("A_log")
dt = mx.array(cap["dt_bias"]).reshape(1, 1, -1)
if dt.shape[-1] != v.shape[-2]:
    pass
state = mx.array(cap["state"])
mask = mx.ones((1, 1), mx.bool_)

def bits(x):
    return np.array(x.astype(mx.float32))

# composed path (gate off, dispatches the raw route -> use_fallback true -> C++ fallback)
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
import importlib
gd = importlib.import_module("mlx_lm.models.gated_delta")
out_c, st_c = gd.gated_delta_update(q, k, v, a, b, A_log, dt, state, mask, use_kernel=True)
mx.eval(out_c, st_c)

# fused path (repeat + raw fused kernel)
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "1"
Hk, Hv = q.shape[-2], v.shape[-2]
qr = mx.repeat(q, Hv // Hk, -2)
kr = mx.repeat(k, Hv // Hk, -2)
out_f, st_f = mx.fast.gated_delta_update_raw(qr, kr, v, a, b, A_log, dt, state, mask)
mx.eval(out_f, st_f)
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"

# fp64 numpy reference (the composed recursion, repeated q/k semantics)
q64 = bits(qr).astype(np.float64).reshape(32, 128)
k64 = bits(kr).astype(np.float64).reshape(32, 128)
v64 = bits(v).astype(np.float64).reshape(32, 128)
a64 = bits(a).astype(np.float64).reshape(32)
b64 = bits(b).astype(np.float64).reshape(32)
Al64 = bits(A_log).astype(np.float64).reshape(32)
dt64 = bits(dt).astype(np.float64).reshape(32)
s64 = bits(state).astype(np.float64).reshape(32, 128, 128)

beta = 1.0 / (1.0 + np.exp(-b64.astype(np.float64)))
g = np.exp(-np.exp(Al64) * np.logaddexp(0.0, (bits(a).astype(np.float64).reshape(32) + dt64)))
state_next = s64 * g[:, None, None]
kv = np.einsum("hvk,hk->hv", state_next, k64)
delta = (v64 - kv) * beta[:, None]
state_ref = state_next + delta[:, :, None] * k64[:, None, :]
out_ref = np.einsum("hvk,hk->hv", state_ref, q64)

def err(x, ref):
    d = np.abs(x.astype(np.float64) - ref)
    scale = np.abs(ref).max() + 1e-30
    return {"max_abs": float(d.max()), "rel_max": float(d.max() / scale)}

res = {
    "composed_vs_fp64": {"out": err(bits(out_c).reshape(32, 128), out_ref),
                         "state": err(bits(st_c).reshape(32, 128, 128), state_ref)},
    "fused_vs_fp64": {"out": err(bits(out_f).reshape(32, 128), out_ref),
                      "state": err(bits(st_f).reshape(32, 128, 128), state_ref)},
}
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[3], "w"), indent=1)
