#!/usr/bin/env python3
"""decode tok/s, Vulkan submissions per token and generated ids for one batch-work setting
(MLX_OMARCHY_BATCH_WORK is set by the caller). Qwen3.8-2B 4-bit, fixed prompt, greedy, GEN tokens.
usage: decode_sub.py MODEL_DIR OUT.json LABEL [GEN=128]"""
import ctypes, hashlib, json, os, sys, time
import mlx.core as mx
from mlx_lm import load, stream_generate
mdl, out, label = sys.argv[1], sys.argv[2], sys.argv[3]; GEN = int(sys.argv[4]) if len(sys.argv) > 4 else 128
lib = os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so")
prov = {"mlx": mx.__version__, "libmlx_sha256": hashlib.sha256(open(lib, "rb").read()).hexdigest(), "label": label,
        "env": {k: v for k, v in os.environ.items() if k.startswith(("MLX_OMARCHY_BATCH", "HK_"))}}
print("PROVENANCE " + json.dumps(prov), flush=True)
L = ctypes.CDLL(lib)
class Snap(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in ("dispatches", "submissions", "copies", "fills", "compute", "finalize",
                                                "commit_work", "commit_noop", "barriers", "barriers_skipped")]
assert L.mlx_omarchy_trace_snapshot_abi_size.restype is not None or True
L.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
assert L.mlx_omarchy_trace_snapshot_abi_size() == ctypes.sizeof(Snap)
def snap():
    s = Snap(); L.mlx_omarchy_trace_snapshot(ctypes.byref(s)); return s
model, tok = load(mdl)
prompt = "The history of France begins in antiquity. " * 60
for _ in stream_generate(model, tok, prompt, max_tokens=8): pass
ids = []; tf = tl = None; s0 = None
for r in stream_generate(model, tok, prompt, max_tokens=GEN):
    t = time.perf_counter()
    if tf is None: tf = t; s0 = snap()
    tl = t; ids.append(int(r.token))
s1 = snap(); steps = len(ids) - 1
res = {"prov": prov, "tokens": len(ids), "decode_tok_s": round(steps / (tl - tf), 3),
       "submits_per_token": round((s1.submissions - s0.submissions) / steps, 2),
       "dispatches_per_token": round((s1.compute - s0.compute) / steps, 1),
       "finalize_per_token": round((s1.finalize - s0.finalize) / steps, 2),
       "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(), "ids_head": ids[:16]}
with open(out, "w") as f: json.dump(res, f)
print(json.dumps(res), flush=True)
