#!/usr/bin/env python3
"""Decode tok/s, Vulkan submissions/token, dispatches/token, and greedy token ids
for one batch-work setting (MLX_OMARCHY_BATCH_WORK set by caller).
Follows receipts/2026-11-21-g13g-batch-work-20000/tools/decode_sub.py with
mlx_provenance verification printed beside every run.

Usage: decode_sub.py MODEL_DIR OUT.json LABEL GEN EXPECT_WHEEL REPO_ROOT
"""
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path

mdl, out, label, gen_s, expect_wheel, repo_root = sys.argv[1:7]
GEN = int(gen_s)

sys.path.insert(0, str(Path(repo_root) / "scripts"))
import mlx_provenance as mp

prov = mp.installed_provenance(expect_wheel=Path(expect_wheel))
if prov["verified"] != "match" or prov["version_match"] is not True:
    print("PROVENANCE_REFUSAL", json.dumps(prov), file=sys.stderr)
    sys.exit(3)
pline = mp.provenance_line(prov)
print(pline, flush=True)

import mlx.core as mx
from mlx_lm import load, stream_generate

lib = os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so")
L = ctypes.CDLL(lib)

class Snap(ctypes.Structure):
    _fields_ = [
        (n, ctypes.c_uint64)
        for n in (
            "dispatches",
            "submissions",
            "copies",
            "fills",
            "compute",
            "finalize",
            "commit_work",
            "commit_noop",
            "barriers",
            "barriers_skipped",
        )
    ]

L.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
assert L.mlx_omarchy_trace_snapshot_abi_size() == ctypes.sizeof(Snap)

def snap():
    s = Snap()
    L.mlx_omarchy_trace_snapshot(ctypes.byref(s))
    return s

model, tok = load(mdl)
prompt = "The history of France begins in antiquity. " * 60
for _ in stream_generate(model, tok, prompt, max_tokens=8):
    pass

ids = []
tf = tl = None
s0 = None
for r in stream_generate(model, tok, prompt, max_tokens=GEN):
    t = time.perf_counter()
    if tf is None:
        tf = t
        s0 = snap()
    tl = t
    ids.append(int(r.token))
s1 = snap()
steps = len(ids) - 1
res = {
    "label": label,
    "model": os.path.basename(mdl.rstrip("/")),
    "device": str(mx.default_device()),
    "batch_work_env": os.environ.get("MLX_OMARCHY_BATCH_WORK", "unset"),
    "provenance": pline,
    "dist_version": prov["dist_version"],
    "verified": prov["verified"],
    "tokens": len(ids),
    "decode_tok_s": round(steps / (tl - tf), 3),
    "submits_per_token": round((s1.submissions - s0.submissions) / steps, 2),
    "dispatches_per_token": round((s1.compute - s0.compute) / steps, 1),
    "finalize_per_token": round((s1.finalize - s0.finalize) / steps, 2),
    "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
    "ids_head": ids[:16],
}
with open(out, "w") as f:
    json.dump(res, f, indent=2)
print(json.dumps(res), flush=True)
