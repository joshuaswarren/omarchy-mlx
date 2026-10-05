#!/usr/bin/env python3
"""M2 bisect probe: capture real activations through block 0 of the H3 int8
DiT and compare each int8 op against its f32 dequantize reference.

Run on the M2 in the tf-m2-venv (short gpu-turn ticket). Writes
~/tfport/bisect-report.txt + npz intermediates to ~/tfport/bisect/."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path.home() / "tfport/engine/minimax-h3-mlx"))
sys.path.insert(0, str(Path.home() / "tfport/engine"))

ROOT = Path(str(Path.home() / "tfport/h3/int8-dit"))
REPORT = Path.home() / "tfport/bisect-report.txt"
OUT = Path.home() / "tfport/bisect"
OUT.mkdir(exist_ok=True)
rep = open(REPORT, "w")


def say(msg):
    print(msg, flush=True)
    rep.write(str(msg) + "\n")
    rep.flush()


from tensorfold.families.h3.weights import load_int8_dit

dit = load_int8_dit(ROOT)
say("model loaded (lazy, int8 state)")

# ---- capture: block 0 input, per-op outputs, and block-0 output ------------
from tensorfold.families.h3 import dit as h3dit

cap = {}
orig_block_call = h3dit.Block.__call__


def cap_block(self, x, tables, rows, rotary):
    out = orig_block_call(self, x, tables, rows, rotary)
    try:
        cap["x_in"] = np.array(x.astype(mx.float32))
        cap["block_out"] = np.array(out.astype(mx.float32))
        say(f"block0 in {x.dtype} {x.shape} -> out {out.dtype} {out.shape}")
    except Exception as e:
        say(f"capture failed: {e!r}")
    return out


h3dit.Block.__call__ = cap_block

orig_il = type(dit.blocks[0].attn.qkv_proj).__call__
cap_il = {}


def cap_il_call(self, x):
    out = orig_il(self, x)
    if "qkv" not in cap_il:
        cap_il["qkv_in"] = np.array(x.astype(mx.float32))
        cap_il["qkv_out"] = np.array(out.astype(mx.float32))
        say(f"int8 qkv_proj in {x.dtype} {x.shape} -> out {out.dtype} {out.shape}")
    elif "out" not in cap_il:
        cap_il["out_in"] = np.array(x.astype(mx.float32))
        cap_il["out_out"] = np.array(out.astype(mx.float32))
        say(f"int8 out_proj in {x.dtype} {x.shape} -> out {out.dtype} {out.shape}")
    return out


# MLP capture
orig_mlpcall = type(dit.blocks[0].mlp).__call__
cap_mlp = {}


def cap_mlp_call(self, x):
    out = orig_mlpcall(self, x)
    if "mlp_out" not in cap_mlp:
        cap_mlp["in"] = np.array(x.astype(mx.float32))
        cap_mlp["mid"] = np.array(getattr(self, "_last_hidden", mx.zeros((1, 1))), copy=True)
        cap_mlp["mlp_out"] = np.array(out.astype(mx.float32))
        say(f"int8 mlp in {x.dtype} {x.shape} -> out {out.dtype} {out.shape}")
    return out


type(dit.blocks[0].attn.qkv_proj).__call__ = cap_il_call
type(dit.blocks[0].mlp).__call__ = cap_mlp_call

# run ONE forward (points 2) through the REAL pipeline with instrumentation
sys.argv = ["h3_generate-rows.py", str(Path.home() / "tfport/h3"),
            "--int8-from-state", str(Path.home() / "tfport/h3/int8-dit"),
            "--text-rows", str(Path.home() / "tfport/h3/text-rows.safetensors"),
            "--width", "768", "--height", "448", "--frames", "124",
            "--points", "2", "--seed", "0",
            "--dump-latents", str(Path.home() / "tfport/h3/latents-bisect.safetensors"),
            "-o", str(Path.home() / "tfport/h3/unused-bisect.mp4")]
import runpy

try:
    runpy.run_path(str(Path.home() / "tfport/h3_generate-rows.py"), run_name="__main__")
except Exception as e:
    say(f"forward raised (ok if block 0 ran first): {e!r}")

if "x_in" not in cap:
    say("NO CAPTURE - forward did not reach block 0")
    sys.exit(1)

np.savez(OUT / "captures.npz", **{f"cap_{k}": v for k, v in cap.items()})
np.savez(OUT / "ops.npz", **{f"il_{k}": v for k, v in cap_il.items()},
         **{f"mlp_{k}": v for k, v in cap_mlp.items()})

# ---- f32 references from the SAME int8 weights/scales ----------------------
def dequant(il, m):
    w = np.array(il.weight, dtype=np.float64).reshape(il.weight.shape)
    s = np.array(il.scales, dtype=np.float64).reshape(il.scales.shape)
    return (w.astype(np.int32) * s).astype(np.float32)


qkv_il = dit.blocks[0].attn.qkv_proj
x_in = cap_il["qkv_in"].astype(np.float64)
w64 = np.array(qkv_il.weight, dtype=np.int32).astype(np.float64)
s64 = np.array(qkv_il.scales, dtype=np.float64).reshape(1, -1)
ref_qkv = (x_in @ w64.T * s64)
rel_qkv = float(np.linalg.norm(cap_il["qkv_out"].astype(np.float64) - ref_qkv) / np.linalg.norm(ref_qkv))
say(f"qkv_proj rel-L2 (int8 op vs f32 dequant ref): {rel_qkv:.6f}")

out_il = dit.blocks[0].attn.out_proj
xin = cap_il["out_in"].astype(np.float64)
w64 = np.array(out_il.weight, dtype=np.int32).astype(np.float64)
s64 = np.array(out_il.scales, dtype=np.float64).reshape(1, -1)
ref_out = (xin @ w64.T * s64)
rel_out = float(np.linalg.norm(cap_il["out_out"].astype(np.float64) - ref_out) / np.linalg.norm(ref_out))
say(f"out_proj rel-L2: {rel_out:.6f}")

mlp = dit.blocks[0].mlp
xin = cap_mlp["in"].astype(np.float64)
w1 = np.array(mlp.w1, dtype=np.int32).astype(np.float64)
s1 = np.array(mlp.s1, dtype=np.float64).reshape(1, -1)
ref_fc1 = (xin @ w1.T * s1)
gate = ref_fc1[:, : ref_fc1.shape[1] // 2]
value = ref_fc1[:, ref_fc1.shape[1] // 2:]
ref_hidden = (gate / (1 + np.exp(-gate)) * value)
h64 = ref_hidden.astype(np.float64)
w2 = np.array(mlp.w2, dtype=np.int32).astype(np.float64)
s2 = np.array(mlp.s2, dtype=np.float64).reshape(1, -1)
ref_fc2 = (h64 @ w2.T * s2)
rel_mlp = float(np.linalg.norm(cap_mlp["mlp_out"].astype(np.float64) - ref_fc2) / np.linalg.norm(ref_fc2))
say(f"mlp rel-L2: {rel_mlp:.6f}")

say("BISECT-OP-DONE")
