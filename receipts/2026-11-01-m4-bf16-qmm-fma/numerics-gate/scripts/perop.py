#!/usr/bin/env python3
"""docs/numerics-gate.md check 1: per-op error on captured operands against fp64.

  perop.py capture --model DIR --teacher teacher.json --out ops.npz     (reference venv, CPU fp32)
  perop.py run     --model DIR --ops ops.npz --kind K --out arm.npz     (GPU wheel, stamp-gated)
  perop.py score   --model DIR --ops ops.npz --kind K --base A.npz --cand B.npz --out score.json
                                                                         (reference venv, numpy fp64)
capture: the reference forward (stock MLX, CPU, float32) over the first 128 tokens of
segments 0 and 5 of the gate text records, at the first, middle and last attention layer, the residual stream
entering the layer, the attention output entering o_proj and the residual entering the MLP
norm. Operands are stored rounded to the model's storage dtype, so every arm and the fp64
reference read exactly the same values.

Kinds (each is the op set its PR changes, run through the model's own modules):
  decode   single-row decode ops (#316): input norm, q/k/v projections (with bias where
           the model has it), RoPE of q and k at the row's position, o_proj plus the
           residual, post norm and the MLP. Rows are evaluated one at a time, all outputs
           of a row in one eval, so the planner sees the decode graph.
  linear   dense bf16 linear layers, 128-row blocks (#317): every projection x @ W.T.
  qmm      4-bit quantized projections, 64-row blocks (#321): every projection.
score: relative L2 error of each output against an fp64 numpy forward of the same operands
(dequantized weights, no intermediate rounding), mean over datasets, per layer and output.
"""
import argparse
import json
import math
import os
import sys

LAYER_FRACS = (0.0, 0.5, 1.0)
BLOCK_SEGMENTS = (0, 5)
DECODE_ROWS = (3, 17, 40, 63, 90, 127)  # positions within each 128-token block


def layers_of(model):
    return model.model.layers


def heads(model):
    args = model.args
    return args.num_attention_heads, args.num_key_value_heads


def pick_layers(ls):
    """First, middle and last layer that has attention and an MLP (hybrid models
    interleave other mixers)."""
    idx = [i for i, L in enumerate(ls) if "self_attn" in L and "mlp" in L]
    n = len(idx)
    return sorted({idx[min(n - 1, int(round(f * (n - 1))))] for f in LAYER_FRACS})


def capture(a):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    mx.set_default_device(mx.cpu)
    model, _ = load(a.model)
    store_dtype = next(v.dtype for k, v in nn.utils.tree_flatten(model.parameters())
                       if k.endswith("scales") or (k.endswith("weight") and v.dtype != mx.uint32))
    model.set_dtype(mx.float32)
    tch = json.load(open(a.teacher))
    ls = layers_of(model)
    picked = pick_layers(ls)
    rec = {}

    class Tap(nn.Module):
        def __init__(self, inner, key):
            super().__init__()
            self.inner, self.key = inner, key

        def __call__(self, x, *args, **kw):
            rec.setdefault(self.key, []).append(x)
            return self.inner(x, *args, **kw)

    for li in picked:
        L = ls[li]
        L.input_layernorm = Tap(L.input_layernorm, f"h_{li}")
        L.post_attention_layernorm = Tap(L.post_attention_layernorm, f"h2_{li}")
        L.self_attn.o_proj = Tap(L.self_attn.o_proj, f"attn_{li}")
        L.mlp.down_proj = Tap(L.mlp.down_proj, f"s_{li}")
        L.self_attn.q_proj = Tap(L.self_attn.q_proj, f"xn_{li}")
        L.mlp.gate_proj = Tap(L.mlp.gate_proj, f"hn_{li}")
    out = {}
    for si in BLOCK_SEGMENTS:
        p = tch["prompts"][si]
        ids = (p["prompt_ids"] + p["tokens"])[:128]
        rec.clear()
        model(mx.array(ids)[None], cache=make_prompt_cache(model))
        for k, v in rec.items():
            x = v[0][0].astype(store_dtype).astype(mx.float32)
            mx.eval(x)
            out[f"{k}_seg{si}"] = np.array(x)
    out["layers"] = np.array(picked)
    out["store_dtype"] = np.array(str(store_dtype))
    np.savez(a.out, **out)
    print(json.dumps({"layers": picked, "store_dtype": str(store_dtype), "arrays": len(out)}))


def run(a):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    import prov
    line = prov.check()
    model, _ = load(a.model)
    ops = np.load(a.ops)
    dt = {"mlx.core.bfloat16": mx.bfloat16, "mlx.core.float16": mx.float16}[str(ops["store_dtype"])]
    ls = layers_of(model)
    res = {}
    for li in ops["layers"].tolist():
        L = ls[li]
        at, mlp = L.self_attn, L.mlp
        for si in BLOCK_SEGMENTS:
            g = lambda k: mx.array(ops[f"{k}_{li}_seg{si}"]).astype(dt)  # noqa: E731
            h, attn, h2, s = g("h"), g("attn"), g("h2"), g("s")
            if a.kind == "decode":
                for r in DECODE_ROWS:
                    hr, ar, h2r = h[r:r + 1][None], attn[r:r + 1][None], h2[r:r + 1][None]
                    xn = L.input_layernorm(hr)
                    q, k, v = at.q_proj(xn), at.k_proj(xn), at.v_proj(xn)
                    nh, nk = heads(model)
                    qr = at.rope(q.reshape(1, 1, nh, -1).transpose(0, 2, 1, 3), offset=r)
                    kr = at.rope(k.reshape(1, 1, nk, -1).transpose(0, 2, 1, 3), offset=r)
                    o = at.o_proj(ar) + hr
                    hn = L.post_attention_layernorm(h2r)
                    d = mlp(hn)
                    outs = {"xn": xn, "q": q, "k": k, "v": v, "q_rope": qr, "k_rope": kr, "o_res": o,
                            "hn": hn, "mlp": d}
                    mx.eval(list(outs.values()))
                    for name, val in outs.items():
                        res[f"{li}|{si}|{r}|{name}"] = np.array(val.astype(mx.float32)).reshape(-1)
            else:
                rows = 128 if a.kind == "linear" else 64
                xn, hn = g("xn")[:rows], g("hn")[:rows]
                proj = {"q": (at.q_proj, xn), "k": (at.k_proj, xn), "v": (at.v_proj, xn),
                        "o": (at.o_proj, attn[:rows]), "gate": (mlp.gate_proj, hn), "up": (mlp.up_proj, hn),
                        "down": (mlp.down_proj, s[:rows])}
                for name, (mod, x) in proj.items():
                    y = mod(x[None])
                    mx.eval(y)
                    res[f"{li}|{si}|0|{name}"] = np.array(y.astype(mx.float32)).reshape(-1)
                    res[f"{li}|{si}|0|{name}_in"] = np.array(x.astype(mx.float32)).reshape(-1)
    np.savez(a.out, **res)
    print(json.dumps({"outputs": len(res), "commit": os.environ.get("EXPECT_COMMIT", ""),
                      "env": {k: v for k, v in os.environ.items() if k.startswith(("MLX_OMARCHY_", "HK_"))}}))


def score(a):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    mx.set_default_device(mx.cpu)
    model, _ = load(a.model)
    ops = np.load(a.ops)
    base, cand = np.load(a.base), np.load(a.cand)
    ls = layers_of(model)

    def f64(x):
        return np.array(x.astype(mx.float32)).astype(np.float64)

    def weight(mod):
        if hasattr(mod, "scales"):
            w = np.array(mod.weight).astype(np.uint64)
            bits, gs = mod.bits, mod.group_size
            per = 32 // bits
            q = np.stack([(w >> (bits * i)) & ((1 << bits) - 1) for i in range(per)], -1).reshape(w.shape[0], -1)
            sc, bi = f64(mod.scales), f64(mod.biases)
            W = q.astype(np.float64) * np.repeat(sc, gs, 1) + np.repeat(bi, gs, 1)
        else:
            W = f64(mod.weight)
        b = f64(mod.bias) if "bias" in mod else None
        return W, b

    def lin(mod, x):
        W, b = weight(mod)
        y = x @ W.T
        return y + b if b is not None else y

    def rms(mod, x):
        return x / np.sqrt(np.mean(x * x, -1, keepdims=True) + mod.eps) * f64(mod.weight)

    def rope(mod, x, off):  # x: heads x dim
        dims, base_, scale = mod.dims, mod.base, mod.scale
        half = dims // 2
        inv = base_ ** (-np.arange(half, dtype=np.float64) / half)
        ang = (off * scale) * inv
        c, s_ = np.cos(ang), np.sin(ang)
        y = x.copy()
        if mod.traditional:
            a1, a2 = x[:, 0:dims:2], x[:, 1:dims:2]
            y[:, 0:dims:2], y[:, 1:dims:2] = a1 * c - a2 * s_, a1 * s_ + a2 * c
        else:
            a1, a2 = x[:, :half], x[:, half:dims]
            y[:, :half], y[:, half:dims] = a1 * c - a2 * s_, a1 * s_ + a2 * c
        return y

    def silu(x):
        return x / (1 + np.exp(-x))

    cells = {}
    for key in base.files:
        li, si, r, name = key.split("|")
        if name.endswith("_in"):
            continue
        li, si, r = int(li), int(si), int(r)
        L = ls[li]
        at, mlp = L.self_attn, L.mlp
        op = lambda k: ops[f"{k}_{li}_seg{si}"].astype(np.float64)  # noqa: E731
        if a.kind == "decode":
            h, attn, h2 = op("h")[r:r + 1], op("attn")[r:r + 1], op("h2")[r:r + 1]
            xn = rms(L.input_layernorm, h)
            if name == "xn":
                ref = xn
            elif name in ("q", "k", "v"):
                ref = lin(getattr(at, f"{name}_proj"), xn)
            elif name in ("q_rope", "k_rope"):
                p = lin(getattr(at, f"{name[0]}_proj"), xn)
                nh, nk = heads(model)
                ref = rope(at.rope, p.reshape(nh if name[0] == "q" else nk, -1), r)
            elif name == "o_res":
                ref = lin(at.o_proj, attn) + h
            elif name == "hn":
                ref = rms(L.post_attention_layernorm, h2)
            elif name == "mlp":
                hn = rms(L.post_attention_layernorm, h2)
                ref = lin(mlp.down_proj, silu(lin(mlp.gate_proj, hn)) * lin(mlp.up_proj, hn))
        else:
            x = base[f"{li}|{si}|0|{name}_in"].astype(np.float64)
            if not np.array_equal(base[f"{li}|{si}|0|{name}_in"], cand[f"{li}|{si}|0|{name}_in"]):
                sys.exit(f"operand mismatch between arms at {key}")
            mod = {"q": at.q_proj, "k": at.k_proj, "v": at.v_proj, "o": at.o_proj,
                   "gate": mlp.gate_proj, "up": mlp.up_proj, "down": mlp.down_proj}[name]
            W, _ = weight(mod)
            ref = lin(mod, x.reshape(-1, W.shape[1]))
        ref = ref.reshape(-1)
        errs = {}
        for arm, d in (("base", base), ("cand", cand)):
            y = d[key].astype(np.float64)
            if not np.all(np.isfinite(y)):
                sys.exit(f"non-finite output {arm} {key}")
            errs[arm] = (float(np.linalg.norm(y - ref) / np.linalg.norm(ref)), float(np.max(np.abs(y - ref))))
        errs["identical"] = bool(np.array_equal(base[key], cand[key]))
        cells.setdefault(f"layer{li}/{name}", []).append(errs)
    table = []
    for cell, rows in sorted(cells.items()):
        b = sum(r["base"][0] for r in rows) / len(rows)
        c = sum(r["cand"][0] for r in rows) / len(rows)
        table.append({"cell": cell, "datasets": len(rows), "base_rel_l2_mean": b, "cand_rel_l2_mean": c,
                      "base_max_abs": max(r["base"][1] for r in rows), "cand_max_abs": max(r["cand"][1] for r in rows),
                      "outputs_identical": sum(r["identical"] for r in rows), "cand_no_worse": c <= b})
    worse = [t for t in table if not t["cand_no_worse"]]
    summary = {"kind": a.kind, "cells": len(table), "cells_cand_no_worse": len(table) - len(worse),
               "worst_ratio": max((t["cand_rel_l2_mean"] / t["base_rel_l2_mean"] for t in table if t["base_rel_l2_mean"] > 0), default=1.0),
               "check1_per_op": not worse, "table": table}
    json.dump(summary, open(a.out, "w"), indent=1)
    print(f"{'cell':<22}{'n':>4}{'base relL2':>13}{'cand relL2':>13}{'identical':>10}")
    for t in table:
        print(f"{t['cell']:<22}{t['datasets']:>4}{t['base_rel_l2_mean']:>13.4e}{t['cand_rel_l2_mean']:>13.4e}"
              f"{t['outputs_identical']:>7}/{t['datasets']:<3}")
    print(json.dumps({k: v for k, v in summary.items() if k != "table"}))


ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=("capture", "run", "score"))
ap.add_argument("--model", required=True)
ap.add_argument("--teacher")
ap.add_argument("--ops")
ap.add_argument("--kind", choices=("decode", "linear", "qmm"))
ap.add_argument("--base")
ap.add_argument("--cand")
ap.add_argument("--out", required=True)
a = ap.parse_args()
{"capture": capture, "run": run, "score": score}[a.mode](a)
