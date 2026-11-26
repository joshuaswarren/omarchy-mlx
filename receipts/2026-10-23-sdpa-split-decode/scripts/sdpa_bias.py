"""Per-op error of the decode attention routes against a float64 reference at decode shapes, signed and in bf16 steps.
Single query, 32 query heads, 8 key-value heads of width 128, bf16 q, K, V. Arms in one process (MLX_OMARCHY_SDPA_DECODE_SPLIT is read live): fused (flag 0), split (flag 1), and the float32 composition
from stock ops (scores, softmax and P@V in float32, output cast to bf16). For each cache length and score spread (the standard deviation of the scaled scores is s squared) it prints, per arm against float64:
mean signed error in units of one bf16 step of the reference value, mean absolute error in steps, the 99.9th percentile of the absolute error in steps, and the relative bias mean(e * sign(ref)) / mean(|ref|);
and the split-minus-fused difference in steps, the bit-equality of fused against the float32 composition, and the same table (BIASM lines) restricted to elements whose reference magnitude exceeds a tenth of the mean magnitude. usage: sdpa_bias.py TRIALS"""
import os
import sys

import mlx.core as mx
import numpy as np

trials = int(sys.argv[1])
HQ, HKV, D = 32, 8, 128
SCALE = D**-0.5
print("BIAS mlx", mx.__version__, mx.default_device(), "trials", trials, flush=True)


def route(q, k, v, flag):
    os.environ["MLX_OMARCHY_SDPA_DECODE_SPLIT"] = flag
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)
    mx.eval(o)
    return np.array(o.astype(mx.float32), dtype=np.float64)


def f32comp(q, k, v):
    rep = HQ // HKV
    qf = (q.astype(mx.float32) * SCALE).reshape(1, HKV, rep, D)
    s = qf @ k.astype(mx.float32).transpose(0, 1, 3, 2)
    p = mx.softmax(s, axis=-1, precise=True)
    o = (p @ v.astype(mx.float32)).reshape(1, HQ, 1, D).astype(mx.bfloat16)
    mx.eval(o)
    return np.array(o.astype(mx.float32), dtype=np.float64)


def reference(q, k, v):
    qd = np.array(q.astype(mx.float32), dtype=np.float64).reshape(HKV, HQ // HKV, D) * SCALE
    kd = np.array(k.astype(mx.float32), dtype=np.float64).reshape(HKV, -1, D)
    vd = np.array(v.astype(mx.float32), dtype=np.float64).reshape(HKV, -1, D)
    s = np.einsum("grd,gkd->grk", qd, kd)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("grk,gkd->grd", p, vd).reshape(HQ, D)


def step(ref):
    a = np.maximum(np.abs(ref), 1e-30)
    return 2.0 ** (np.floor(np.log2(a)) - 7)


rng = np.random.default_rng(1)
for spread in (1.0, 3.0, 5.0):
    sd = spread**0.5
    for kv in (300, 1000, 2000, 4000, 7000):
        acc = {n: [] for n in ("fused", "split", "f32comp")}
        diffs = []
        fdiff = []
        refs = []
        for _ in range(trials):
            q = mx.array(rng.standard_normal((1, HQ, 1, D)).astype(np.float32) * sd).astype(mx.bfloat16)
            k = mx.array(rng.standard_normal((1, HKV, kv, D)).astype(np.float32) * sd).astype(mx.bfloat16)
            v = mx.array(rng.standard_normal((1, HKV, kv, D)).astype(np.float32)).astype(mx.bfloat16)
            mx.eval(q, k, v)
            ref = reference(q, k, v)
            outs = {"fused": route(q, k, v, "0").reshape(HQ, D), "split": route(q, k, v, "1").reshape(HQ, D), "f32comp": f32comp(q, k, v).reshape(HQ, D)}
            st = step(ref)
            for n, o in outs.items():
                acc[n].append(((o - ref) / st).ravel())
            diffs.append(((outs["split"] - outs["fused"]) / st).ravel())
            fdiff.append(((outs["fused"] - outs["f32comp"]) / st).ravel())
            refs.append(ref.ravel())
        refv = np.concatenate(refs)
        line = f"BIAS spread {spread:.0f} kv {kv}:"
        for n in ("fused", "split", "f32comp"):
            e = np.concatenate(acc[n])
            bias = (e * np.sign(refv) * step(refv)).mean() / np.abs(refv).mean()
            line += f" | {n} mean {e.mean():+.4f} abs {np.abs(e).mean():.4f} p999 {np.percentile(np.abs(e), 99.9):.2f} relbias {bias:+.2e}"
        d = np.concatenate(diffs)
        line += f" | split-fused mean {d.mean():+.4f} abs {np.abs(d).mean():.4f} bit-equal {np.mean(d == 0):.3f}"
        fd = np.concatenate(fdiff)
        line += f" | fused-f32comp bit-equal {np.mean(fd == 0):.4f} max {np.abs(fd).max():.2f} steps"
        print(line, flush=True)
        big = np.abs(refv) > 0.1 * np.abs(refv).mean()
        line2 = f"BIASM spread {spread:.0f} kv {kv} (|ref| > 0.1 mean|ref|, {big.mean():.2f} of elements):"
        for n in ("fused", "split", "f32comp"):
            e = np.concatenate(acc[n])[big]
            line2 += f" | {n} mean {e.mean():+.4f} abs {np.abs(e).mean():.4f} p999 {np.percentile(np.abs(e), 99.9):.2f}"
        dm = d[big]
        line2 += f" | split-fused mean {dm.mean():+.4f} abs {np.abs(dm).mean():.4f} bit-equal {np.mean(dm == 0):.3f}"
        print(line2, flush=True)
print("BIAS done", flush=True)
