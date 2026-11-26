"""tfref-tail.py with the single-query decode attention computed in float32 from stock ops (scores, softmax and P@V in float32, only the output cast to the model dtype).
The prelude below replaces mx.fast.scaled_dot_product_attention for q_len 1 without a mask; every other call (prefill, masked) goes to the original. No kernel change."""
import atexit

import mlx.core as mx

_calls = [0]
atexit.register(lambda: print("ATTN32 patched calls", _calls[0], flush=True))
_orig_sdpa = mx.fast.scaled_dot_product_attention


def _sdpa_f32(q, k, v, *args, scale=None, mask=None, **kw):
    if q.shape[2] != 1 or mask is not None or args:
        return _orig_sdpa(q, k, v, *args, scale=scale, mask=mask, **kw)
    _calls[0] += 1
    b, hq, _, d = q.shape
    hkv = k.shape[1]
    rep = hq // hkv
    qf = (q.astype(mx.float32) * scale).reshape(b, hkv, rep, d)
    s = qf @ k.astype(mx.float32).transpose(0, 1, 3, 2)
    p = mx.softmax(s, axis=-1, precise=True)
    o = p @ v.astype(mx.float32)
    return o.reshape(b, hq, 1, d).astype(q.dtype)


mx.fast.scaled_dot_product_attention = _sdpa_f32

#!/usr/bin/env python3
"""S=1 teacher-forced capture: one model forward per token, so the GDN decode kernel runs at every step; also lsef (LogSumExp of the same row cast to float32) and lse64 (LogSumExp of that row in float64 on the host, numpy), the independent reference of the per-op check; am64 and an64 are the host numpy index of the maximum and the minimum of that row (lowest index on ties).
Also the ArgMax and ArgMin index of the vocabulary row (am, an) of the float32 log-prob row, and the ArgMax and ArgMin index (amb, anb) and the LogSumExp (lseb) of the row in the model's own dtype (the bf16 logits row of the 9B, which is the row the decode step reduces), at every step.
Writes one npz per text segment: ids, target_lp (log-prob of the next text token), top_i / top_v (top 20 ids and log-probs).
usage: tfref.py --model DIR --text FILE --device gpu|cpu --dtype native|float32 --nseg N --maxtok T --stride S --budget SEC --out DIR
A segment stops early when the wall budget is spent; the comparison uses the steps all arms share."""
import argparse
import os
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--text", required=True)
ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
ap.add_argument("--dtype", choices=["native", "float32"], default="native")
ap.add_argument("--nseg", type=int, default=4)
ap.add_argument("--maxtok", type=int, default=160)
ap.add_argument("--stride", type=int, default=1500)
ap.add_argument("--budget", type=float, default=1e9)
ap.add_argument("--out", required=True)
a = ap.parse_args()

import mlx.core as mx  # noqa: E402

if a.device == "cpu":
    mx.set_default_device(mx.cpu)
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

model, tok = load(a.model)
if a.dtype == "float32":
    model.set_dtype(mx.float32)
all_ids = tok.encode(open(a.text, encoding="utf-8").read())
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
for s in range(a.nseg):
    ids = all_ids[s * a.stride : s * a.stride + a.maxtok]
    if len(ids) < 8 or time.time() - t0 > a.budget:
        break
    cache = make_prompt_cache(model)
    tl, ti, tv, tam, tan, tamb, tanb, tlse, tl64, tlsef, ta64, tn64 = [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(len(ids) - 1):
        out = model(mx.array([[ids[i]]], dtype=mx.int32), cache=cache)
        row = out[0, -1]
        lp = row.astype(mx.float32)
        lp = lp - mx.logsumexp(lp)
        idx = mx.argpartition(-lp, 19)[:20]
        v = lp[idx]
        order = mx.argsort(-v)
        idx, v = idx[order], v[order]
        am, an = mx.argmax(lp), mx.argmin(lp)
        amb, anb, lseb = mx.argmax(row), mx.argmin(row), mx.logsumexp(row).astype(mx.float32)
        lsef = mx.logsumexp(row.astype(mx.float32))
        mx.eval(lp[ids[i + 1]], idx, v, am, an, amb, anb, lseb, lsef)
        x64 = np.array(row.astype(mx.float32)).astype(np.float64)
        m64 = x64.max()
        tl64.append(float(m64 + np.log(np.exp(x64 - m64).sum())))
        tlsef.append(float(lsef))
        ta64.append(int(np.argmax(x64)))
        tn64.append(int(np.argmin(x64)))
        tamb.append(int(amb))
        tanb.append(int(anb))
        tlse.append(float(lseb))
        tam.append(int(am))
        tan.append(int(an))
        tl.append(float(lp[ids[i + 1]]))
        ti.append(np.array(idx))
        tv.append(np.array(v))
        if time.time() - t0 > a.budget:
            break
    n = len(tl)
    np.savez(os.path.join(a.out, f"seg{s}.npz"), ids=np.array(ids[: n + 1]), target_lp=np.array(tl),
             top_i=np.stack(ti), top_v=np.stack(tv), am=np.array(tam), an=np.array(tan),
             amb=np.array(tamb), anb=np.array(tanb), lseb=np.array(tlse), lsef=np.array(tlsef), lse64=np.array(tl64), am64=np.array(ta64), an64=np.array(tn64))
    print(f"TFREF seg {s} steps {n} device {a.device} dtype {a.dtype} elapsed {time.time() - t0:.1f}s", flush=True)
print(f"TFREF done device {a.device} dtype {a.dtype} mlx {mx.__version__}", flush=True)
