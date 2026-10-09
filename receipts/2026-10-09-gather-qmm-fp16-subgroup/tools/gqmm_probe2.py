"""MatmulGap H45 probe (v2 of gqmm_probe.py): mx.gather_qmm at the MoE decode shape (hidden 2048, expert intermediate 768, 4-bit group 64,
top-8 of S slots, one token; one gate/up/down set per 'layer'), fp16 and bf16.
Arms (full eval per iteration, N iterations after warm-up): gq_default (the wheel's route), gq_scalar (MLX_OMARCHY_GATHER_QMM_SUB=0),
take_qmm (take of the routed experts + one batched quantized_matmul per projection).
Per arm: median ms, spread = (p75 - p25) / median in percent (a cell is void above 10 %), output hash, and the error against a float32
reference (dequantize the 8 routed experts, run the layer in float32): max abs error / max abs reference.
One dispatch trace per arm on stderr ('@@ARM <dtype> <arm>' ... '@@END').
Per-submit guard: any single eval over 20 s aborts. usage: gqmm_probe2.py [slots=40] [iters=60]"""
import hashlib
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

S = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
D, I, K, GS, BITS = 2048, 768, 8, 64, 4
SLOTS = list(range(0, S, max(1, S // K)))[:K]
idx = mx.array([[SLOTS]], dtype=mx.uint32)
sl = mx.array(SLOTS, dtype=mx.uint32)
LIMIT_S = 20.0
sys.stderr.write(f"@@HOST {mx.__version__} {mx.device_info().get('device_name')}\n")


def quant(shape, dt, seed):
    mx.random.seed(seed)
    w = (mx.random.normal(shape) * 0.02).astype(dt)
    wq, s, b = mx.quantize(w, group_size=GS, bits=BITS)
    mx.eval(wq, s, b)
    return wq, s, b


def timed_eval(fn):
    t = time.perf_counter()
    mx.eval(fn())
    dt = time.perf_counter() - t
    if dt > LIMIT_S:
        raise SystemExit(f"ABORT: one eval took {dt:.1f} s, over the {LIMIT_S:.0f} s submit limit")
    return dt * 1e3


def med_iqr(fn, n=N, warm=5):
    for _ in range(warm):
        timed_eval(fn)
    ms = sorted(timed_eval(fn) for _ in range(n))
    med = ms[len(ms) // 2]
    return round(med, 3), round(100 * (ms[(3 * len(ms)) // 4] - ms[len(ms) // 4]) / med, 1)


def sha(a):
    return hashlib.sha256(np.array(a.astype(mx.float32)).tobytes()).hexdigest()[:16]


for dt, name in ((mx.float16, "fp16"), (mx.bfloat16, "bf16")):
    g_, u_, d_ = quant((S, I, D), dt, 1), quant((S, I, D), dt, 2), quant((S, D, I), dt, 3)
    mx.random.seed(4)
    x = mx.random.normal((1, 1, 1, 1, D)).astype(dt)
    mx.eval(x)

    def gq(w, hh):
        return mx.gather_qmm(hh, w[0], w[1], w[2], rhs_indices=idx, transpose=True, group_size=GS, bits=BITS)

    def qb(w, hh):
        return mx.quantized_matmul(hh, mx.take(w[0], sl, axis=0), mx.take(w[1], sl, axis=0), mx.take(w[2], sl, axis=0),
                                   transpose=True, group_size=GS, bits=BITS)

    def layer_gq(x=x, g_=g_, u_=u_, d_=d_):
        g = gq(g_, x)
        return gq(d_, mx.sigmoid(g) * g * gq(u_, x))

    def layer_take(x=x, g_=g_, u_=u_, d_=d_):
        xx = mx.broadcast_to(x.reshape(1, 1, D), (K, 1, D))
        g = qb(g_, xx)
        return qb(d_, mx.sigmoid(g) * g * qb(u_, xx))

    def deq(w, e):
        return mx.dequantize(w[0][e], w[1][e], w[2][e], group_size=GS, bits=BITS).astype(mx.float32)

    xf = x.reshape(D).astype(mx.float32)
    ys = []
    for e in SLOTS:
        gg = deq(g_, e) @ xf
        ys.append(deq(d_, e) @ (mx.sigmoid(gg) * gg * (deq(u_, e) @ xf)))
    ref = mx.stack(ys)  # (K, D)
    mx.eval(ref)
    ref_max = float(mx.max(mx.abs(ref)).item()) or 1.0

    def err(out):
        o = out.reshape(K, D).astype(mx.float32)
        return round(float(mx.max(mx.abs(o - ref)).item()) / ref_max, 6)

    arms = {}
    for arm, fn, sub in (("gq_default", layer_gq, None), ("gq_scalar", layer_gq, "0"), ("take_qmm", layer_take, None)):
        if sub is None:
            os.environ.pop("MLX_OMARCHY_GATHER_QMM_SUB", None)
        else:
            os.environ["MLX_OMARCHY_GATHER_QMM_SUB"] = sub
        out = fn()
        mx.eval(out)
        o_hash, o_err = sha(out), err(out)
        os.environ["MLX_OMARCHY_TRACE_DISPATCH"] = "1"
        sys.stderr.write(f"@@ARM {name} {arm}\n")
        sys.stderr.flush()
        mx.eval(fn())
        mx.synchronize()
        sys.stderr.write("@@END\n")
        sys.stderr.flush()
        del os.environ["MLX_OMARCHY_TRACE_DISPATCH"]
        ms, iqr = med_iqr(lambda fn=fn: fn())
        arms[arm] = {"ms": ms, "iqr_pct": iqr, "hash": o_hash, "err_vs_fp32": o_err}
    os.environ.pop("MLX_OMARCHY_GATHER_QMM_SUB", None)
    print(json.dumps({"dtype": name, "slots": S, "iters": N, "arms": arms}), flush=True)
