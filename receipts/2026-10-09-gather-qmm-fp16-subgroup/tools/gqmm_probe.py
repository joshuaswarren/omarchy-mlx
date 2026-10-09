"""MatmulGap H44 probe: mx.gather_qmm at the MoE decode shape (Qwen3-30B-A3B layer: hidden 2048, expert intermediate 768, 4-bit group
64, top-8 of S slots, one token), one projection set (gate, up, down) per "layer", dtypes fp16 and bf16.
Arms per dtype (full eval per iteration, median of N after warm-up):
  gq_default : gather_qmm with the route the wheel picks (env untouched)
  gq_scalar  : gather_qmm with MLX_OMARCHY_GATHER_QMM_SUB=0 (the scalar kernel)
  take_qmm   : mx.take of the 8 routed experts + one batched quantized_matmul per projection (w7G's workaround)
Output hashes (sha256, first 16 hex, of the float32-converted layer output) say whether arms are bit-identical.
One pass per arm with MLX_OMARCHY_TRACE_DISPATCH on: markers '@@ARM <dtype> <arm>' / '@@END' on stderr beside the DISPATCH lines.
usage: gqmm_probe.py [slots=40] [iters=20]; prints one JSON line per dtype on stdout."""
import hashlib
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

S = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
D, I, K, GS, BITS = 2048, 768, 8, 64, 4
SLOTS = list(range(0, S, max(1, S // K)))[:K]
idx = mx.array([[SLOTS]], dtype=mx.uint32)
sl = mx.array(SLOTS, dtype=mx.uint32)
sys.stderr.write(f"@@HOST {mx.__version__} {mx.device_info().get('device_name')}\n")


def quant(shape, dt, seed):
    mx.random.seed(seed)
    w = (mx.random.normal(shape) * 0.02).astype(dt)
    wq, s, b = mx.quantize(w, group_size=GS, bits=BITS)
    mx.eval(wq, s, b)
    return wq, s, b


def med(f, n=N, warm=3):
    for _ in range(warm):
        f()
    ms = []
    for _ in range(n):
        t = time.perf_counter_ns()
        f()
        ms.append((time.perf_counter_ns() - t) / 1e6)
    ms.sort()
    return round(ms[len(ms) // 2], 3), round(100 * (ms[-1] / ms[0] - 1), 1)


def h(a):
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

    def layer_gq():
        g = gq(g_, x)
        return gq(d_, mx.sigmoid(g) * g * gq(u_, x))

    def layer_take():
        xx = mx.broadcast_to(x.reshape(1, 1, D), (K, 1, D))
        g = qb(g_, xx)
        return qb(d_, mx.sigmoid(g) * g * qb(u_, xx))

    arms = {}
    for arm, fn, sub in (("gq_default", layer_gq, None), ("gq_scalar", layer_gq, "0"), ("take_qmm", layer_take, None)):
        if sub is None:
            os.environ.pop("MLX_OMARCHY_GATHER_QMM_SUB", None)
        else:
            os.environ["MLX_OMARCHY_GATHER_QMM_SUB"] = sub
        out = fn()
        mx.eval(out)
        hh = h(out)
        os.environ["MLX_OMARCHY_TRACE_DISPATCH"] = "1"
        sys.stderr.write(f"@@ARM {name} {arm}\n")
        sys.stderr.flush()
        mx.eval(fn())
        mx.synchronize()
        sys.stderr.write("@@END\n")
        sys.stderr.flush()
        del os.environ["MLX_OMARCHY_TRACE_DISPATCH"]
        ms, spread = med(lambda fn=fn: mx.eval(fn()))
        arms[arm] = {"ms": ms, "spread_pct": spread, "hash": hh}
    os.environ.pop("MLX_OMARCHY_GATHER_QMM_SUB", None)
    print(json.dumps({"dtype": name, "slots": S, "arms": arms}), flush=True)
