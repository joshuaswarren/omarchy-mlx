"""Decode attention microbench, one Mac, one process, the ledger stack. usage: sdpa_bench.py REPEATS
Single-query grouped-query attention as in decode: q (1, 32, 1, 128) against K and V (1, 8, KV, 128), bf16, scale 128^-0.5, no mask, for KV in 128 to 1024 in steps of 32 to 64 (the break-even sweep).
Two shapes of work, each the median of REPEATS timed evaluations after one warm-up:
  independent: 36 calls on 36 distinct K and V sets in one evaluation (a throughput number: microseconds per call, and the K+V bytes per call divided by it);
  chain: 36 calls where the output of one call is the query of the next (output reshaped to the query shape), the dependency chain of a decode step (microseconds per call).
Lines start with SDPA."""
import statistics
import sys
import time

import mlx.core as mx

reps = int(sys.argv[1])
HQ, HKV, D, N = 32, 8, 128, 36
SCALE = D**-0.5
print(f"SDPA mlx {mx.__version__} device {mx.default_device()}", flush=True)


def timed(fn):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts), min(ts)


for kv in (128, 192, 256, 288, 320, 352, 384, 416, 448, 512, 640, 768, 1024):
    ks = [mx.random.normal((1, HKV, kv, D), dtype=mx.bfloat16) for _ in range(N)]
    vs = [mx.random.normal((1, HKV, kv, D), dtype=mx.bfloat16) for _ in range(N)]
    q0 = mx.random.normal((1, HQ, 1, D), dtype=mx.bfloat16)
    mx.eval(ks, vs, q0)
    nbytes = 2 * HKV * kv * D * 2

    def independent():
        mx.eval([mx.fast.scaled_dot_product_attention(q0, ks[i], vs[i], scale=SCALE) for i in range(N)])

    def chain():
        q = q0
        for i in range(N):
            q = mx.fast.scaled_dot_product_attention(q, ks[i], vs[i], scale=SCALE)
        mx.eval(q)

    mi, bi = timed(independent)
    mc, bc = timed(chain)
    print(f"SDPA kv {kv}: independent {mi / N * 1e6:.1f} us per call ({nbytes / (mi / N) / 1e9:.1f} GB/s of K+V, best {bi / N * 1e6:.1f} us); chain {mc / N * 1e6:.1f} us per call (best {bc / N * 1e6:.1f})", flush=True)
    del ks, vs
print("SDPA done", flush=True)
