"""f16 a @ b 4096^3 only: same timing loop as metal_baseline (3 warm-ups, then median).
usage: python nnprobe.py [reps]  -> one JSON line"""
import json
import sys
import time

import mlx.core as mx

N, REPS = 4096, int(sys.argv[1]) if len(sys.argv) > 1 else 30
mx.random.seed(0)
a = mx.random.normal((N, N)).astype(mx.float16)
b = mx.random.normal((N, N)).astype(mx.float16)
mx.eval(a, b)
for _ in range(3):
    mx.eval(a @ b)
t = []
for _ in range(REPS):
    t0 = time.perf_counter()
    mx.eval(a @ b)
    t.append(time.perf_counter() - t0)
t.sort()
print(json.dumps({"mlx": mx.__version__, "median_ms": round(t[len(t) // 2] * 1e3, 3),
                  "tflops": round(2 * N ** 3 / t[len(t) // 2] / 1e12, 3)}), flush=True)
