"""MatmulGap H13-H16 row-shape check: the cells the direct-route rows apply to (all 4096^3).

Per cell: sha256 (16 hex) of the raw output bytes, median wall time of 5 timed runs after 2
warm-ups, TFLOP/s, and max |out - fp64| over 64 sampled output rows (fp64 truth on the exact
operand values). Equal hashes between two wheels mean identical stored bits.
usage: python rowcheck.py > out.jsonl
"""
import hashlib
import json
import time

import mlx.core as mx
import numpy as np

N = 4096
CELLS = [
    ("f16_nn", mx.float16, False, False),
    ("bf16_nn", mx.bfloat16, False, False),
    ("bf16_nt", mx.bfloat16, False, True),
    ("bf16_tn", mx.bfloat16, True, False),
    ("f32_nn", mx.float32, False, False),
    ("f32_nt", mx.float32, False, True),
    ("f32_tn", mx.float32, True, False),
]
RAW = {mx.float16: mx.uint16, mx.bfloat16: mx.uint16, mx.float32: mx.uint32}

info = mx.device_info()
print(json.dumps({"k": "host", "mlx": mx.__version__, "device": info.get("device_name"),
                  "bf16_8": info.get("cooperative_matrix_bf16_8")}), flush=True)
rng = np.random.default_rng(20261007)
a_np = rng.standard_normal((N, N)).astype(np.float32)
b_np = rng.standard_normal((N, N)).astype(np.float32)
rows = np.sort(rng.choice(N, 64, replace=False))
for name, dt, a_t, b_t in CELLS:
    # Stored a is [k, m] when a_t and b is [n, k] when b_t; the logical product is m x n.
    a = mx.array(a_np).astype(dt)
    b = mx.array(b_np).astype(dt)

    def run(a=a, b=b, a_t=a_t, b_t=b_t):
        return (a.T if a_t else a) @ (b.T if b_t else b)

    for _ in range(2):
        mx.eval(run())
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = run()
        mx.eval(out)
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times) // 2]
    raw = np.array(out.view(RAW[dt]))
    a64 = np.array(a.astype(mx.float32)).astype(np.float64)
    b64 = np.array(b.astype(mx.float32)).astype(np.float64)
    lhs = a64.T[rows] if a_t else a64[rows]
    truth = lhs @ (b64.T if b_t else b64)
    got = np.array(out.astype(mx.float32)).astype(np.float64)[rows]
    print(json.dumps({"k": "cell", "cell": name,
                      "sha": hashlib.sha256(raw.tobytes()).hexdigest()[:16],
                      "median_ms": round(med * 1e3, 3),
                      "tflops": round(2.0 * N ** 3 / med / 1e12, 3),
                      "max_abs_err_fp64": float(np.abs(got - truth).max())}), flush=True)
