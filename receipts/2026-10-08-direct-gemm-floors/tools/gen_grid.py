"""MatmulGap H34: speed and output hash of dense matmuls at LLM-like shapes, per chip grid, one product per eval.

usage: gen_grid.py <arm> <chip g13g|g14c>
Adaptive timing: evals = clamp(int(2.0 / first), 5, 18) after one warmup product; median of the timed evals.
"""
import hashlib
import json
import sys
import time

import mlx.core as mx
import numpy as np

arm, chip = sys.argv[1], sys.argv[2]
NS = {"g14c": (4096, 9728, 12288), "g13c": (4096, 9728, 12288), "g13g": (4096, 9728)}[chip]
KS = (2560, 11008)
# (dtype, orientation, m list)
GRID = {
    "g14c": [("bf16", "nt", (512, 2048, 8192)), ("f16", "nt", (512, 2048, 8192)), ("bf16", "tn", (8192,))],
    "g13g": [("f16", "nt", (512, 2048)), ("f16", "tn", (2048,)), ("bf16", "tn", (512, 2048)),
             ("f32", "nt", (512, 2048)), ("f32", "nn", (512, 2048)), ("f32", "tn", (512, 2048))],
    "g13c": [("bf16", "nt", (512, 2048, 8192)), ("bf16", "tn", (8192,)), ("f32", "nt", (8192,)),
             ("f16", "nt", (512, 2048, 8192)), ("f16", "tn", (8192,))],
}[chip]
DT = {"f16": mx.float16, "bf16": mx.bfloat16, "f32": mx.float32}
print(json.dumps({"k": "host", "arm": arm, "mlx": mx.__version__, "device": mx.device_info().get("device_name")}), flush=True)
for name, ori, ms in GRID:
    for m in ms:
        for n in NS:
            for k in KS:
                cid = f"{name}:{ori}:{m}x{n}x{k}"
                rng = np.random.default_rng(int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
                a = mx.array(rng.standard_normal((k, m) if ori == "tn" else (m, k), dtype=np.float32)).astype(DT[name])
                b = mx.array(rng.standard_normal((n, k) if ori == "nt" else (k, n), dtype=np.float32)).astype(DT[name])
                mx.eval(a, b)
                aa = a.T if ori == "tn" else a
                bb = b.T if ori == "nt" else b
                c = aa @ bb
                mx.eval(c)
                sha = hashlib.sha256(np.array(c.astype(mx.float32)).tobytes()).hexdigest()[:16]
                start = time.perf_counter()
                mx.eval(aa @ bb)
                first = time.perf_counter() - start
                evals = max(5, min(18, int(2.0 / max(first, 1e-4))))
                times = []
                for _ in range(evals):
                    start = time.perf_counter()
                    mx.eval(aa @ bb)
                    times.append(time.perf_counter() - start)
                times.sort()
                med = times[len(times) // 2]
                print(json.dumps({"k": "cell", "arm": arm, "dtype": name, "ori": ori, "m": m, "n": n, "kdim": k,
                                  "median_us": round(med * 1e6, 1), "evals": evals,
                                  "tflops": round(2 * m * n * k / med / 1e12, 3), "sha": sha}), flush=True)
