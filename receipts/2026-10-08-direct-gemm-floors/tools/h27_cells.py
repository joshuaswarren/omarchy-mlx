"""MatmulGap H26/H27: matmul at MLX level over dtype x orientation x m (n = k = 4096), one product per eval, hash per cell.

usage: h27_cells.py <arm> <chip: g13g|g14c>   chip selects the m list so each arm run stays short.
"""
import hashlib
import json
import sys
import time

import mlx.core as mx
import numpy as np

arm, chip = sys.argv[1], sys.argv[2]
MS = {"g13g": (128, 256, 512, 1024, 2048, 4096), "g13c": (512, 1024, 2048, 4096), "g14c": (512, 1024, 2048, 4096)}[chip]
DT = {"f16": mx.float16, "bf16": mx.bfloat16, "f32": mx.float32}
print(json.dumps({"k": "host", "arm": arm, "mlx": mx.__version__, "device": mx.device_info().get("device_name")}), flush=True)
for name, dt in DT.items():
    if chip == "g14c" and name == "f32":
        continue
    for ori in ("nt", "nn", "tn"):
        for m in MS:
            n = k = 4096
            rng = np.random.RandomState(int(hashlib.sha256(f"{name}:{ori}:{m}".encode()).hexdigest()[:8], 16))
            a = mx.array(rng.standard_normal((k, m) if ori == "tn" else (m, k)).astype(np.float32)).astype(dt)
            b = mx.array(rng.standard_normal((n, k) if ori == "nt" else (k, n)).astype(np.float32)).astype(dt)
            mx.eval(a, b)
            aa = a.T if ori == "tn" else a
            bb = b.T if ori == "nt" else b
            c = aa @ bb
            mx.eval(c)
            sha = hashlib.sha256(np.array(c.astype(mx.float32)).tobytes()).hexdigest()[:16]
            for _ in range(3):
                mx.eval(aa @ bb)
            times = []
            for _ in range(18):
                start = time.perf_counter()
                mx.eval(aa @ bb)
                times.append(time.perf_counter() - start)
            times.sort()
            med = times[len(times) // 2]
            print(json.dumps({"k": "cell", "arm": arm, "dtype": name, "ori": ori, "m": m, "median_us": round(med * 1e6, 1),
                              "tflops": round(2 * m * n * k / med / 1e12, 3), "sha": sha}), flush=True)
