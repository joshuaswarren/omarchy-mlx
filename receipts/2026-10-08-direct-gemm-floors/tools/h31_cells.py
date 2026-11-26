"""MatmulGap H31: f16 a @ b.T at m = 512..4096 (n = k = 4096) at MLX level, one product per eval, hash per cell."""
import hashlib
import json
import sys
import time

import mlx.core as mx
import numpy as np

arm = sys.argv[1]
print(json.dumps({"k": "host", "arm": arm, "mlx": mx.__version__, "device": mx.device_info().get("device_name")}), flush=True)
for m in (512, 1024, 2048, 4096):
    n = k = 4096
    rng = np.random.RandomState(int(hashlib.sha256(f"h31:f16:nt:{m}".encode()).hexdigest()[:8], 16))
    a = mx.array(rng.standard_normal((m, k)).astype(np.float32)).astype(mx.float16)
    b = mx.array(rng.standard_normal((n, k)).astype(np.float32)).astype(mx.float16)
    mx.eval(a, b)
    c = a @ b.T
    mx.eval(c)
    sha = hashlib.sha256(np.array(c.astype(mx.float32)).tobytes()).hexdigest()[:16]
    for _ in range(3):
        mx.eval(a @ b.T)
    times = []
    for _ in range(30):
        start = time.perf_counter()
        mx.eval(a @ b.T)
        times.append(time.perf_counter() - start)
    times.sort()
    med = times[len(times) // 2]
    print(json.dumps({"k": "cell", "arm": arm, "m": m, "median_us": round(med * 1e6, 1),
                      "tflops": round(2 * m * n * k / med / 1e12, 3), "sha": sha}), flush=True)
