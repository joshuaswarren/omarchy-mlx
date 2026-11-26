"""#317 diagnostic (not part of the gate): does the bf16 linear route run at the gate's
128-row shape, and are its bits equal to the cooperative-matrix kernel's?
Prints one JSON line per Llama-3.2-1B linear shape: sha256 of the bf16 output bytes and the
median ms of 20 timed calls. Run on p2 with and without MLX_OMARCHY_NO_MATMUL_FMA=1, and on main."""
import hashlib, json, os, sys, time
import prov
prov.check()
import mlx.core as mx
import numpy as np

rng = np.random.default_rng(317)
for m in (128, 1):
    for (n, k) in ((2048, 2048), (512, 2048), (8192, 2048), (2048, 8192)):
        x = mx.array(rng.standard_normal((1, m, k)).astype(np.float32)).astype(mx.bfloat16)
        w = mx.array((rng.standard_normal((n, k)) / np.sqrt(k)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(x, w)
        y = x @ w.T
        mx.eval(y)
        h = hashlib.sha256(np.array(y.astype(mx.float32)).tobytes()).hexdigest()[:16]
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); mx.eval(x @ w.T); ts.append((time.perf_counter() - t0) * 1e3)
        print(json.dumps({"m": m, "n": n, "k": k, "sha": h, "ms_median": round(sorted(ts)[10], 4),
                          "no_matmul_fma": os.environ.get("MLX_OMARCHY_NO_MATMUL_FMA", "")}), flush=True)
