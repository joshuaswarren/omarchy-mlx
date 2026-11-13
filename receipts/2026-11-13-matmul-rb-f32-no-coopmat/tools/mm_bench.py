"""Dense matmul timing, collector protocol (2 warm-ups, 8 timed a @ b with eval + synchronize, median), plus the
sha256 of every result's bytes so two arms can be compared bit for bit. usage: mm_bench.py <label> <out.json>"""
import hashlib, json, os, statistics, sys, time
import mlx.core as mx
res = {"label": sys.argv[1], "device": str(mx.default_device()), "rows": [],
       "env": {k: v for k, v in os.environ.items() if k.startswith("MLX_OMARCHY")}}
sync = getattr(mx, "synchronize", None)
mx.random.seed(20261113)
for dt in (mx.float32,):
    for n, at, bt in ((256, 0, 0), (512, 0, 0), (1024, 0, 0), (2048, 0, 0), (4096, 0, 0),
                      (1024, 0, 1), (1024, 1, 0), (2048, 0, 1)):
        a = mx.random.normal((n, n)).astype(dt)
        b = mx.random.normal((n, n)).astype(dt)
        mx.eval(a, b)
        A = a.T if at else a
        B = b.T if bt else b
        for _ in range(2):
            c = A @ B; mx.eval(c)
            if sync: sync()
        ts = []
        for _ in range(8):
            t0 = time.perf_counter(); c = A @ B; mx.eval(c)
            if sync: sync()
            ts.append((time.perf_counter() - t0) * 1000)
        med = statistics.median(ts)
        h = hashlib.sha256(bytes(memoryview(c))).hexdigest()
        # host reference on a 64x64 corner (float64)
        import numpy as np
        An = np.array(A[:64].astype(mx.float32), dtype=np.float64); Bn = np.array(B[:, :64].astype(mx.float32), dtype=np.float64)
        err = float(np.max(np.abs(np.array(c[:64, :64].astype(mx.float32), dtype=np.float64) - An @ Bn)))
        row = {"dtype": str(dt), "n": n, "aT": at, "bT": bt, "median_ms": round(med, 3), "min_ms": round(min(ts), 3),
               "max_ms": round(max(ts), 3), "tflops": round(2 * n ** 3 / (med / 1000) / 1e12, 4), "sha256": h,
               "max_abs_err_corner": err}
        res["rows"].append(row)
        print(f"{sys.argv[1]} {row['dtype']} n={n} aT={at} bT={bt} {row['median_ms']} ms {row['tflops']} TFLOP/s err {err:.2e} {h[:12]}", flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
