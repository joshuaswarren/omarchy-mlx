"""MatmulGap H3 MLX-level check: matmuls of every dense dtype through the
installed wheel, float64 numpy truth on the exact operand values, and an output
hash per case. Run plain and with MLX_OMARCHY_NO_COOPMAT=1; equal hashes mean
identical stored bits. Then times a 4096^3 a @ b per dtype for 10 s.
usage: python mlxcheck2.py"""
import hashlib
import json
import os
import time

import mlx.core as mx
import numpy as np

CASES = [
    ("nn_1024", lambda a, b: a @ b, (1024, 1024), (1024, 1024)),
    ("nt_512x1024x768", lambda a, b: a @ b.T, (512, 768), (1024, 768)),
    ("tn_256x520x512", lambda a, b: a.T @ b, (512, 256), (512, 520)),
    ("tt_130x202x64", lambda a, b: a.T @ b.T, (64, 130), (202, 64)),
    ("batched_nt", lambda a, b: a @ b.swapaxes(-1, -2), (3, 4, 130, 72), (3, 4, 202, 72)),
    ("bcast_nn_odd_m", lambda a, b: a @ b, (2, 97, 520), (520, 1000)),
    ("slice_offset", lambda a, b: a[:, 2:66] @ b[2:66, :], (160, 80), (80, 34)),
]
DTYPES = {"float16": mx.float16, "bfloat16": mx.bfloat16, "float32": mx.float32}

rng = np.random.default_rng(1234)
res = {}
for dname, dt in DTYPES.items():
    for name, fn, sa, sb in CASES:
        a = mx.array(rng.standard_normal(sa).astype(np.float32)).astype(dt)
        b = mx.array(rng.standard_normal(sb).astype(np.float32)).astype(dt)
        out = fn(a, b)
        mx.eval(out)
        got = np.array(out.astype(mx.float32)).astype(np.float64)
        truth = fn(np.array(a.astype(mx.float32)).astype(np.float64),
                   np.array(b.astype(mx.float32)).astype(np.float64))
        rounded = np.array(mx.array(truth.astype(np.float32)).astype(dt).astype(mx.float32))
        raw = np.array(out.view(mx.uint16) if dt != mx.float32 else out.view(mx.uint32))
        err = np.abs(got - truth)
        res[f"{dname}/{name}"] = {
            "sha": hashlib.sha256(raw.tobytes()).hexdigest()[:16],
            "exact_rounded": round(float(np.mean(got == rounded)), 5),
            "mean_abs_err": float(err.mean()), "max_abs_err": float(err.max())}
timing = {}
FL = 2 * 4096 ** 3
for dname, dt in DTYPES.items():
    a = mx.random.normal((4096, 4096)).astype(dt)
    b = mx.random.normal((4096, 4096)).astype(dt)
    mx.eval(a, b)
    mx.eval(a @ b)
    t0 = time.time()
    n = 0
    while time.time() - t0 < 10:
        mx.eval(a @ b)
        n += 1
    timing[dname] = round(n * FL / (time.time() - t0) / 1e12, 3)
print(json.dumps({"no_coopmat": os.environ.get("MLX_OMARCHY_NO_COOPMAT", ""),
                  "mlx": mx.__version__, "tflops_4096": timing, "cases": res}))
