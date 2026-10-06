"""MatmulGap H2 MLX-level check: f16 matmuls through the installed wheel, with
float64 numpy truth and an output hash per case. Run once plain (direct route)
and once with MLX_OMARCHY_NO_COOPMAT=1 (register-blocked route); equal hashes
mean identical stored bits. usage: python mlxcheck.py"""
import hashlib
import json
import os

import numpy as np
import mlx.core as mx

CASES = [
    ("nn_4096", lambda a, b: a @ b, (4096, 4096), (4096, 4096)),
    ("nt_512x4096x4096", lambda a, b: a @ b.T, (512, 4096), (4096, 4096)),
    ("tn_256x1024x512", lambda a, b: a.T @ b, (512, 256), (512, 1024)),
    ("tt_96x200x64", lambda a, b: a.T @ b.T, (64, 96), (200, 64)),
    ("batched_nt", lambda a, b: a @ b.swapaxes(-1, -2), (3, 4, 130, 72), (3, 4, 200, 72)),
    ("bcast_nn_odd_m", lambda a, b: a @ b, (2, 97, 520), (520, 1000)),
    ("slice_offset", lambda a, b: a[:, 2:66] @ b[2:66, :], (64, 80), (80, 64)),
]

rng = np.random.default_rng(1234)
res = {}
for name, fn, sa, sb in CASES:
    a = rng.standard_normal(sa).astype(np.float16)
    b = rng.standard_normal(sb).astype(np.float16)
    out = fn(mx.array(a), mx.array(b))
    mx.eval(out)
    got = np.array(out)
    truth = fn(a.astype(np.float64), b.astype(np.float64))
    exact = float(np.mean(got == truth.astype(np.float32).astype(np.float16)))
    err = np.abs(got.astype(np.float64) - truth)
    res[name] = {"sha": hashlib.sha256(got.tobytes()).hexdigest()[:16],
                 "exact_rounded": round(exact, 5),
                 "mean_abs_err": float(err.mean()), "max_abs_err": float(err.max())}
print(json.dumps({"no_coopmat": os.environ.get("MLX_OMARCHY_NO_COOPMAT", ""),
                  "mlx": mx.__version__, "cases": res}))
