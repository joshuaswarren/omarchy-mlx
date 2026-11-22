#!/usr/bin/env python3
# G412: isolated small-M qmm prefill timing + cross-arm identity, one arm per
# process. The tile64 receipt's method at the skinny-M shapes the M4 Pro kit
# workload lands on (prompt_tokens 23). Run twice:
#   MLX_OMARCHY_QMM_COOPMAT_ROWS=32 python3 small_m_sweep.py
#   MLX_OMARCHY_QMM_COOPMAT_ROWS=16 python3 small_m_sweep.py
# The two outputs must agree on every digest (bit-identity gate) and differ
# only in ms (the timing the default flip is decided from).
import hashlib
import os
import time

import mlx.core as mx
import numpy as np

mx.random.seed(0)

ARMS = os.environ.get("MLX_OMARCHY_QMM_COOPMAT_ROWS", "unset")
SHAPES_M = [17, 20, 23, 26, 31]
SHAPES_NK = [(4096, 4096), (12288, 4096), (1024, 4096)]

for (n, k) in SHAPES_NK:
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    del w
    for m in SHAPES_M:
        x = mx.random.normal((1, m, k)).astype(mx.bfloat16)
        f = lambda: mx.quantized_matmul(
            x, wq, s, b, transpose=True, group_size=64, bits=4
        )
        for _ in range(2):
            mx.eval(f())
        out = f()
        mx.eval(out)
        digest = hashlib.sha256(
            np.asarray(out.astype(mx.float32)).tobytes()
        ).hexdigest()[:16]
        best = 1e9
        for rep in range(5):
            t = time.perf_counter()
            for _ in range(8):
                mx.eval(f())
            best = min(best, (time.perf_counter() - t) / 8 * 1e6)
        print(
            f"arm={ARMS} N={n} K={k} M={m} us={best:.1f} sha={digest}",
            flush=True,
        )
