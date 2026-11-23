#!/usr/bin/env python3
# G412: isolated small-M qmm prefill timing + cross-arm identity, one arm per
# process. The tile64 receipt's method at the skinny-M shapes the M4 Pro kit
# workload lands on (prompt_tokens 23). The two arms are forced through the
# occupancy-floor selector, because MLX_OMARCHY_QMM_COOPMAT_ROWS never moves
# the tile pick at M in 17..127 (it only gates the 64-row route at M >= 128):
#   env -u MLX_OMARCHY_GPU_CORES python3 small_m_sweep.py        # ctl: shipped pick
#   MLX_OMARCHY_GPU_CORES=100000 python3 small_m_sweep.py        # m16: floor forces the 16-row twin
# The two outputs must agree on every digest (bit-identity gate) and differ
# only in us (the timing the default flip is decided from). The canary row
# (M=520, N=4096) must time >= 1.3x apart: it proves the two arms really
# dispatched different kernels (the tile geometry predicts 1.5-2x; a tie
# within ~5% means the env did not reach the selector and voids the run).
import hashlib
import os
import time

import mlx.core as mx
import numpy as np

mx.random.seed(0)

cores = os.environ.get("MLX_OMARCHY_GPU_CORES", "")
ARM = "ctl32" if cores == "" else f"cores{cores}"
SHAPES_M = [17, 20, 23, 26, 31]
SHAPES_NK = [(4096, 4096), (12288, 4096), (1024, 4096)]
# The canary: large M where the ctl arm keeps 32 rows and the m16 arm's
# forced floor takes 16, so the two kernels are visibly different.
CANARY = (520, 4096, 4096)


def bench(m, n, k):
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    del w
    x = mx.random.normal((1, m, k)).astype(mx.bfloat16)
    f = lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
    for _ in range(2):
        mx.eval(f())
    out = f()
    mx.eval(out)
    digest = hashlib.sha256(np.asarray(out.astype(mx.float32)).tobytes()).hexdigest()[:16]
    best = 1e9
    for rep in range(5):
        t = time.perf_counter()
        for _ in range(8):
            mx.eval(f())
        best = min(best, (time.perf_counter() - t) / 8 * 1e6)
    print(f"arm={ARM} N={n} K={k} M={m} us={best:.1f} sha={digest}", flush=True)


for (n, k) in SHAPES_NK:
    for m in SHAPES_M:
        bench(m, n, k)
bench(*CANARY)
