"""MatmulGap H27 amendment 4: which kernels does one product run per cell (dispatch trace). No timing.

usage: route_trace.py <arm> <chip g13g|g13c|g14c>; stderr of this process (the trace) goes to a file chosen by the caller.
Writes a marker line '@@CELL dtype ori m' to stderr before and '@@END' after each product, so the caller can split the trace.
"""
import hashlib
import os
import sys

import mlx.core as mx
import numpy as np

arm, chip = sys.argv[1], sys.argv[2]
MS = {"g13g": (128, 256, 512, 1024, 2048, 4096), "g13c": (512, 1024, 2048, 4096), "g14c": (512, 1024, 2048, 4096)}[chip]
DT = {"f16": mx.float16, "bf16": mx.bfloat16, "f32": mx.float32}
sys.stderr.write(f"@@HOST {arm} {mx.__version__} {mx.device_info().get('device_name')}\n")
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
            os.environ["MLX_OMARCHY_TRACE_DISPATCH"] = "1"
            sys.stderr.write(f"@@CELL {name} {ori} {m}\n")
            sys.stderr.flush()
            mx.eval(aa @ bb)
            sys.stderr.write("@@END\n")
            sys.stderr.flush()
            del os.environ["MLX_OMARCHY_TRACE_DISPATCH"]
