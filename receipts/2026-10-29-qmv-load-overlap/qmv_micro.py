"""4-bit group-64 quantized matmul at ONE row, the decode shapes of the Qwen3.8-2B family, plus an exactness fingerprint.

Per shape: 16 dependent calls in one eval (x feeds the next call's input through a cheap add so the calls cannot overlap or be elided),
median of 9 evals, microseconds per call. Fingerprint: sha256 over the bf16 bytes of the outputs for 64 fixed-seed RANDOM activation rows
(one call per row) against the same random quantized weight, so two builds can be compared bit for bit through the device compiler.
Prints the loaded libmlx digests. Usage: qmv_micro.py [label]
"""
import glob
import hashlib
import importlib.metadata as md
import os
import statistics
import sys
import time

import mlx.core as mx

SHAPES = [("up/gate K2048 N6144", 2048, 6144), ("down K6144 N2048", 6144, 2048), ("proj K2048 N2048", 2048, 2048),
          ("kv K2048 N512", 2048, 512), ("head K2048 N248320", 2048, 248320)]
CHAIN = 16
EVALS = 9
FP_ROWS = 64


def libs():
    base = os.path.dirname(mx.__file__)
    return {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
            for p in sorted(glob.glob(os.path.join(base, "lib", "libmlx*"))) if os.path.isfile(p)}


label = sys.argv[1] if len(sys.argv) > 1 else "arm"
print("MICRO", label, "mlx_omarchy", md.version("mlx_omarchy"), libs(), flush=True)
for name, K, N in SHAPES:
    mx.random.seed(11)
    w0 = mx.random.normal((N, K)).astype(mx.bfloat16)
    w, s, b = mx.quantize(w0, group_size=64, bits=4)
    mx.eval(w, s, b)
    x = mx.random.normal((1, 1, K)).astype(mx.bfloat16)
    mx.eval(x)
    times = []
    for _ in range(EVALS):
        t0 = time.perf_counter()
        y = x
        for _ in range(CHAIN):
            y = mx.quantized_matmul(y, w, s, b, transpose=True, group_size=64, bits=4)
            y = x + y[..., :K] if N >= K else x + mx.concatenate([y] * (K // N), axis=-1)
        mx.eval(y)
        mx.synchronize()
        times.append((time.perf_counter() - t0) / CHAIN * 1e6)
    h = hashlib.sha256()
    mx.random.seed(23)
    for _ in range(FP_ROWS):
        xr = mx.random.normal((1, 1, K)).astype(mx.bfloat16)
        out = mx.quantized_matmul(xr, w, s, b, transpose=True, group_size=64, bits=4)
        mx.eval(out)
        h.update(repr(out.view(mx.uint16).tolist()).encode())
    print(f"MICRO {label} {name}: median {statistics.median(times):.1f} us/call (min {min(times):.1f}) fp {h.hexdigest()[:16]}", flush=True)
