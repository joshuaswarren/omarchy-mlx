"""Maskless prefill A/B on one chip: is the default MASKLESS route unchanged by the masked-recur32 change (the shader's hot loop gained
a mask branch)? Run once per wheel (qgpr = control, qrm = change) in a fresh process; mirrored by the ticket (qgpr qrm qrm qgpr).
Pre-registered rule (written before the wheel was built). Per T in (128, 512): fixed pseudo-random bf16 q/k/v/beta, f32 g, zero and
non-zero f32 initial state, B=1, Hk=Hv=32, Dk=Dv=128, maskless, through mx.fast.gated_delta_update; the output and final state are
hashed (sha256 of the raw bytes) and the kernel call is timed (median of 7 timed calls after 3 warm-ups, each call one submit of
a few ms).
PASS iff, for every T and both initial states, the output hash and the state hash of qrm equal those of qgpr (bit-identical) AND
the median time of qrm is within max(2%, the spread (max-min)/median of the two qgpr medians) of the qgpr median.
Output: one MLCHECK json line per process; compare with mlcheck_decide.py."""
import hashlib
import json
import statistics
import sys
import time

import mlx.core as mx

H, D = 32, 128


def rnd(shape, seed, scale, shift=0.0, dtype=mx.bfloat16):
    mx.random.seed(seed)
    return (mx.random.uniform(shape=shape, low=-1.0, high=1.0) * scale + shift).astype(dtype)


def digest(x):
    x32 = x.astype(mx.float32)
    mx.eval(x32)
    return hashlib.sha256(memoryview(x32)).hexdigest()[:16]


out = {"mlx": mx.__version__, "cases": {}}
for T in (128, 512):
    q, k, v = (rnd((1, T, H, D), s, 0.25) for s in (1, 2, 3))
    beta = rnd((1, T, H), 4, 0.2, 0.3)
    g = rnd((1, T, H), 5, 0.07, 0.92, mx.float32).__abs__()
    for name, h0 in (("zero", mx.zeros((1, H, D, D), mx.float32)), ("nz", rnd((1, H, D, D), 6, 0.1, 0.0, mx.float32))):
        mx.eval(q, k, v, beta, g, h0)
        times = []
        for i in range(10):
            mx.synchronize()
            t0 = time.perf_counter()
            y, hf = mx.fast.gated_delta_update(q, k, v, g, beta, h0)
            mx.eval(y, hf)
            mx.synchronize()
            if i >= 3:
                times.append(time.perf_counter() - t0)
        out["cases"][f"T{T}-{name}"] = {"y": digest(y), "state": digest(hf), "median_ms": round(statistics.median(times) * 1e3, 3)}
print("MLCHECK " + json.dumps(out), flush=True)
