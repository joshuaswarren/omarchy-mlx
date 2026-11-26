"""Decode attention alone: how fast does one query token read a KV cache of length L on this chip?

q [1, H, 1, D] against k, v [1, Hkv, L, D] (bf16 first, the dtype of the measured models, then f16), through mx.fast.scaled_dot_product_attention, the call mlx-lm makes per full-attention
layer at decode. Reports time per call and the KV bytes read per second (K+V = 2 * Hkv * L * D * 2 bytes); compare with the copy
bandwidth printed first (the roof). Shapes are those of the three models measured on the 13-inch M1:
  9B  (qwen3_5, 8 full-attention layers): H 16, Hkv 4, D 256     2B (6 layers): H 8, Hkv 2, D 256     4B (Qwen3, 36 layers): H 32, Hkv 8, D 128
Each call runs in its own eval + synchronise (as a decode layer's dependency chain does), plus a chain of 8 independent calls on
distinct caches to show the per-call cost with submission overhead amortised. Usage: sdpa-decode-bench.py [L ...]
"""
import glob
import hashlib
import importlib.metadata as md
import json
import os
import statistics
import sys
import time

import mlx.core as mx

GIB = 1 << 30


def libs():
    base = os.path.dirname(mx.__file__)
    out = {}
    for p in sorted(glob.glob(os.path.join(base, "lib", "libmlx*"))):
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                out[os.path.basename(p)] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return out


print("PROVENANCE mlx-omarchy", md.version("mlx-omarchy"), "mlx file", mx.__file__, "SPLIT env", os.environ.get("MLX_OMARCHY_SDPA_DECODE_SPLIT"), "libs", json.dumps(libs()), flush=True)

big = mx.zeros((GIB // 4,), dtype=mx.float32)
mx.eval(big)
mx.synchronize()
bw = []
for _ in range(4):
    t = time.perf_counter()
    z = big * 1.0
    mx.eval(z)
    mx.synchronize()
    bw.append(2 * GIB / (time.perf_counter() - t) / 1e9)
print(f"ROOF elementwise copy bandwidth: {statistics.median(bw):.1f} GB/s", flush=True)
del big, z

lengths = [int(a) for a in sys.argv[1:]] or [512, 2048, 8192, 16384]
shapes = [("9B", 16, 4, 256), ("2B", 8, 2, 256), ("4B", 32, 8, 128)]
WANT = os.environ.get("SDPA_BENCH_DTYPES", "bf16,f16").split(",")
for dtype_name, dtype in [d for d in (("bf16", mx.bfloat16), ("f16", mx.float16)) if d[0] in WANT]:
  for name, h, hkv, d in shapes:
    for length in lengths:
        mx.random.seed(7)
        q = mx.random.normal((1, h, 1, d), dtype=dtype)
        caches = [
            (mx.random.normal((1, hkv, length, d), dtype=dtype), mx.random.normal((1, hkv, length, d), dtype=dtype))
            for _ in range(8)
        ]
        mx.eval(q, [c for kv in caches for c in kv])
        mx.synchronize()
        scale = d ** -0.5
        for k, v in caches[:2]:
            mx.eval(mx.fast.scaled_dot_product_attention(q, k, v, scale=scale))
        mx.synchronize()
        single = []
        for i in range(12):
            k, v = caches[i % 8]
            t = time.perf_counter()
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            mx.eval(o)
            mx.synchronize()
            single.append(time.perf_counter() - t)
        chain = []
        for _ in range(5):
            t = time.perf_counter()
            outs = [mx.fast.scaled_dot_product_attention(q, k, v, scale=scale) for k, v in caches]
            mx.eval(outs)
            mx.synchronize()
            chain.append((time.perf_counter() - t) / 8)
        kv_bytes = 2 * hkv * length * d * 2
        ms = statistics.median(single)
        mc = statistics.median(chain)
        print(
            f"SDPA {dtype_name} {name} H{h} Hkv{hkv} D{d} L{length}: single {ms * 1e6:.0f} us = {kv_bytes / ms / 1e9:.1f} GB/s; "
            f"8 chained {mc * 1e6:.0f} us each = {kv_bytes / mc / 1e9:.1f} GB/s (KV {kv_bytes / 1e6:.1f} MB)",
            flush=True,
        )
        del caches, q
print("DONE", flush=True)
