"""Dump or compare decode SDPA outputs bit for bit, and time decode SDPA.

usage: sdpa_bits.py dump|compare <file.npz>
Decode shapes: q (1, H, 1, D), k/v (1, KV, L, D), bf16 and f16, D 64/128/256,
L from 1 to 600; plus the bf16 rows arm (4 causal query rows), batch 2 and a
7100-key cache, with f16 controls. compare exits 1 on any differing or
missing case, before the timing. The timing is host wall time per call
(200 calls of SDPA plus the query scaling in one eval), not a shader
timestamp; the dispatch trace gives shader times.
"""
import sys
import time

import prov

prov.check()
import mlx.core as mx
import numpy as np

mx.random.seed(11)
cases = []
for dt in (mx.bfloat16, mx.float16):
    for h, kv, d in ((16, 4, 128), (8, 2, 256), (32, 8, 64)):
        for L in (1, 7, 33, 100, 600):
            cases.append((dt, h, kv, d, L))
out = {}
# The bf16 arm's other shapes: four causal query rows (the rows arm,
# head 128), batch 2 (one token), and a 7100-key cache near the arm's
# 7168-key stream; f16 controls at the same shapes where they apply.
extra = [(mx.bfloat16, 16, 4, 128, 300, 1, 4, "causal"),
         (mx.bfloat16, 16, 4, 128, 300, 2, 1, None),
         (mx.bfloat16, 16, 4, 128, 7100, 1, 1, None),
         (mx.float16, 16, 4, 128, 300, 2, 1, None),
         (mx.float16, 16, 4, 128, 7100, 1, 1, None)]
cases = [(dt, h, kv, d, L, 1, 1, None) for (dt, h, kv, d, L) in cases] + extra
for i, (dt, h, kv, d, L, b, t, mask) in enumerate(cases):
    q = mx.random.normal((b, h, t, d)).astype(dt)
    k = mx.random.normal((b, kv, L, d)).astype(dt)
    v = mx.random.normal((b, kv, L, d)).astype(dt)
    y = mx.fast.scaled_dot_product_attention(q, k, v, scale=d ** -0.5, mask=mask)
    mx.eval(y)
    out[f"c{i}"] = np.array(y.view(mx.uint16))

mode, path = sys.argv[1], sys.argv[2]
if mode == "dump":
    np.savez(path, **out)
    print(f"dumped {len(out)} cases")
else:
    ref = np.load(path)
    bad = [k for k in out if not np.array_equal(out[k], ref[k])]
    for k in bad[:8]:
        print("MISMATCH", cases[int(k[1:])])
    print(f"{len(out) - len(bad)}/{len(out)} cases bit-identical")
    if bad or set(ref.files) != set(out):
        print("COMPARE FAILED", flush=True)
        sys.exit(1)

for dt in (mx.bfloat16, mx.float16):
    for L in (40, 400):
        q = mx.random.normal((1, 16, 1, 128)).astype(dt)
        k = mx.random.normal((1, 4, L, 128)).astype(dt)
        v = mx.random.normal((1, 4, L, 128)).astype(dt)
        mx.eval(q, k, v)
        run = lambda: mx.eval([mx.fast.scaled_dot_product_attention(q * (1.0 + i * 1e-3), k, v, scale=128 ** -0.5)
                               for i in range(200)])
        run()
        t0 = time.perf_counter(); run(); t = time.perf_counter() - t0
        print(f"{str(dt):14s} L={L:4d} sdpa+scale wall {t / 200 * 1e6:6.1f} us")
