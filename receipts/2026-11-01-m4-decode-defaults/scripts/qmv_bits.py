"""Dump or compare decode quantized_matmul outputs bit for bit.

usage: qmv_bits.py dump <file.npz> | compare <file.npz>
compare exits 1 on any differing or missing case.
Covers 4-bit g64/g32, 8-bit g64/g32, bf16 and f16 activations, 1-4 rows,
shapes with K % 512 == 0 and not, N % 8 == 0 and not.
"""
import sys

import prov

prov.check()
import mlx.core as mx
import numpy as np

mx.random.seed(7)
CASES = []
for bits, gs in ((4, 64), (4, 32), (8, 64), (8, 32)):
    for dt in (mx.bfloat16, mx.float16):
        for m in (1, 2, 4):
            for n, k in ((2048, 2048), (512, 4096), (1000, 1536), (8192, 2048), (100, 768)):
                CASES.append((bits, gs, dt, m, n, k))

out = {}
for i, (bits, gs, dt, m, n, k) in enumerate(CASES):
    w = mx.random.normal((n, k)).astype(dt)
    q = mx.quantize(w, group_size=gs, bits=bits)
    x = mx.random.normal((m, k)).astype(dt)
    y = mx.quantized_matmul(x, *q, transpose=True, group_size=gs, bits=bits)
    mx.eval(y)
    out[f"c{i}"] = np.array(y.view(mx.uint16))

mode, path = sys.argv[1], sys.argv[2]
if mode == "dump":
    np.savez(path, **out)
    print(f"dumped {len(out)} cases")
else:
    ref = np.load(path)
    bad = [k for k in out if not np.array_equal(out[k], ref[k])]
    for k in bad[:10]:
        i = int(k[1:])
        diff = int((out[k] != ref[k]).sum())
        print(f"MISMATCH {CASES[i][:2]} {CASES[i][2]} m={CASES[i][3]} n={CASES[i][4]} k={CASES[i][5]}: {diff} values")
    print(f"{len(out) - len(bad)}/{len(out)} cases bit-identical")
    if bad or set(ref.files) != set(out):
        print("COMPARE FAILED", flush=True)
        sys.exit(1)
