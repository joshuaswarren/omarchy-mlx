"""Dump or compare bf16 4-bit/group-64 prefill quantized_matmul outputs.

usage: qmm_fma_bits.py dump|compare <file.npz>
Multi-row bf16 x against transposed 4-bit/g64 weights (the prefill route a
driver without the cooperative matrix sends to the tile kernel or, with the
bf16 scalar-FMA build, to QmmPrefillFma{,Precise}BF16). Prints each case's
max |GPU - float64 reference|, also in bf16 ulps at the case's largest
|reference|, and the mean error; on
compare, the cases that are not bit-identical to the dump.
"""
import sys
import time

import prov

prov.check()
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

mx.random.seed(5)
cases = []
for m in (5, 64, 77, 130, 512, 1024, 1500):
    for n, k in ((1000, 2048), (2050, 1024), (3072, 2048)):
        cases.append((m, n, k))
out, worst = {}, 0.0
for i, (m, n, k) in enumerate(cases):
    w = mx.random.normal((n, k)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    x = mx.random.normal((m, k)).astype(mx.bfloat16)
    y = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
    mx.eval(y)
    t0 = time.perf_counter()
    for _ in range(5):
        mx.eval(mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4))
    gflops = 2 * m * n * k * 5 / (time.perf_counter() - t0) / 1e9
    out[f"c{i}"] = np.array(y.view(mx.uint16))
    deq = np.array(mx.dequantize(wq, s, b, group_size=64, bits=4).astype(mx.float32)).astype(np.float64)
    ref = np.array(x.astype(mx.float32)).astype(np.float64) @ deq.T
    got = np.array(y.astype(mx.float32)).astype(np.float64)
    err = np.abs(got - ref)
    big = float(np.max(np.abs(ref)))
    ulp_big = 2.0 ** (np.floor(np.log2(big)) - 7)
    worst = max(worst, float(err.max()) / ulp_big)
    print(f"m={m:5d} n={n:5d} k={k:5d} max|ref|={big:8.2f} max_err={float(err.max()):.5f} "
          f"= {float(err.max()) / ulp_big:.3f} bf16 ulps at max|ref| "
          f"mean_err={float(err.mean()):.5f} gflops={gflops:.0f}", flush=True)
print(f"worst max_err in bf16 ulps at the case max|ref|: {worst:.3f}")
mode, path = sys.argv[1], sys.argv[2]
if mode == "dump":
    np.savez(path, **out)
    print(f"dumped {len(out)} cases")
else:
    ref = np.load(path)
    bad = [c for c in out if not np.array_equal(out[c], ref[c])]
    for c in bad:
        i = int(c[1:])
        print(f"DIFFERS {cases[i]}: {int((out[c] != ref[c]).sum())} of {out[c].size} values")
    print(f"{len(out) - len(bad)}/{len(out)} cases bit-identical")
    if bad or set(ref.files) != set(out):
        print("COMPARE FAILED", flush=True)
        sys.exit(1)
