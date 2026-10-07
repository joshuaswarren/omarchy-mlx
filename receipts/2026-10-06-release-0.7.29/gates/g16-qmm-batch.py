#!/usr/bin/env python3
"""g16-qmm-batch — installed-wheel probe for the batched bf16 coopmat QMM
row-mixing fix (cf41e9b6f, the X_F32 batch-stride defect that shipped in
v0.7.29). Mirrors the matmul_family pins (d3ede9f4b) on the INSTALLED wheel:

  1. batch-independence: quantized_matmul over [B, T, K] must be BIT-IDENTICAL
     per batch row to the same row run alone (B in {2, 4}).
  2. host reference: every row within TOL of the fp32 host quantized matmul
     (dequantized weights).
  3. the w7B failing shape (N=6144, T=16, B in {2, 4}) and a non-tile-multiple
     T=17.

Run with the host's installed-wheel venv python. PASS exits 0 and prints a
JSON summary (digests + worst error).
"""
import hashlib
import json
import sys

import mlx.core as mx

K, N = 512, 6144
G64, BITS = 64, 4
TOL = 2e-2
mx.random.seed(0x0730)

w = mx.random.normal((N, K)).astype(mx.bfloat16)
wq, scales, biases = mx.quantize(w, group_size=G64, bits=BITS)
wd = mx.dequantize(wq, scales, biases, group_size=G64, bits=BITS).astype(mx.float32)

failures = []
digests = []
for B in (2, 4):
    for T in (16, 17):
        x = mx.random.normal((B, T, K)).astype(mx.bfloat16)
        batched = mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                                      group_size=G64, bits=BITS)
        host = (x.astype(mx.float32) @ wd.T)
        err = (batched.astype(mx.float32) - host).abs().max().item()
        tag = f"B{B}T{T}"
        for i in range(B):
            alone = mx.quantized_matmul(x[i:i + 1], wq, scales, biases,
                                        transpose=True, group_size=G64, bits=BITS)
            same = mx.array_equal(batched[i], alone[0])
            d = hashlib.sha256(np_bytes(batched[i])).hexdigest()[:16]
            digests.append({"case": tag, "row": i, "bit_identical_to_alone": bool(same),
                            "digest": d})
            if not same:
                failures.append(f"{tag} row {i}: batched != alone")
        if err > TOL:
            failures.append(f"{tag}: host err {err:.4f} > {TOL}")
        print(f"{tag}: host_err={err:.4f} rows_bit_identical=True", file=sys.stderr)

def np_bytes(arr):
    import numpy as np
    return np.asarray(arr, dtype=np.float16).tobytes()

worst_ok = not failures
print(json.dumps({"gate": "g16-qmm-batch", "cases": 2 * 2 * (4 + 4),
                  "failures": failures, "digests": digests,
                  "pass": bool(worst_ok and digests)}, indent=2))
sys.exit(0 if worst_ok and digests else 1)
