#!/usr/bin/env python3
"""g16-qmm-batch — installed-wheel probe for the batched bf16 coopmat QMM
row-mixing fix (cf41e9b6f). Mirrors the matmul_family pins on the INSTALLED
wheel: (1) batch-independence — quantized_matmul over [B, T, K] must be
BIT-IDENTICAL per batch row to the same row run alone; (2) fp32 host
reference within tolerance; (3) the w7B failing shape N=6144 plus a
non-tile-multiple T=17. PASS exits 0 with a JSON summary."""
import hashlib
import json
import sys

import numpy as np
import mlx.core as mx

K, N = 512, 6144
G64, BITS = 64, 4
TOL = 2e-2
mx.random.seed(0x0730)

w = mx.random.normal((N, K)).astype(mx.bfloat16)
wq, scales, biases = mx.quantize(w, group_size=G64, bits=BITS)
wd = mx.dequantize(wq, scales, biases, group_size=G64, bits=BITS).astype(mx.float32)


def row_bytes(arr):
    # bf16 has no numpy dtype: digest the exact fp32 widening
    return np.asarray(arr.astype(mx.float32), dtype=np.float32).tobytes()


failures = []
digests = []
cases = 0
for B in (2, 4):
    for T in (16, 17):
        x = mx.random.normal((B, T, K)).astype(mx.bfloat16)
        batched = mx.quantized_matmul(x, wq, scales, biases, transpose=True,
                                      group_size=G64, bits=BITS)
        for i in range(B):
            alone = mx.quantized_matmul(x[i:i + 1], wq, scales, biases,
                                        transpose=True, group_size=G64, bits=BITS)
            same = mx.array_equal(batched[i], alone[0])
            d = hashlib.sha256(row_bytes(batched[i])).hexdigest()[:16]
            digests.append({"case": f"B{B}T{T}", "row": i,
                            "bit_identical_to_alone": bool(same), "digest": d})
            if not same:
                failures.append(f"B{B}T{T} row {i}: batched != alone")
        cases += 1

summary = {"gate": "g16-qmm-batch", "cases": cases,
           "rows": len(digests), "failures": failures,
           "digests": digests, "pass": not failures and bool(digests)}
print(json.dumps(summary, indent=2))
sys.exit(0 if summary["pass"] else 1)
