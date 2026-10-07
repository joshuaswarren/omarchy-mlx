#!/usr/bin/env python3
"""g16b-qmm-route-probe — installed-wheel probe pinning each dispatched
coopmat build of the batched bf16 quantized_matmul route (companion to
g16-qmm-batch, which covers the M16 shape end to end). Covers both fixed
shader sites in qmm_coopmat.comp (qmm_tile X_F32 x_slice and qmm_tile_two):

  1. M16 FullN build (kernel QmmPrefillCoopmatM16BF16X32FullN):
     B in {2, 4}, m=16, N=256, K=2048.
  2. 32-row FullN build (kernel QmmPrefillCoopmatBF16X32FullN):
     B in {2, 4}, m=128, N=2048, K=2048 (ceil(m/32)*(N/32)=256 tile
     units, past the cores*6 threshold where coopmat_tile_rows picks 32).
  3. TwoN build (kernel QmmPrefillCoopmatBF16X32FullNTwoN, the
     qmm_tile_two site): B in {2, 4}, m=128, N=2048, K=2048.

Host-env knobs set per shape group (restored after): group 3 sets
MLX_OMARCHY_QMM_TWON=1 (the TwoN route requires it and N % 64 == 0);
groups 1-2 set nothing (landed default routing). Weights quantized with
mx.quantize (affine, group 64, 4 bits, transposed), reference from
mx.dequantize on CPU in fp64 numpy, distinct random rows per batch.

Contract (self-calibrating): for each shape the same x runs batched
[B, m, K] and flattened [B*m, K] (flat has batch=1 and never enters the
batch indexing). Exit nonzero if any batched row error exceeds 3x the
max flat error of the same shape; wrong-row defects miss by 100x+,
bf16 rounding noise stays under 1 ulp of the largest |y|. Prints a
final RESULT line.
"""
import os
import sys

import numpy as np

K = 2048
GS = 64
BITS = 4
CASES = [
    (2, 16, 256, None),
    (4, 16, 256, None),
    (2, 128, 2048, None),
    (4, 128, 2048, None),
    (2, 128, 2048, "MLX_OMARCHY_QMM_TWON=1"),
    (4, 128, 2048, "MLX_OMARCHY_QMM_TWON=1"),
]


def fixture(n):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(7)
    w = mx.array(rng.uniform(-2, 2, (n, K)).astype(np.float32))
    wq, s, b = mx.quantize(w, group_size=GS, bits=BITS)
    s = mx.astype(mx.astype(s, mx.bfloat16), mx.float32)
    b = mx.astype(mx.astype(b, mx.bfloat16), mx.float32)
    w_hat = mx.dequantize(wq, s, b, group_size=GS, bits=BITS)
    return (np.array(wq), np.array(s), np.array(b),
            np.asarray(w_hat, dtype=np.float64))


def row_errors(x_btk, wq, s, b, w_hat64):
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    wq_g = mx.array(wq)
    s_g = mx.astype(mx.array(s), mx.bfloat16)
    b_g = mx.astype(mx.array(b), mx.bfloat16)
    x = mx.array(x_btk.astype(np.float32)).astype(mx.bfloat16)
    B, m, _ = x_btk.shape
    n = w_hat64.shape[0]
    out = {}
    for layout in ("batched", "flat"):
        xf = mx.reshape(x, (B * m, K)) if layout == "flat" else x
        y = mx.quantized_matmul(xf, wq_g, s_g, b_g, transpose=True,
                                group_size=GS, bits=BITS)
        y = mx.reshape(y, (B, m, n))
        mx.eval(y)
        x64 = np.asarray(mx.astype(xf, mx.float32),
                         dtype=np.float64).reshape(B, m, K)
        ref = np.einsum("btk,nk->btn", x64, w_hat64)
        got = np.asarray(mx.astype(y, mx.float32), dtype=np.float64)
        out[layout] = np.abs(got - ref).max(axis=(1, 2))
    return out["batched"], out["flat"]


def main():
    import mlx.core as mx
    mx.set_default_device(mx.gpu)
    failures = []
    for (B, m, n, env) in CASES:
        old = None
        if env is not None:
            name, val = env.split("=")
            old = os.environ.get(name)
            os.environ[name] = val
        np_wq, np_s, np_b, w_hat64 = fixture(n)
        rng = np.random.default_rng(11 + B * 1000 + m)
        x_distinct = rng.uniform(-2, 2, (B, m, K)).astype(np.float32)
        batched, flat = row_errors(x_distinct, np_wq, np_s, np_b, w_hat64)
        if env is not None:
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        flat_max = float(flat.max())
        bad = [int(i) for i, e in enumerate(batched) if e > 3.0 * flat_max]
        status = "FAIL" if bad else "ok"
        print(f"B={B} m={m} N={n} {env or 'default'}: "
              f"batched={np.array2string(batched, precision=4)} "
              f"flat_max={flat_max:.4f} bad={bad} {status}",
              flush=True)
        if bad:
            failures.append(f"B{B} m{m} N{n} {env or 'default'} rows {bad}")
    if failures:
        print(f"RESULT: FAIL ({len(failures)} shape groups with bad rows)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("RESULT: PASS (all coopmat build variants batch-clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
