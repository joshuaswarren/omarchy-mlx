"""Hardware version of the qmv batch doctest: does the batch-shared kernel give the same bits as the per-row kernel, on this wheel?

For each cell (K, N) and batch B in 2, 3, 4 and 12 seeded datasets: x [B, 1, K] bf16 against a 4-bit group-64 weight [N, K];
want = MLX_OMARCHY_QMV_BATCH=0 (one single-row kernel per row), got = MLX_OMARCHY_QMV_BATCH=1 (the batch-shared kernel).
Cells include the 512-wide path (K % 512 == 0, N % 8 == 0) and the narrow path (K = 448, N = 130).
Prints per cell the number of datasets whose outputs differ, and a sha256 over every `got` output so that two wheels can be compared byte for byte.
"""
import glob
import hashlib
import importlib.metadata as md
import os

import mlx.core as mx

CELLS = [(2048, 6144), (6144, 2048), (2048, 520), (448, 130)]
DATASETS = 12


def libs():
    base = os.path.dirname(mx.__file__)
    return {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
            for p in sorted(glob.glob(os.path.join(base, "lib", "libmlx*"))) if os.path.isfile(p)}


def run(x, w, s, b, flag):
    os.environ["MLX_OMARCHY_QMV_BATCH"] = flag
    y = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
    mx.eval(y)
    mx.synchronize()
    os.environ.pop("MLX_OMARCHY_QMV_BATCH", None)
    return y


print("PROVENANCE mlx_omarchy", md.version("mlx_omarchy"), libs(), "device", mx.default_device(), flush=True)
total = hashlib.sha256()
bad_total = 0
for k, n in CELLS:
    for rows in (2, 3, 4):
        bad = 0
        h = hashlib.sha256()
        for seed in range(DATASETS):
            mx.random.seed(1000003 * k + 101 * n + 17 * rows + seed)
            w0 = (mx.random.normal((n, k)) * 0.05).astype(mx.float32)
            w, s, b = mx.quantize(w0, group_size=64, bits=4)
            s, b = s.astype(mx.bfloat16), b.astype(mx.bfloat16)
            x = mx.random.normal((rows, 1, k)).astype(mx.bfloat16)
            mx.eval(w, s, b, x)
            want = run(x, w, s, b, "0")
            got = run(x, w, s, b, "1")
            if not mx.array_equal(want, got).item():
                bad += 1
            h.update(repr(got.view(mx.uint16).tolist()).encode())
        bad_total += bad
        total.update(h.digest())
        print(f"CELL K={k} N={n} B={rows}: datasets differing from the per-row kernel {bad} of {DATASETS}; sha256 of the batch outputs {h.hexdigest()[:16]}", flush=True)
print(f"SUMMARY differing datasets {bad_total}; combined sha256 {total.hexdigest()[:16]}", flush=True)
