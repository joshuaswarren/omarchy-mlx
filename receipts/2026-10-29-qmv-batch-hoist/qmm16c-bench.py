"""4-bit group-64 quantized matmul at 1 to 16 rows: shipped tile route against the block GEMV route (shipped 4-row kernel, and hoisted 4-, 8- and 16-row kernels).

x [1, T, K] bf16 against a transposed 4-bit group-64 weight [N, K]. Modes, interleaved in one process, 3 rounds:
  tile : MLX_OMARCHY_QMV_SMALL_T unset (shipped route)
  b4   : MLX_OMARCHY_QMV_SMALL_T=16, MLX_OMARCHY_QMV_SMALL_T_BLOCK=4  (ceil(T/4) Batch4 dispatches)
  b4h  : ... BLOCK=5   (ceil(T/4) dispatches of the 4-row kernel with the nibble conversions hoisted out of the row loop)
  b8h  : ... BLOCK=8   (ceil(T/8) dispatches of the hoisted 8-row kernel; a remainder of 4 or fewer rows uses the hoisted 4-row kernel)
  b16h : ... BLOCK=16  (one dispatch of the hoisted 16-row kernel up to T=16)
Per call = own eval + synchronise, so a fixed floor is inside every number. Prints the median per mode, the ratio tile/mode, the largest
absolute difference of each block mode to the tile route, of the shipped 4-row kernel to each hoisted mode (expected 0: same values, same fma chain), and the
largest absolute output value, plus provenance (libmlx digests; compare with the wheel RECORD).
"""
import glob
import hashlib
import importlib.metadata as md
import os
import statistics
import time

import mlx.core as mx

ENV = "MLX_OMARCHY_QMV_SMALL_T"
ENV_BLOCK = "MLX_OMARCHY_QMV_SMALL_T_BLOCK"
CELLS = [("mlp 2048x6144", 2048, 6144), ("proj 2048x2048", 2048, 2048), ("lm 2048x16384", 2048, 16384)]
ROWS = [1, 2, 4, 6, 8, 10, 12, 14, 16]
MODES = [("tile", None, None), ("b4", "16", "4"), ("b4h", "16", "5"), ("b8h", "16", "8"), ("b16h", "16", "16")]
CALLS = 30


def libs():
    base = os.path.dirname(mx.__file__)
    return {os.path.basename(p): hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
            for p in sorted(glob.glob(os.path.join(base, "lib", "libmlx*"))) if os.path.isfile(p)}


def setmode(flag, block):
    for k, v in ((ENV, flag), (ENV_BLOCK, block)):
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def run(x, w, s, b):
    t = []
    for _ in range(CALLS):
        t0 = time.perf_counter()
        y = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
        mx.eval(y)
        mx.synchronize()
        t.append(time.perf_counter() - t0)
    return statistics.median(t) * 1e3, y


def diff(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())


print("PROVENANCE mlx_omarchy", md.version("mlx_omarchy"), libs(), "device", mx.default_device())
for name, K, N in CELLS:
    mx.random.seed(7)
    w0 = mx.random.normal((N, K)).astype(mx.bfloat16)
    w, s, b = mx.quantize(w0, group_size=64, bits=4)
    mx.eval(w, s, b)
    for T in ROWS:
        x = mx.random.normal((1, T, K)).astype(mx.bfloat16)
        mx.eval(x)
        times = {m: [] for m, _, _ in MODES}
        outs = {}
        for rnd in range(3):
            for m, flag, block in MODES:
                setmode(flag, block)
                ms, outs[m] = run(x, w, s, b)
                times[m].append(ms)
        setmode(None, None)
        med = {m: statistics.median(v) for m, v in times.items()}
        cells = "  ".join(f"{m} {med[m]:.3f}" for m, _, _ in MODES)
        ratios = "  ".join(f"tile/{m} {med['tile'] / med[m]:.2f}" for m, _, _ in MODES[1:])
        d_tile = "  ".join(f"|tile-{m}| {diff(outs['tile'], outs[m]):.4g}" for m, _, _ in MODES[1:])
        d_blk = "  ".join(f"|b4-{m}| {diff(outs['b4'], outs[m]):.4g}" for m in ("b4h", "b8h", "b16h"))
        peak = float(mx.max(mx.abs(outs["tile"].astype(mx.float32))).item())
        print(f"QMM {name} T={T}: ms {cells} | {ratios} | {d_tile} | {d_blk} | max|y| {peak:.4g}")
