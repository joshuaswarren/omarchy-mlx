"""6-bit mx.quantized_matmul (the lm_head) against a float64 host reference, with dispatch ids and per-call time.

Muse head shape: N=202048 output rows (lm_head.weight 202048 x 1536 words), K=8192, group size 64, 6-bit affine, transposed; the sealed tensors
are bf16 scales and biases, so bf16 is the primary cell and f16 the second. --variants loops the streaming kernel variants (env
MLX_OMARCHY_QMM6_VARIANT 0..5; "d" = unset, the dispatch default). Rows (tokens)
1, 2, 8, 17: decode, the verify steps, and the step the T-curve found at 17. Weights are random normal, so the numbers show
numerics and cost, not model quality. The weights are quantized in row chunks (the float tensor would be 8 GB).
The reference decodes the raw packed words itself (6-bit codes are 4 per 3 bytes, little endian), checks that decode against
mx.dequantize on the CPU stream, then contracts in float64 on --cols sampled output rows.
f32: rel err (max abs / max ref) <= 1e-4. f16 and bf16: share of outputs beyond 1.5 steps of the output dtype = 0 and beyond 2.5 steps = 0.
Exit 0 only if every cell passes and none raised; 4 on a libmlx mismatch; 5 if the host decode disagrees with mx.dequantize.

usage: m2_qmm6.py [--n 202048] [--k 8192] [--gs 64] [--rows 1,2,8,17] [--reps 20] [--cols 256] [--chunk 6314]
                  [--cells bf16,f16] [--variants d] [--expect-libmlx HEX16]
"""
import argparse
import hashlib
import os
import re
import statistics
import sys
import tempfile
import time

import mlx.core as mx
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=202048)
ap.add_argument("--k", type=int, default=8192)
ap.add_argument("--gs", type=int, default=64)
ap.add_argument("--rows", default="1,2,8,17")
ap.add_argument("--reps", type=int, default=20)
ap.add_argument("--cols", type=int, default=256)
ap.add_argument("--chunk", type=int, default=6314)
ap.add_argument("--cells", default="bf16,f16")
ap.add_argument("--variants", default="d")
ap.add_argument("--expect-libmlx", default="")
a = ap.parse_args()
N, K, GS, BITS = a.n, a.k, a.gs, 6
assert N % a.chunk == 0 and K % GS == 0 and K % 4 == 0, "chunk must divide N; K must be a multiple of the group size and of 4"
rng = np.random.default_rng(23)
print("device", mx.default_device(), "mlx", mx.__version__)


def loaded(name):
    paths = set()
    with open("/proc/self/maps") as maps:
        for line in maps:
            parts = line.split(maxsplit=5)
            if len(parts) == 6 and name in parts[5]:
                paths.add(parts[5].strip().removesuffix(" (deleted)"))
    return {p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] for p in sorted(paths)}


libs = loaded("libmlx.so")
print("mlx file", mx.__file__, "| libmlx loaded", libs)
if a.expect_libmlx and set(libs.values()) != {a.expect_libmlx}:
    print(f"LIBMLX MISMATCH: wanted exactly {{{a.expect_libmlx}}}")
    sys.exit(4)


def traced(run):
    """Run run() with MLX_OMARCHY_TRACE_DISPATCH set and fd 2 sent to a file; return the dispatched kernel ids in order."""
    with tempfile.TemporaryFile("w+") as tmp:
        saved = os.dup(2)
        os.dup2(tmp.fileno(), 2)
        os.environ["MLX_OMARCHY_TRACE_DISPATCH"] = "1"
        try:
            run()
        finally:
            os.environ.pop("MLX_OMARCHY_TRACE_DISPATCH", None)
            os.dup2(saved, 2)
            os.close(saved)
        tmp.seek(0)
        text = tmp.read()
    return [int(m) for m in re.findall(r"DISPATCH kernel=(-?\d+)", text)]


def decode_codes(words):
    """uint32 words [rows, K*6/32] -> 6-bit codes [rows, K]: 4 codes per 3 bytes, little endian, low code first."""
    b = np.ascontiguousarray(words).view(np.uint8).reshape(words.shape[0], -1, 3).astype(np.uint32)
    v = b[:, :, 0] | (b[:, :, 1] << 8) | (b[:, :, 2] << 16)
    codes = np.stack([v & 63, (v >> 6) & 63, (v >> 12) & 63, (v >> 18) & 63], axis=-1)
    return codes.reshape(words.shape[0], -1)


def build(dt):
    ws, ss, bs = [], [], []
    for _ in range(N // a.chunk):
        wf = mx.array(rng.standard_normal((a.chunk, K), dtype=np.float32)).astype(dt)
        w, s, b = mx.quantize(wf, group_size=GS, bits=BITS)
        mx.eval(w, s, b)
        ws.append(w)
        ss.append(s)
        bs.append(b)
    w, s, b = mx.concatenate(ws), mx.concatenate(ss), mx.concatenate(bs)
    mx.eval(w, s, b)
    return w, s, b


def step_shares(dev, ref, mant):
    """Share of outputs beyond 0.5, 1.5, 2.5 steps of the output dtype (mant mantissa bits) and the max, for |ref| >= 0.05 sigma."""
    sigma = float(np.std(ref))
    keep = np.abs(ref) >= 0.05 * sigma
    step = np.exp2(np.floor(np.log2(np.abs(ref[keep]))) - mant)
    d = np.abs(dev[keep] - ref[keep]) / step
    return float((d > 0.5).mean()), float((d > 1.5).mean()), float((d > 2.5).mean()), float(d.max())


cols = np.sort(rng.choice(N, size=min(a.cols, N), replace=False))
failed = False
for name, dt, mant in (("bf16", mx.bfloat16, 7), ("f16", mx.float16, 10), ("f32", mx.float32, 23)):
    if name not in a.cells.split(","):
        continue
    w, s, b = build(dt)
    wsel = np.array(w[mx.array(cols)])
    ssel = np.array(s[mx.array(cols)].astype(mx.float32)).astype("float64")
    bsel = np.array(b[mx.array(cols)].astype(mx.float32)).astype("float64")
    codes = decode_codes(wsel).astype("float64")
    wd = codes * np.repeat(ssel, GS, axis=1) + np.repeat(bsel, GS, axis=1)
    with mx.stream(mx.cpu):
        deq = mx.dequantize(w[mx.array(cols)], s[mx.array(cols)].astype(mx.float32), b[mx.array(cols)].astype(mx.float32), group_size=GS, bits=BITS)
        mx.eval(deq)
    dd = float(np.abs(np.array(deq.astype(mx.float32)) - wd).max() / np.abs(wd).max())
    print(f"{name} host decode vs mx.dequantize (CPU stream) on {len(cols)} rows: max rel diff {dd:.2e}")
    if dd > 1e-3:
        print("HOST DECODE DISAGREES WITH mx.dequantize: reference not trusted")
        sys.exit(5)
    for var, rows in [(v, int(r)) for v in a.variants.split(",") for r in a.rows.split(",")]:
        if var == "d":
            os.environ.pop("MLX_OMARCHY_QMM6_VARIANT", None)
        else:
            os.environ["MLX_OMARCHY_QMM6_VARIANT"] = var
        x = mx.array(rng.standard_normal((rows, K), dtype=np.float32)).astype(dt)
        try:
            ids = traced(lambda: mx.eval(mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GS, bits=BITS)))
            y = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GS, bits=BITS)
            mx.eval(y)
        except (RuntimeError, ValueError) as exc:  # a layout this build refuses is a result
            print(f"{name} variant={var} rows={rows}: ERROR {str(exc).strip().splitlines()[0][:160]}")
            failed = True
            continue
        ref = np.array(x.astype(mx.float32)).astype("float64") @ wd.T
        got = np.array(y.astype(mx.float32)).astype("float64")[:, cols]
        finite = bool(np.isfinite(got).all())
        if name == "f32":
            err = float(np.abs(got - ref).max() / np.abs(ref).max())
            ok = finite and err <= 1e-4
            verdict = f"rel err {err:.2e} (limit 1e-4)"
        else:
            s05, s15, s25, smax = step_shares(got, ref, mant)
            ok = finite and s15 == 0.0 and s25 == 0.0
            verdict = f"steps: >0.5 {s05:.2e}, >1.5 {s15:.2e}, >2.5 {s25:.2e}, max {smax:.2f} (rule: >1.5 and >2.5 = 0)"
        failed |= not ok
        samples = []
        for r in range(a.reps + 1):
            t0 = time.perf_counter()
            mx.eval(mx.quantized_matmul(x, w, s, b, transpose=True, group_size=GS, bits=BITS))
            if r:
                samples.append((time.perf_counter() - t0) * 1e3)
        print(f"{name} variant={var} rows={rows} N={N} K={K} g{GS} 6-bit: {verdict} {'ok' if ok else 'FAIL'}; finite {finite}; "
              f"kernel ids {ids}; median {statistics.median(samples):.3f} ms per call "
              f"(min {min(samples):.3f}, max {max(samples):.3f}, {a.reps} calls)")
sys.exit(1 if failed else 0)
