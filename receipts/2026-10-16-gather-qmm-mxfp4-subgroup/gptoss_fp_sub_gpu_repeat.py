"""gptoss_fp_sub_gpu.py: mxfp4 gather_qmm at the gpt-oss-20b expert shape (32 experts, K 2880, N 2880, top-4 per token): the default route (subgroup kernel when the wheel has it) against the scalar route
(MLX_OMARCHY_GATHER_QMM_SUB=0, same process), each against an fp64 numpy reference on a column subset, plus the per-call GPU time (median of 7 after a warm-up) as a cost diagnostic.
Cells: bf16, f16 and bf16 again (the repeat shows whether the first cell pays a clock ramp); index_count 4 (one decoded token, top-4) and 64 (a short sorted prefill block, x passed as [B, 1, K]). Prints CELLMARK / CELLEND on stderr around the default-route call so the wrapper can read the kernel id from the backend trace.
Random weights: cost and numerics only, not model speed. Rule: |gpu - reference| <= rel * max|reference| with rel 8e-3 (bf16) and 2e-3 (f16)."""
import os
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

print("libmlx loaded:", sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "libmlx" in ln}), flush=True)
print("default device:", mx.default_device(), "mlx", mx.__version__, flush=True)
E, K, N, GS = 32, 2880, 2880, 32
rs = np.random.RandomState(11)
words = rs.randint(0, 2**32, size=(E, N, K // 8), dtype=np.uint64).astype(np.uint32)
scale_bytes = rs.randint(120, 133, size=(E, N, K // GS)).astype(np.uint8)
LUT = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
w = mx.array(words)
sc = mx.array(scale_bytes)
mx.eval(w, sc)


def reference(x32, rhs, cols):
    """fp64 reference for output columns `cols`: out[b, c] = sum_k x[b, k] * scale * e2m1(code)."""
    out = np.zeros((len(rhs), len(cols)))
    for b, e in enumerate(rhs):
        for j, c in enumerate(cols):
            nib = ((words[e, c][:, None] >> (np.arange(8, dtype=np.uint32)[None, :] * 4)) & 0xF).reshape(-1)
            val = LUT[nib & 7] * np.where(nib & 8, -1.0, 1.0)
            s = np.where(scale_bytes[e, c] == 0, 2.0**-127, 2.0 ** (scale_bytes[e, c].astype(np.float64) - 127))
            out[b, j] = float(np.dot(x32[b].astype(np.float64), val * np.repeat(s, GS)))
    return out


bad = 0
cols = np.arange(0, N, 23)[:128]
for pass_no, (dt, rel) in enumerate(((mx.bfloat16, 8e-3), (mx.float16, 2e-3), (mx.bfloat16, 8e-3))):
    dname = str(dt).split(".")[-1] + ("" if pass_no < 2 else " (second pass)")
    for B in (4, 64):
        rhs = np.sort(rs.randint(0, E, size=B)).astype(np.uint32) if B > 4 else np.array([3, 7, 7, 20], dtype=np.uint32)
        x32 = rs.uniform(-2, 2, size=(B, K)).astype(np.float32)
        x = mx.array(x32).astype(dt).reshape(B, 1, K)
        lhs = mx.arange(B, dtype=mx.uint32)
        r = mx.array(rhs)
        mx.eval(x, lhs, r)
        xs = np.array(x.astype(mx.float32)).reshape(B, K)
        ref = reference(xs, rhs, cols)
        scale = float(np.abs(ref).max())

        def call(sub):
            if sub is None:
                os.environ.pop("MLX_OMARCHY_GATHER_QMM_SUB", None)
            else:
                os.environ["MLX_OMARCHY_GATHER_QMM_SUB"] = sub
            y = mx.gather_qmm(x, w, sc, None, lhs, r, transpose=True, group_size=GS, bits=4, mode="mxfp4")
            mx.eval(y)
            return y

        line = f"{dname} index_count {B}:"
        try:
            call(None)  # warm-up
            print(f"CELLMARK {line}", file=sys.stderr, flush=True)
            yd = call(None)
            print(f"CELLEND {line}", file=sys.stderr, flush=True)
            res = {}
            for tag, sub in (("default", None), ("scalar", "0")):
                call(sub)
                times = []
                for _ in range(7):
                    t0 = time.perf_counter()
                    y = call(sub)
                    mx.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)
                got = np.array(y.astype(mx.float32)).reshape(B, N)[:, cols].astype(np.float64)
                err = float(np.abs(got - ref).max())
                ok = bool(np.isfinite(got).all()) and err <= rel * scale
                bad += not ok
                res[tag] = (statistics.median(times), min(times), max(times), err, ok)
                print(f"TIMES {dname} index_count {B} {tag}: " + " ".join(f"{t:.3f}" for t in times), flush=True)
            os.environ.pop("MLX_OMARCHY_GATHER_QMM_SUB", None)
            d, s = res["default"], res["scalar"]
            line += f" default ms {d[0]:.3f} (min {d[1]:.3f} max {d[2]:.3f}) err {d[3]:.3e} {'OK' if d[4] else 'FAIL'}; scalar ms {s[0]:.3f} (min {s[1]:.3f} max {s[2]:.3f}) err {s[3]:.3e} {'OK' if s[4] else 'FAIL'}; ref max {scale:.2f} tol {rel * scale:.3e}; speedup {s[0] / d[0]:.2f}x"
            if B == 4:
                line += f"; 24 layers x 3 calls x default = {72 * d[0]:.0f} ms, x scalar = {72 * s[0]:.0f} ms per token (expert matmuls only)"
        except Exception as e:  # report verbatim
            bad += 1
            line += f" ERROR {type(e).__name__}: {str(e)[:300]}"
        print(line, flush=True)
print("peak memory GiB", round(mx.get_peak_memory() / 2**30, 2), flush=True)
print("RESULT", "FAIL" if bad else "PASS", flush=True)
sys.exit(1 if bad else 0)
