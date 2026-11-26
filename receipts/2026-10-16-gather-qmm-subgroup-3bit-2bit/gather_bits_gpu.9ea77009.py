"""gather_bits_gpu.py <bits> <group>: affine gather_qmm, f16, transposed, routed-MoE shapes. Each case runs on the GPU twice (the default route, and the scalar
kernel with MLX_OMARCHY_GATHER_QMM_SUB=0) and on the CPU stream, and is compared with an fp64 dequantize reference over the gathered experts.
Pass rule: each GPU error <= 3 x the CPU stream's error + 1e-3 x the output scale. The last case is the real Muse expert shape (K 4096, N 4096, top-8, 24 experts):
per-call GPU time of both routes is printed as a cost diagnostic (median of 5 after a warm-up), not a benchmark. No bf16."""
import os
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

BITS, GS = int(sys.argv[1]), int(sys.argv[2])
mx.random.seed(11)
print("libmlx loaded:", sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "libmlx" in ln}), flush=True)
print("default device:", mx.default_device(), "mlx", mx.__version__, "bits", BITS, "group", GS, flush=True)
cases = [(1, 8, 16, 2048, 1024, False), (1, 8, 64, 4096, 512, False), (32, 8, 16, 2048, 1024, False), (3, 2, 4, 192 if GS <= 64 else 256, 40, False), (1, 8, 24, 4096, 4096, True)]
if len(sys.argv) > 3 and sys.argv[3] == "small":
    cases = cases[3:4]  # dispatch-trace pass: one small case only
bad = 0
for T, top_k, E, K, N, real in cases:
    w = mx.multiply(mx.random.normal((E, N, K), stream=mx.cpu), 0.05, stream=mx.cpu).astype(mx.float16, stream=mx.cpu)
    wq, sc, bi = mx.quantize(w, group_size=GS, bits=BITS, stream=mx.cpu)
    del w
    x = mx.random.normal((T, 1, 1, K), stream=mx.cpu).astype(mx.float16, stream=mx.cpu)
    idx = mx.random.randint(0, E, (T, top_k), stream=mx.cpu).astype(mx.uint32, stream=mx.cpu)
    mx.eval(wq, sc, bi, x, idx)

    def call(stream, wq=wq, sc=sc, bi=bi, x=x, idx=idx):
        y = mx.gather_qmm(x, wq, sc, bi, rhs_indices=idx, transpose=True, group_size=GS, bits=BITS, stream=stream)
        mx.eval(y)
        return y

    ix = np.array(idx)
    uniq = np.unique(ix)
    wd = np.array(mx.dequantize(wq[mx.array(uniq)], sc[mx.array(uniq)].astype(mx.float32), bi[mx.array(uniq)].astype(mx.float32), group_size=GS, bits=BITS, stream=mx.cpu), dtype=np.float64)
    pos = {int(e): i for i, e in enumerate(uniq)}
    xn = np.array(x.astype(mx.float32), dtype=np.float64)[:, 0, 0]
    ref = np.stack([np.einsum("k,jnk->jn", xn[t], wd[[pos[int(e)] for e in ix[t]]]) for t in range(T)])[:, :, None, :]
    scale = max(float(np.abs(ref).max()), 1e-6)
    cpu = np.array(call(mx.cpu).astype(mx.float32), dtype=np.float64)
    res = {}
    for name, flag in (("default", None), ("scalar", "0")):
        if flag is None:
            os.environ.pop("MLX_OMARCHY_GATHER_QMM_SUB", None)
        else:
            os.environ["MLX_OMARCHY_GATHER_QMM_SUB"] = flag
        y = call(mx.gpu)
        times = []
        if real:
            for _ in range(5):
                t0 = time.perf_counter()
                y = call(mx.gpu)
                mx.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
        res[name] = (np.array(y.astype(mx.float32), dtype=np.float64), statistics.median(times) if times else None)
    os.environ.pop("MLX_OMARCHY_GATHER_QMM_SUB", None)
    ec = np.abs(cpu - ref).max()
    line = f"T={T} k={top_k} E={E} K={K} N={N}: cpu-ref {ec:.3e} scale {scale:.2f}"
    ok = True
    for name in ("default", "scalar"):
        eg = np.abs(res[name][0] - ref).max()
        good = bool(np.isfinite(res[name][0]).all()) and eg <= 3 * ec + 1e-3 * scale
        ok &= good
        line += f" | {name} gpu-ref {eg:.3e}" + (f" ms/call {res[name][1]:.2f}" if res[name][1] is not None else "")
    line += f" | default-vs-scalar {np.abs(res['default'][0] - res['scalar'][0]).max():.3e} {'OK' if ok else 'FAIL'}"
    bad += not ok
    print(line, flush=True)
print("peak memory GiB", round(mx.get_peak_memory() / 2**30, 2), flush=True)
print("RESULT", "FAIL" if bad else "PASS", flush=True)
sys.exit(1 if bad else 0)
