"""gather3_real_shape.py: the routed-expert shape of the real Muse Spark Open 3bit checkpoint (gate/up/down are [experts, N=4096, K=4096], 3-bit affine, group 64;
read from the safetensors header of model-00008, layer 5) on the default (GPU) stream against the CPU stream. The expert count is 24 instead of 256 so
the random weights fit; the per-call work depends on top_k and N and K, not on the expert count (except the staged scale/bias copy, which scales with it: printed).
fp16 activations (no bf16). Prints the GPU-vs-CPU gap and the per-call GPU time (median of 5 after one warm-up; a cost diagnostic, not a benchmark)."""
import statistics
import sys
import time

import mlx.core as mx
import numpy as np

mx.random.seed(5)
print("libmlx loaded:", sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "libmlx" in ln}), flush=True)
print("default device:", mx.default_device(), "mlx", mx.__version__, flush=True)
E, K, N, GS = 24, 4096, 4096, 64
bad = 0
w = (mx.random.normal((E, N, K), stream=mx.cpu) * 0.05).astype(mx.float16, stream=mx.cpu)
wq, sc, bi = mx.quantize(w, group_size=GS, bits=3, stream=mx.cpu)
mx.eval(wq, sc, bi)
del w
print("quantized:", wq.shape, wq.dtype, sc.shape, sc.dtype, "words per row", wq.shape[-1], "= K*3/32", K * 3 // 32, flush=True)
for T, top_k in ((1, 8), (8, 8)):
    x = mx.random.normal((T, 1, 1, K)).astype(mx.float16)
    idx = mx.random.randint(0, E, (T, top_k)).astype(mx.uint32)

    def call(stream, x=x, idx=idx):
        y = mx.gather_qmm(x, wq, sc, bi, rhs_indices=idx, transpose=True, group_size=GS, bits=3, stream=stream)
        mx.eval(y)
        return y

    y_cpu = np.array(call(mx.cpu).astype(mx.float32), dtype=np.float64)
    call(mx.gpu)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        y = call(mx.gpu)
        mx.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    y_gpu = np.array(y.astype(mx.float32), dtype=np.float64)
    scale = max(np.abs(y_cpu).max(), 1e-6)
    gap = np.abs(y_gpu - y_cpu).max()
    ok = bool(np.isfinite(y_gpu).all()) and gap <= 5e-3 * scale
    bad += not ok
    print(f"T={T} top_k={top_k} K={K} N={N} E={E}: gpu-vs-cpu max abs {gap:.3e} output scale {scale:.3f} ms/call median {statistics.median(times):.2f} min {min(times):.2f} max {max(times):.2f} {'OK' if ok else 'FAIL'}", flush=True)
print("peak memory GiB", round(mx.get_peak_memory() / 2**30, 2), flush=True)
print("RESULT", "FAIL" if bad else "PASS", flush=True)
sys.exit(1 if bad else 0)
