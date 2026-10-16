"""gather3_gpu_check.py [bits, default 3]: affine gather_qmm on the default (GPU) stream against the CPU stream and an fp64 dequantize reference.
Shapes follow a routed-MoE call (x [T,1,1,K], experts [E,N,K], indices [T,top_k]). Prints per-case max abs error and the CPU/GPU gap.
fp32 and fp16 only (no bf16). Exit 1 when a GPU error exceeds 3x the CPU path's own error to the fp64 reference plus 1e-3 relative to the output scale."""
import sys

import mlx.core as mx
import numpy as np

mx.random.seed(7)
BITS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
dev = mx.default_device()
print("libmlx loaded:", sorted({ln.split()[-1] for ln in open("/proc/self/maps") if "libmlx" in ln}), flush=True)
print("default device:", dev, "mlx", mx.__version__, flush=True)
bad = 0
cases = [(1, 8, 16, 2048, 1024), (1, 8, 64, 4096, 512), (32, 8, 16, 2048, 1024), (3, 2, 4, 192, 37)]
for dtype in (mx.float32, mx.float16):
    for T, top_k, E, K, N in cases:
        w = (mx.random.normal((E, N, K)) * 0.5).astype(mx.float32)
        wq, sc, bi = mx.quantize(w, group_size=64, bits=BITS)
        x = mx.random.normal((T, 1, 1, K)).astype(dtype)
        idx = mx.random.randint(0, E, (T, top_k)).astype(mx.uint32)
        sc, bi = sc.astype(dtype), bi.astype(dtype)

        def run(stream):
            y = mx.gather_qmm(x, wq, sc, bi, rhs_indices=idx, transpose=True, group_size=64, bits=BITS, stream=stream)
            mx.eval(y)
            return np.array(y.astype(mx.float32), dtype=np.float64)

        wd = np.array(mx.dequantize(wq, sc.astype(mx.float32), bi.astype(mx.float32), group_size=64, bits=BITS, stream=mx.cpu), dtype=np.float64)
        xn = np.array(x.astype(mx.float32), dtype=np.float64)
        ix = np.array(idx)
        ref = np.einsum("tk,tjnk->tjn", xn[:, 0, 0], np.stack([wd[ix[t]] for t in range(T)]))[:, :, None, :]
        gpu = run(mx.gpu)
        cpu = run(mx.cpu)
        scale = max(np.abs(ref).max(), 1e-6)
        eg, ec = np.abs(gpu - ref).max(), np.abs(cpu - ref).max()
        ok = eg <= 3 * ec + 1e-3 * scale
        bad += not ok
        print(f"{str(dtype):16s} T={T:2d} k={top_k} E={E:2d} K={K} N={N}: gpu-ref {eg:.3e} cpu-ref {ec:.3e} gpu-cpu {np.abs(gpu - cpu).max():.3e} scale {scale:.3f} {'OK' if ok else 'FAIL'}", flush=True)
print("RESULT", "FAIL" if bad else "PASS", flush=True)
sys.exit(1 if bad else 0)
