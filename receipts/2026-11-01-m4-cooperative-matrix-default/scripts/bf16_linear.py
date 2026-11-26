"""bf16 linear-layer matmul (x @ W.T) at Llama 3.2 1B shapes: GFLOP/s and
the error against an f32 CPU-stream reference.

Arms, on one wheel and one cooperative-matrix driver:
  default                       this change's pick (scalar FMA on G16G)
  MLX_OMARCHY_NO_MATMUL_FMA=1   the cooperative-matrix pick
Run each arm in its own process. A profiling wheel with
MLX_OMARCHY_GPU_PROFILE set records which kernel each arm dispatched.
"""
import time

import prov

prov.check()
import mlx.core as mx  # noqa: E402

SHAPES = [(2048, 2048), (512, 2048), (8192, 2048), (2048, 8192)]  # (N out, K in)
mx.random.seed(0)
for m in (128, 512, 2048):
    cells = []
    for n, k in SHAPES:
        x = mx.random.normal((m, k)).astype(mx.bfloat16)
        ws = [mx.random.normal((n, k)).astype(mx.bfloat16) for _ in range(8)]
        mx.eval(x, ws)
        y = x @ ws[0].T
        ref = mx.matmul(x.astype(mx.float32), ws[0].astype(mx.float32).T, stream=mx.cpu)
        err = mx.max(mx.abs(y.astype(mx.float32) - ref) / (mx.abs(ref) + 1.0)).item()

        def run():
            mx.eval([x @ w.T for w in ws])

        run()
        ts = []
        for _ in range(5):
            t = time.perf_counter()
            run()
            ts.append(time.perf_counter() - t)
        t = sorted(ts)[2] / len(ws)
        cells.append(f"{n}x{k}: {2 * m * n * k / t / 1e9:6.0f} (err {err:.1e})")
    print(f"M={m:5d}  " + "  ".join(cells), flush=True)
