"""Quantized matmul microbench: (bits, group) x dtype x M. Shapes = K2 MLP up_proj (4096->12288).
Warmup 3 calls, then 20 timed calls (5 at M=512), wall clock around mx.eval; dense matmul reference at the end."""
import time, sys, json
import mlx.core as mx
K, N = 4096, 12288
def bench(bits, gs, dtype, M, iters):
    w = mx.random.normal((N, K)).astype(dtype)
    wq, s, b = mx.quantize(w, group_size=gs, bits=bits)
    x = mx.random.normal((M, K)).astype(dtype)
    mx.eval(wq, s, b, x)
    f = lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=bits)
    for _ in range(3): mx.eval(f())
    t = time.perf_counter()
    for _ in range(iters): mx.eval(f())
    dt = (time.perf_counter() - t) / iters
    wbytes = wq.nbytes + s.nbytes + b.nbytes
    return dict(bits=bits, gs=gs, dtype=str(dtype).split('.')[-1], M=M, ms=round(dt*1e3, 3),
                GBps=round(wbytes/dt/1e9, 1), GFLOPs=round(2*M*N*K/dt/1e9, 1))
rows = []
for dtype in (mx.float16, mx.bfloat16):
    for bits, gs in ((4, 64), (4, 32), (8, 64), (8, 32)):
        for M in (1, 32, 512):
            r = bench(bits, gs, dtype, M, 20 if M < 512 else 5); rows.append(r); print(json.dumps(r), flush=True)
# dense reference
for dtype in (mx.float16, mx.bfloat16):
    w = mx.random.normal((N, K)).astype(dtype); x = mx.random.normal((512, K)).astype(dtype); mx.eval(w, x)
    for _ in range(2): mx.eval(x @ w.T)
    t = time.perf_counter()
    for _ in range(5): mx.eval(x @ w.T)
    dt = (time.perf_counter()-t)/5
    print(json.dumps(dict(dense=str(dtype), M=512, ms=round(dt*1e3,2), GFLOPs=round(2*512*N*K/dt/1e9,1))), flush=True)
print(json.dumps(dict(peak_memory_bytes=mx.get_peak_memory())), flush=True)
