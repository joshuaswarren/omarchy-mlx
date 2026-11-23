import time
import mlx.core as mx

V, K = 248320, 2048
mx.random.seed(0)
x = mx.random.normal((1, K)).astype(mx.bfloat16)
w = (mx.random.normal((V, K)) * 0.02).astype(mx.bfloat16)
mx.eval(x, w)
res = {}
for bits in (4, 3, 2):
    wq, s, b = mx.quantize(w, group_size=64, bits=bits)
    mx.eval(wq, s, b)
    f = lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=bits)
    for _ in range(5):
        mx.eval(f())
    reps, inner = 5, 20
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        outs = [f() for _ in range(inner)]
        mx.eval(outs)
        best = min(best, (time.perf_counter() - t0) / inner)
    nbytes = wq.nbytes + s.nbytes + b.nbytes
    res[bits] = best
    print(f"bits={bits}: {best*1e3:.3f} ms/call  bytes={nbytes/1e6:.1f} MB  {nbytes/best/1e9:.1f} GB/s", flush=True)
    del wq, s, b
print("3-bit / 4-bit time ratio", round(res[3] / res[4], 3))
