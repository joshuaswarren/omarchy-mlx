# f16 kernels that round with omarchy_f16_bits_rne (head) or packHalf2x16 (old): time and output digest.
import hashlib, statistics, sys, time
import mlx.core as mx
import mlx.nn as nn
mx.random.seed(7)
def digest(a):
    return hashlib.sha256(memoryview(a.view(mx.uint16)) if a.dtype == mx.float16 else memoryview(a)).hexdigest()[:16]
def bench(name, fn, iters=200):
    out = fn(); mx.eval(out); mx.synchronize()
    for _ in range(10): mx.eval(fn())
    mx.synchronize()
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(iters // 5): mx.eval(fn())
        mx.synchronize(); ts.append((time.perf_counter() - t) / (iters // 5) * 1e6)
    print(f"{name:28s} median_us {statistics.median(ts):9.1f} min_us {min(ts):9.1f} digest {digest(out)}", flush=True)
x32 = mx.random.normal((16 * 1024 * 1024,))
bench("cast f32->f16 16M", lambda: x32.astype(mx.float16))
w = mx.random.normal((4096, 4096)).astype(mx.float16)
wq, sc, bi = mx.quantize(w, group_size=64, bits=4)
xv = mx.random.normal((1, 4096)).astype(mx.float16)
bench("qmm vec f16 1x4096x4096", lambda: mx.quantized_matmul(xv, wq, sc, bi, transpose=True, group_size=64, bits=4))
xm = mx.random.normal((8, 4096)).astype(mx.float16)
bench("qmm f16 8x4096x4096", lambda: mx.quantized_matmul(xm, wq, sc, bi, transpose=True, group_size=64, bits=4))
xp = mx.random.normal((512, 4096)).astype(mx.float16)
bench("qmm f16 512x4096x4096", lambda: mx.quantized_matmul(xp, wq, sc, bi, transpose=True, group_size=64, bits=4), iters=50)
g = mx.random.normal((64, 11008)).astype(mx.float16); u = mx.random.normal((64, 11008)).astype(mx.float16)
sw = mx.compile(lambda a, b: nn.silu(a) * b)
bench("silu*up f16 64x11008", lambda: sw(g, u))
h = mx.random.normal((64, 4096)).astype(mx.float16); nw = mx.random.normal((4096,)).astype(mx.float16)
bench("rms_norm f16 64x4096", lambda: mx.fast.rms_norm(h, nw, 1e-6))
for keys in (1100, 4096):
    q = mx.random.normal((1, 16, 1, 128)).astype(mx.float16)
    k = mx.random.normal((1, 8, keys, 128)).astype(mx.float16); v = mx.random.normal((1, 8, keys, 128)).astype(mx.float16)
    bench(f"sdpa decode f16 {keys} keys", lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5))
