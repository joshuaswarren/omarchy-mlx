"""Split the decode GEMV's per-call cost: fixed part vs per-byte part.

Times COPIES independent 4-bit/g64 bf16 GEMVs in one graph for N x K
shapes, plus a copy-bandwidth ceiling (a + 1 over 512 MB, read+write).
usage: gemv_sweep.py [reps]
"""
import sys
import time

import prov

prov.check()
import mlx.core as mx

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def timed(fn):
    fn()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def main():
    a = mx.random.normal((128 << 20,)).astype(mx.float32)
    mx.eval(a)
    t = timed(lambda: mx.eval(a + 1))
    print(f"copy ceiling (f32 512 MB in, 512 MB out): {2 * a.nbytes / t / 1e9:6.1f} GB/s", flush=True)
    del a
    for n, k in [(64, 2048), (512, 512), (512, 2048), (512, 8192), (2048, 512),
                 (2048, 2048), (2048, 8192), (8192, 2048)]:
        wbytes = n * k // 2 + 2 * 2 * n * (k // 64)
        copies = max(8, min(256, (384 << 20) // wbytes))
        ws = []
        for _ in range(copies):
            q = mx.quantize(mx.random.normal((n, k)).astype(mx.bfloat16), group_size=64, bits=4)
            mx.eval(q)
            ws.append(q)
        x = mx.random.normal((1, k)).astype(mx.bfloat16)

        def run():
            outs = [mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
                    for wq, s, b in ws]
            mx.eval(outs)

        t = timed(run) / copies
        print(f"N={n:5d} K={k:5d}  {t * 1e6:7.1f} us/gemv  {wbytes / t / 1e9:6.1f} GB/s", flush=True)
        del ws


if __name__ == "__main__":
    main()
