import sys, time, mlx.core as mx
mx.random.seed(0)
for (M,K,N) in [(512,4096,4096),(2048,4096,4096),(2048,4096,12288),(2048,12288,4096)]:
    w = mx.random.normal((N,K)).astype(mx.float16)*0.02
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    x = mx.random.normal((M,K)).astype(mx.float16)
    f = lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
    y = f(); mx.eval(y)
    for _ in range(3): mx.eval(f())
    ts=[]
    for _ in range(10):
        t=time.perf_counter(); mx.eval(f()); ts.append(time.perf_counter()-t)
    ts.sort(); t=ts[len(ts)//2]
    ref = (x.astype(mx.float32) @ mx.dequantize(wq,s,b,group_size=64,bits=4).astype(mx.float32).T)
    err = mx.max(mx.abs(y.astype(mx.float32)-ref)).item()
    print(f"QMMF16 M={M} K={K} N={N} ms {t*1e3:.3f} best {ts[0]*1e3:.3f} TFLOPs {2*M*K*N/t/1e12:.3f} maxerr {err:.4f}", flush=True)
