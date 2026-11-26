import os, time
import mlx.core as mx
mx.random.seed(0)
env=os.environ.get('MLX_OMARCHY_QMM_BF16W','unset')
for (N,K) in [(4096,4096),(1024,4096),(12288,4096),(4096,12288),(151936,4096)]:
    w = (mx.random.normal((N,K))*0.02).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=4); del w
    for M in [256,512,768,1024,1536,2048]:
        x = mx.random.normal((1,M,K)).astype(mx.bfloat16)
        f=lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
        for _ in range(2): mx.eval(f())
        best=1e9
        for rep in range(3):
            t=time.perf_counter(); R=6
            for _ in range(R): mx.eval(f())
            best=min(best,(time.perf_counter()-t)/R*1e3)
        print(f"env={env} N={N} K={K} M={M} ms={best:.3f}", flush=True)
