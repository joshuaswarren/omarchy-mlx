# qmm cost vs pass length T for [1,T,K] bf16 x 4-bit/g64 W (Qwen3-8B shapes)
import time, sys, os, mlx.core as mx
mx.random.seed(0)
shapes = [(12288,4096),(4096,12288),(4096,4096),(1024,4096)]
for N,K in shapes:
    copies = max(2, min(64, int(1.2e9 // (N*K//2))))
    ws=[]
    for c in range(copies):
        w = mx.random.normal((N,K)).astype(mx.bfloat16); ws.append(mx.quantize(w, group_size=64, bits=4)); mx.eval(ws[-1]); del w
    bytes_per = sum(a.nbytes for a in ws[0]); line=[]
    for T in range(1, 9):
        x = mx.random.normal((1,T,K)).astype(mx.bfloat16)
        def run(reps):
            outs=[mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4) for r in range(reps) for (wq,s,b) in ws]; mx.eval(outs)
        run(1); best=1e9
        for t in range(5):
            t0=time.perf_counter(); run(2); best=min(best,time.perf_counter()-t0)
        line.append(f"T{T}={1e6*best/(2*copies):.0f}us")
    print(f"N={N} K={K} MB={bytes_per/1e6:.1f} GB/s@T1={bytes_per*2*copies/1e9:.1f}/.. ", " ".join(line), flush=True)
    del ws
