"""Decode (M=1) quantized matmul vs an fp64-free fp32 reference (dequantize, fp32
matmul): rel-L2 and max |d| per format/dtype/shape, plus the dispatched kernel
id. Run with and without MLX_OMARCHY_QMM_VEC_Q4_WORD=0 (generic kernel)."""
import json
import mlx.core as mx
mx.random.seed(11)
for dt in (mx.bfloat16, mx.float16):
    for bits, gs in ((4, 64), (4, 32), (8, 64), (8, 32)):
        for K, N, batch in ((4096, 12288, 1), (12288, 4096, 1), (1024, 1000, 1), (4096, 1027, 3)):
            w = (mx.random.normal((N, K)) * 0.02).astype(dt)
            wq, s, b = mx.quantize(w, group_size=gs, bits=bits)
            x = mx.random.normal((batch, 1, K)).astype(dt)
            y = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=bits)
            ref = x.astype(mx.float32) @ mx.dequantize(wq, s, b, group_size=gs, bits=bits).astype(mx.float32).T
            d = y.astype(mx.float32) - ref
            print(json.dumps(dict(dt=str(dt).split('.')[-1], q=f"q{bits}g{gs}", K=K, N=N, B=batch,
                rel=(mx.sqrt((d*d).sum())/mx.sqrt((ref*ref).sum())).item(), max=mx.abs(d).max().item())), flush=True)
