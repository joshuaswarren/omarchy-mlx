"""Coopmat prefill for 4/32, 8/64, 8/32 vs an fp32 reference (dequantize, then
fp32 matmul), next to the error of the shipped 4/64 route. Prints max-abs and
relative-L2 error per (format, dtype, M, N); run twice (default, and
MLX_OMARCHY_NO_COOPMAT=1 for the tile path) to compare; out_sha256 hashes the
output bytes, so equal hashes mean bit-identical outputs."""
import hashlib, json, os
import mlx.core as mx
mx.random.seed(7)
K = 4096
rows = []
for dtype in (mx.bfloat16, mx.float16):
    for bits, gs in ((4, 64), (4, 32), (8, 64), (8, 32)):
        for M, N in ((5, 4096), (17, 1024), (33, 4096), (100, 1000), (512, 4096)):
            w = (mx.random.normal((N, K)) * 0.02).astype(dtype)
            wq, s, b = mx.quantize(w, group_size=gs, bits=bits)
            x = mx.random.normal((M, K)).astype(dtype)
            y = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=bits)
            wd = mx.dequantize(wq, s, b, group_size=gs, bits=bits).astype(mx.float32)
            ref = x.astype(mx.float32) @ wd.T
            d = y.astype(mx.float32) - ref
            r = dict(dtype=str(dtype).split('.')[-1], bits=bits, gs=gs, M=M, N=N,
                     max_abs=mx.abs(d).max().item(),
                     rel_l2=(mx.sqrt((d * d).sum()) / mx.sqrt((ref * ref).sum())).item(),
                     out_sha256=hashlib.sha256(bytes(memoryview(y.view(mx.uint16)))).hexdigest()[:16])
            rows.append(r); print(json.dumps(r), flush=True)
