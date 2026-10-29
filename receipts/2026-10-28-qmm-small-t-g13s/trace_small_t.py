# One quantized_matmul per pass length, under MLX_OMARCHY_TRACE_DISPATCH=1 (kernel ids on stderr).
import sys
import mlx.core as mx
mx.random.seed(0)
w = mx.random.normal((9216, 2560)).astype(mx.bfloat16)
wq, s, b = mx.quantize(w, group_size=64, bits=4)
mx.eval(wq, s, b)
for rows in (1, 2, 4, 6, 7):
    x = mx.random.normal((1, rows, 2560)).astype(mx.bfloat16)
    mx.eval(x)
    print(f"=== rows={rows}", file=sys.stderr, flush=True)
    mx.eval(mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4))
