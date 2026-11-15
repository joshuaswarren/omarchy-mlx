# Device reopen: a fresh process opens the GPU after the measured row and runs one op. Exit 0 and REOPEN-OK on success.
import mlx.core as mx

x = mx.ones((64,), dtype=mx.float32) + 1
mx.eval(x)
assert mx.default_device() == mx.gpu, mx.default_device()
assert float(x[0].item()) == 2.0
print("REOPEN-OK", mx.default_device())
