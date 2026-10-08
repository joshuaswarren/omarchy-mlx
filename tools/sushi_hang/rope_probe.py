import mlx.core as mx
a = mx.random.normal((1, 8, 32, 128)).astype(mx.bfloat16)
try:
    r = mx.fast.rope(a, 128, traditional=False, base=10000.0, scale=1.0, offset=0)
    mx.eval(r)
    print("kwargs ok", r.shape)
except Exception as e:
    print("kwargs fail:", e)
try:
    r = mx.fast.rope(a, 128, False, 10000.0, 1.0, 0)
    mx.eval(r)
    print("positional ok", r.shape)
except Exception as e:
    print("positional fail:", e)
