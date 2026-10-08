import mlx.core as mx, sys
print("mlx", mx.__version__)
print("fast:", [n for n in dir(mx.fast) if not n.startswith("_")])
try:
    from mlx.core.fast import metal_kernel
    print("metal_kernel ok")
except Exception as e:
    print("metal_kernel import fail:", e)
print("compile:", hasattr(mx, "compile"))
print("defaults:", mx.default_device())
