"""Vulkan device reopen check: a fresh process opens the GPU, runs a small
matmul and a 4-bit quantized_matmul, and compares them with the CPU stream.
Prints REOPEN_OK or exits non-zero."""
import prov

prov.check()
import mlx.core as mx  # noqa: E402

print("device:", mx.default_device(), flush=True)
info = getattr(mx, "device_info", None)
if info:
    try:
        d = info()
        print("device_name:", d.get("device_name") or d.get("architecture"), flush=True)
    except Exception as e:  # report, do not fail on the name lookup
        print("device_info error:", e)
mx.random.seed(1)
a = mx.random.normal((64, 256))
b = mx.random.normal((256, 128))
g = a @ b
c = mx.matmul(a, b, stream=mx.cpu)
wq, s, bb = mx.quantize(mx.random.normal((256, 256)), group_size=64, bits=4)
x = mx.random.normal((4, 256))
qg = mx.quantized_matmul(x, wq, s, bb, transpose=True, group_size=64, bits=4)
qc = mx.quantized_matmul(x, wq, s, bb, transpose=True, group_size=64, bits=4, stream=mx.cpu)
mx.eval(g, c, qg, qc)
e1 = mx.max(mx.abs(g - c)).item()
e2 = mx.max(mx.abs(qg - qc)).item()
print(f"matmul max|gpu-cpu|={e1:.2e} qmm max|gpu-cpu|={e2:.2e}", flush=True)
assert e1 < 1e-3 and e2 < 1e-3
print("REOPEN_OK", flush=True)
