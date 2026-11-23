import hashlib
import os

import mlx.core as mx
import numpy as np

mx.random.seed(0)
w = (mx.random.normal((4096, 4096)) * 0.02).astype(mx.bfloat16)
wq, s, b = mx.quantize(w, group_size=64, bits=4)
x = mx.random.normal((1, 23, 4096)).astype(mx.bfloat16)
out = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=4)
mx.eval(out)
cores = os.environ.get("MLX_OMARCHY_GPU_CORES", "")
arm = "ctl32" if cores == "" else f"cores{cores}"
print(
    "arm=" + arm,
    hashlib.sha256(np.asarray(out.astype(mx.float32)).tobytes()).hexdigest()[:16],
)
