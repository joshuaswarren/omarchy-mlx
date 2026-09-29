# Bit-exact check of the bf16 head_dim-256 decode SDPA route: print per-L digests, run under two wheels (or MLX_OMARCHY_SDPA_DECODE_NATIVE=0 vs default) and compare.
import hashlib
import json
import sys

import mlx.core as mx
import numpy as np

H, KV, D = 8, 2, 256
out = {}
for L in (12, 13, 33, 64, 100, 128, 129, 200, 256, 500, 1024, 1500, 2047, 2048):
    h = hashlib.sha256()
    for seed in range(5):
        rng = np.random.default_rng(1000 * L + seed)
        q = mx.array(rng.standard_normal((1, H, 1, D)).astype(np.float32)).astype(mx.bfloat16)
        kf = mx.array(rng.standard_normal((1, KV, 2304, D)).astype(np.float32) * (1.0 + seed)).astype(mx.bfloat16)
        vf = mx.array(rng.standard_normal((1, KV, 2304, D)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(q, kf, vf)
        k = kf[:, :, :L, :]  # strided cache slice like the model's KV cache
        v = vf[:, :, :L, :]
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=D ** -0.5)
        mx.eval(y)
        h.update(np.array(y.view(mx.uint16)).tobytes())
    out[L] = h.hexdigest()[:16]
print("SDPAHASH", json.dumps(out))
print("SDPAHASHALL", hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:24])
