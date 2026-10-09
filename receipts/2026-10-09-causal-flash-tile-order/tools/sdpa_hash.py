"""Output hash of the causal flash route on the 4B attention shape (H40 rule 1). usage: sdpa_hash.py [L,L,...]
Forces the route with MLX_OMARCHY_SDPA_CAUSAL_FLASH=1 (set here), deterministic inputs from a fixed seed, prints one JSON line
with the wheel version and sha256 (first 16 hex) of the float32-converted output per length, plus the same hash for the
composed route (flag off) so a reader can see that flash and composed differ (they are not bit-identical to each other)."""
import hashlib
import json
import os
import sys

import mlx.core as mx
import numpy as np

lens = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "512,1000,1024,2048").split(",")]
Hq, Hkv, HD = 32, 8, 128
res = {"mlx": mx.__version__, "device": mx.device_info().get("device_name"), "flash": {}, "composed": {}}
rng = np.random.default_rng(7)
for L in lens:
    q = mx.array(rng.standard_normal((1, Hq, L, HD)).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((1, Hkv, L, HD)).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((1, Hkv, L, HD)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    for flag, name in (("1", "flash"), ("0", "composed")):
        os.environ["MLX_OMARCHY_SDPA_CAUSAL_FLASH"] = flag
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=HD ** -0.5, mask="causal")
        mx.eval(o)
        res[name][str(L)] = hashlib.sha256(np.array(o.astype(mx.float32)).tobytes()).hexdigest()[:16]
print(json.dumps(res))
