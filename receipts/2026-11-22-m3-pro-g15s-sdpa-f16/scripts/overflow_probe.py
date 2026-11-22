"""Astra R1 input: f16 decode SDPA, q and k zero, v filled with a constant, uniform attention -> exact answer = the constant.
usage: overflow_probe.py EXPECT_WHEEL REPO_ROOT"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[2]) / "scripts"))
import mlx_provenance as mp
prov = mp.installed_provenance(expect_wheel=Path(sys.argv[1])); print(mp.provenance_line(prov), flush=True)
if prov["verified"] != "match" or prov["version_match"] is not True: sys.exit(3)
import os
import mlx.core as mx
print("MLX_OMARCHY_SDPA_F16_NO_COOPMAT =", os.environ.get("MLX_OMARCHY_SDPA_F16_NO_COOPMAT"))
for keys in (1023, 1024, 4096, 8192):
    for val in (1024.0, 2047.0, 2048.0, 4096.0, 8192.0):
        q = mx.zeros((1, 16, 1, 128), dtype=mx.float16); k = mx.zeros((1, 8, keys, 128), dtype=mx.float16)
        v = mx.full((1, 8, keys, 128), val, dtype=mx.float16)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=128 ** -0.5); mx.eval(y)
        yf = y.astype(mx.float32)
        print(f"keys={keys:5d} v={val:6.0f} out_min={yf.min().item()} out_max={yf.max().item()} inf={mx.isinf(yf).sum().item()}", flush=True)
