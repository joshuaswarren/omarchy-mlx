"""MatmulGap fill job: dense matmul bits over a shape grid, one arm per process. Rerun-safe (skips finished cells).

Usage: shape_sweep.py <out.jsonl>   (run under the arm's venv; compare arms with shape_sweep_cmp.py)
Grid hits the direct-route rows (min_m 512 / 4096, n >= 4096) and the gate edges (odd k remainder, odd n,
odd and even element offsets). Inputs come from numpy so every arm sees the same bits.
"""
import hashlib
import itertools
import json
import sys

import mlx.core as mx
import numpy as np

out = sys.argv[1]
try:
    done = {json.loads(x)["id"] for x in open(out) if x.strip()}
except FileNotFoundError:
    done = set()

DT = {"f16": mx.float16, "bf16": mx.bfloat16, "f32": mx.float32}
MS = (512, 640, 1536, 4096)
NS = (4096, 4098, 6144)
KS = (2048, 4104, 4096, 6144)
ORI = ("nn", "nt", "tn")
OFF = (0, 1, 2)


def normal(rng, *shape):
    return mx.array(rng.standard_normal(shape).astype(np.float32))


def operands(rng, ori, m, n, k, off, dt):
    a = normal(rng, m, k + off) if ori != "tn" else normal(rng, k, m + off)
    b = normal(rng, n, k) if ori == "nt" else normal(rng, k, n)
    a, b = a.astype(dt), b.astype(dt)
    a = a[..., off:] if off else a
    return (a.T if ori == "tn" else a), (b.T if ori == "nt" else b)


f = open(out, "a")
print(json.dumps({"k": "host", "mlx": mx.__version__, "device": mx.device_info().get("device_name")}), file=f, flush=True)
for name, ori, m, n, k, off in itertools.product(DT, ORI, MS, NS, KS, OFF):
    cid = f"{name}:{ori}:{m}x{n}x{k}:o{off}"
    if cid in done or (off and (m, n, k) != (4096, 4096, 4096)) or (n == 4098 and off):
        continue
    rng = np.random.RandomState(int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
    a, b = operands(rng, ori, m, n, k, off, DT[name])
    c = a @ b
    mx.eval(c)
    raw = np.array(c.astype(mx.float32)).tobytes()
    print(json.dumps({"k": "cell", "id": cid, "sha": hashlib.sha256(raw).hexdigest()[:16]}), file=f, flush=True)
