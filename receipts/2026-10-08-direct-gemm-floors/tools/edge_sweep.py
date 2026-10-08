"""MatmulGap H33: output hashes at row shapes with partial tiles. Rerun-safe (skips finished cells).

usage: edge_sweep.py <out.jsonl>   (run under one arm's venv; compare arms with shape_sweep_cmp.py)
m = 513 / 1537 / 4097 leave partial 64-row tiles, n = 4098 leaves a partial 64-column (and 128-column) tile; every cell has
n >= 4096 and m above the 512 floors, so the direct rows of every chip can fire.
"""
import hashlib
import itertools
import json
import sys

import mlx.core as mx
import numpy as np

out = sys.argv[1]
try:
    with open(out) as f:
        done = {json.loads(x)["id"] for x in f if x.strip()}
except FileNotFoundError:
    done = set()

DT = {"f16": mx.float16, "bf16": mx.bfloat16, "f32": mx.float32}
MS = (513, 1000, 1537, 4097)
NS = (4096, 4098, 6144)
KS = (2048, 4104)
ORI = ("nn", "nt", "tn")


def normal(rng, *shape):
    return mx.array(rng.standard_normal(shape, dtype=np.float32))


with open(out, "a") as f:
    print(json.dumps({"k": "host", "mlx": mx.__version__, "device": mx.device_info().get("device_name")}), file=f, flush=True)
    for name, ori, m, n, k in itertools.product(DT, ORI, MS, NS, KS):
        cid = f"{name}:{ori}:{m}x{n}x{k}"
        if cid in done:
            continue
        rng = np.random.default_rng(int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
        a = normal(rng, k, m) if ori == "tn" else normal(rng, m, k)
        b = normal(rng, n, k) if ori == "nt" else normal(rng, k, n)
        a, b = a.astype(DT[name]), b.astype(DT[name])
        aa = a.T if ori == "tn" else a
        bb = b.T if ori == "nt" else b
        c = aa @ bb
        mx.eval(c)
        raw = np.array(c.astype(mx.float32)).tobytes()
        print(json.dumps({"k": "cell", "id": cid, "sha": hashlib.sha256(raw).hexdigest()[:16]}), file=f, flush=True)
