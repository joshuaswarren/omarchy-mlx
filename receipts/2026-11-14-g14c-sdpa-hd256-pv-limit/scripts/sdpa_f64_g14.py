# Width-256 split-K decode SDPA accuracy gate: MLX_OMARCHY_SDPA_SPLIT_PV 0 (stock pass 1) and 1 (value-phase pass 1), each against a
# float64 numpy reference computed from the same bf16 inputs. Random bf16 q, k, v sliced from a larger buffer (the KV-cache layout),
# split route forced (DECODE_SPLIT=1). Two score scales: sigma 0.5 (flat softmax) and sigma 2.0 (peaked softmax).
# Output per row: max abs error of each arm against f64, and the max abs difference between the arms.
import json
import os
import sys

import numpy as np
import mlx.core as mx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_provenance import installed_provenance as _installed_provenance

_PROV = _installed_provenance()
print("PROVENANCE " + json.dumps(_PROV, sort_keys=True), flush=True)
if _PROV["verified"] == "mismatch":
    raise SystemExit("provenance refusal: " + str(_PROV["mismatch"]))

FLAG = "MLX_OMARCHY_SDPA_SPLIT_PV"
os.environ["MLX_OMARCHY_SDPA_DECODE_SPLIT"] = "1"
HD = 256
CONFIGS = [(8, 2, n) for n in (512, 1024, 2048, 2176, 4096)] + [(16, 4, n) for n in (2048, 4096)] + [(4, 1, 4096)]


def ref64(q, k, v, scale):
    qh, kvh = q.shape[1], k.shape[1]
    g = qh // kvh
    qf = q.astype(np.float64)[0, :, 0, :]
    out = np.empty((qh, HD))
    for h in range(qh):
        kk = k.astype(np.float64)[0, h // g]
        vv = v.astype(np.float64)[0, h // g]
        s = kk @ qf[h] * scale
        p = np.exp(s - s.max())
        out[h] = (p / p.sum()) @ vv
    return out


for sigma in (0.5, 2.0):
    for qh, kvh, n in CONFIGS:
        mx.random.seed(11)
        cap = n + 256
        q = (mx.random.normal((1, qh, 1, HD)) * sigma).astype(mx.bfloat16)
        kb = (mx.random.normal((1, kvh, cap, HD)) * sigma).astype(mx.bfloat16)
        vb = (mx.random.normal((1, kvh, cap, HD)) * 0.5).astype(mx.bfloat16)
        k, v = kb[..., :n, :], vb[..., :n, :]
        mx.eval(q, k, v)
        scale = HD ** -0.5
        ref = ref64(np.array(q.astype(mx.float32)), np.array(k.astype(mx.float32)), np.array(v.astype(mx.float32)), scale)
        got = {}
        for arm in ("0", "1"):
            os.environ[FLAG] = arm
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale).astype(mx.float32)
            mx.eval(o)
            got[arm] = np.array(o)[0, :, 0, :].astype(np.float64)
        print("ROW", json.dumps({"sigma": sigma, "qh": qh, "kvh": kvh, "keys": n, "workgroups": kvh * ((n + 127) // 128),
                                 "err_stock_vs_f64": float(np.abs(got["0"] - ref).max()),
                                 "err_variant_vs_f64": float(np.abs(got["1"] - ref).max()),
                                 "arms_maxabs_diff": float(np.abs(got["0"] - got["1"]).max()),
                                 "ref_maxabs": float(np.abs(ref).max())}), flush=True)
print("FUSE-DONE", flush=True)
os._exit(0)
