# Float64 gate for the value-phase pass 1 at op level: random bf16 q (1,QH,1,HD), K and V (1,KVH,cap,HD) sliced to SDPA_CTX keys (KV-cache layout),
# split route forced; the flag-off arm (old pass 1), the flag-on arm (variant) and the fused kernel are each compared with a float64 reference on the same inputs.
# Same bounds as check_shape in test_sdpa_split.cpp: max error <= stock max error + 2 half steps (half step = max |ref| * 2^-8), mean error <= stock mean + half/2, max error <= 3 half steps.
import os
import numpy as np
import mlx.core as mx

N = int(os.environ.get("SDPA_CTX", "1024"))
QH = int(os.environ.get("SDPA_QH", "8"))
KVH = int(os.environ.get("SDPA_KVH", "2"))
HD = int(os.environ.get("SDPA_HD", "256"))
AMP = float(os.environ.get("SDPA_AMP", "1.0"))
FLAG = "MLX_OMARCHY_SDPA_SPLIT_PV"
mx.random.seed(5)
cap = N + 128
q = (mx.random.normal((1, QH, 1, HD)) * AMP).astype(mx.bfloat16)
kb = (mx.random.normal((1, KVH, cap, HD)) * AMP).astype(mx.bfloat16)
vb = (mx.random.normal((1, KVH, cap, HD)) * AMP).astype(mx.bfloat16)
k, v = kb[..., :N, :], vb[..., :N, :]
mx.eval(q, kb, vb)
scale = HD ** -0.5

def run(split, arm):
    os.environ["MLX_OMARCHY_SDPA_DECODE_SPLIT"] = split
    os.environ[FLAG] = arm
    o = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale).astype(mx.float32)
    mx.eval(o)
    return np.array(o).astype(np.float64)[0, :, 0, :]

fused = run("0", "0")
stock = run("1", "0")
variant = run("1", "1")
g = QH // KVH
qn = np.array(q.astype(mx.float32)).astype(np.float64)[0, :, 0, :]
kn = np.array(k.astype(mx.float32)).astype(np.float64)[0]
vn = np.array(v.astype(mx.float32)).astype(np.float64)[0]
ref = np.zeros((QH, HD))
for h in range(QH):
    sc = kn[h // g] @ qn[h] * scale
    p = np.exp(sc - sc.max())
    ref[h] = (p / p.sum()) @ vn[h // g]
half = float(np.abs(ref).max()) * 2 ** -8
e = {n: np.abs(a - ref) for n, a in (("fused", fused), ("stock", stock), ("variant", variant))}
ok = bool(e["variant"].max() <= e["stock"].max() + 2 * half and e["variant"].mean() <= e["stock"].mean() + half / 2 and e["variant"].max() <= 3 * half)
print({"keys": N, "qh": QH, "kvh": KVH, "hd": HD, "half_step": half,
       "err_max": {n: float(x.max()) for n, x in e.items()}, "err_mean": {n: float(x.mean()) for n, x in e.items()},
       "arms_maxabs_diff": float(np.abs(variant - stock).max()), "f64_gate_pass": ok})
print("FUSE-DONE")
