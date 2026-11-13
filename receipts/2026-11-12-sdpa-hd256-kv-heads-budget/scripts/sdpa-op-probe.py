# SDPA hd256 value-phase pass 1 (MLX_OMARCHY_SDPA_SPLIT_PV), op level: mx.fast.scaled_dot_product_attention on random bf16
# q (1,8,1,256), k and v (1,2,N,256) sliced from a larger buffer (the KV-cache layout), split route forced (DECODE_SPLIT=1).
# Arms "0" (old pass 1) and "1" (value-phase wide) interleaved in ABBA blocks, 36-call chains; engagement = max abs output difference.
import os, time, random, statistics as st
import mlx.core as mx

# Provenance gate (scripts/mlx_provenance.py, shipped beside this probe): the first line of every row log names the
# libmlx.so actually loaded; a mismatch against the wheel RECORD refuses to emit a number.
import json as _json, sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlx_provenance import installed_provenance as _installed_provenance
_PROV = _installed_provenance()
print("PROVENANCE " + _json.dumps(_PROV, sort_keys=True), flush=True)
if _PROV["verified"] == "mismatch":
    raise SystemExit("provenance refusal: " + str(_PROV["mismatch"]))

N = int(os.environ.get("SDPA_CTX", "480"))
REPS = int(os.environ.get("SDPA_NTOK", "128"))
CHAIN = 36
FLAG = "MLX_OMARCHY_SDPA_SPLIT_PV"
os.environ["MLX_OMARCHY_SDPA_DECODE_SPLIT"] = "1"
mx.random.seed(3)
cap = N + 256
q = (mx.random.normal((1, 8, 1, 256)) * 0.5).astype(mx.bfloat16)
kb = (mx.random.normal((1, 2, cap, 256)) * 0.5).astype(mx.bfloat16)
vb = (mx.random.normal((1, 2, cap, 256)) * 0.5).astype(mx.bfloat16)
k, v = kb[..., :N, :], vb[..., :N, :]
mx.eval(q, kb, vb)
scale = 256 ** -0.5

def one():
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)

outs = {}
for arm in ("0", "1"):
    os.environ[FLAG] = arm
    o = one().astype(mx.float32)
    mx.eval(o)
    outs[arm] = o
diff = float(mx.max(mx.abs(outs["0"] - outs["1"])))

def chain():
    x = q
    for _ in range(CHAIN):
        x = mx.fast.scaled_dot_product_attention(x, k, v, scale=scale)
    mx.eval(x)

for arm in ("0", "1"):
    os.environ[FLAG] = arm
    chain()
times = {"0": [], "1": []}
for i in range(REPS):
    arm = "01"[i % 2] if (i // 2) % 2 == 0 else "10"[i % 2]
    os.environ[FLAG] = arm
    t0 = time.perf_counter()
    chain()
    times[arm].append((time.perf_counter() - t0) * 1e6 / CHAIN)

def boot(a, b, n=2000):
    r = random.Random(7)
    g = []
    for _ in range(n):
        ma = st.median(r.choices(a, k=len(a)))
        mb = st.median(r.choices(b, k=len(b)))
        g.append(100 * (ma - mb) / ma)
    g.sort()
    return round(g[int(0.025 * n)], 2), round(g[int(0.975 * n)], 2)

m0, m1 = st.median(times["0"]), st.median(times["1"])
print({"keys": N, "reps_per_arm": len(times["0"]), "engaged_op_maxabs_diff": diff,
       "stock_us_per_call": round(m0, 2), "variant_us_per_call": round(m1, 2),
       "gain_pct": round(100 * (m0 - m1) / m0, 2), "gain_ci95": boot(times["0"], times["1"])})
print("FUSE-DONE")
