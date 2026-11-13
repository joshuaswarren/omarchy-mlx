import os
# engagement variant: adds a REAL-CALL line (the model's own attention call under both flag values, float64 error, per-call time); SDPA_ENGAGE_ONLY=1 skips the model A/B.
# chunked-prefill variant of sdpa-hd256-probe.py (128-token prefill chunks, same decode A/B)
# SDPA hd256 value-phase pass 1 (MLX_OMARCHY_SDPA_SPLIT_PV) A/B on Qwen3.8-2B (head width 256), arms interleaved step by step in ONE process.
# The flag is read live in eval_gpu, so os.environ is flipped between decode steps; the token fed is the same fixed id for both arms.
# Engagement proof: logits after the same prefill and one step differ between the arms (the variant adds partial sums in another order).
import glob, json, os, random, statistics as st, sys, time
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
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

CTX = int(os.environ.get("SDPA_CTX", "480"))
NTOK = int(os.environ.get("SDPA_NTOK", "256"))
FLAG = "MLX_OMARCHY_SDPA_SPLIT_PV"
P = os.environ.get("SDPA_MODEL_DIR") or glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--SiddhJagani--Qwen3.8-2B-mlx-4Bit/snapshots/*"))[0]
model, tok = load(P)

# Engagement proof at the model's own attention call: capture the first width-256 single-query call of a decode step (real q, K, V views of the cache),
# then run that exact call under both flag values: output difference, float64 error of each arm, and per-call time.
import numpy as np
_orig_sdpa = mx.fast.scaled_dot_product_attention
_cap = {"on": False, "call": None}
def _wrapped(q, k, v, *a, **kw):
    if _cap["on"] and _cap["call"] is None and q.shape[-1] == 256 and q.shape[2] == 1:
        _cap["call"] = (q, k, v, a, dict(kw))
    return _orig_sdpa(q, k, v, *a, **kw)
mx.fast.scaled_dot_product_attention = _wrapped

ids = (tok.encode("hello world " * 400) * 8)[: CTX + NTOK + 8]
prompt = mx.array(ids[:CTX])[None]

def prefill():
    cache = make_prompt_cache(model)
    # 128-token chunks: a one-shot prefill of CTX positions allocates CTX x vocab logits (384 MiB at 768) and the 9B on a 15 GB M1 ran out of device memory.
    for lo in range(0, CTX, 128):
        out = model(prompt[:, lo:lo + 128], cache=cache)
        mx.eval(out)
    return cache, out

def step(cache, token_id):
    out = model(mx.array([[token_id]]), cache=cache)
    mx.eval(out)
    return out


def boot(a, b, n=2000):
    r = random.Random(7)
    gg = []
    for _ in range(n):
        ma = st.median(r.choices(a, k=len(a)))
        mb = st.median(r.choices(b, k=len(b)))
        gg.append(100 * (ma - mb) / ma)
    gg.sort()
    return round(gg[int(0.025 * n)], 2), round(gg[int(0.975 * n)], 2)

def real_call_check():
    cache2, _ = prefill()
    _cap["on"] = True
    os.environ.pop(FLAG, None)
    step(cache2, ids[CTX])
    _cap["on"] = False
    q, k, v, a, kw = _cap["call"]
    mx.eval(q, k, v)
    scale = kw.get("scale", 256 ** -0.5)
    outs = {}
    for arm in ("0", "1"):
        os.environ[FLAG] = arm
        o = _orig_sdpa(q, k, v, *a, **kw).astype(mx.float32)
        mx.eval(o)
        outs[arm] = np.array(o).astype(np.float64)
    qh, kvh, keys = q.shape[1], k.shape[1], k.shape[2]
    grp = qh // kvh
    qn = np.array(q.astype(mx.float32)).astype(np.float64)[0, :, 0, :]
    kn = np.array(k.astype(mx.float32)).astype(np.float64)[0]
    vn = np.array(v.astype(mx.float32)).astype(np.float64)[0]
    ref = np.zeros((qh, 256))
    for h in range(qh):
        sc = kn[h // grp] @ qn[h] * scale
        p = np.exp(sc - sc.max())
        ref[h] = (p / p.sum()) @ vn[h // grp]
    e0 = np.abs(outs["0"][0, :, 0, :] - ref)
    e1 = np.abs(outs["1"][0, :, 0, :] - ref)
    def chain(arm):
        os.environ[FLAG] = arm
        x = q
        for _ in range(36):
            x = _orig_sdpa(x, k, v, *a, **kw)
        mx.eval(x)
    for arm in ("0", "1"):
        chain(arm)
    t = {"0": [], "1": []}
    for i in range(400):
        arm = "01"[i % 2] if (i // 2) % 2 == 0 else "10"[i % 2]
        t0 = time.perf_counter()
        chain(arm)
        t[arm].append((time.perf_counter() - t0) * 1e6 / 36)
    m0, m1 = st.median(t["0"]), st.median(t["1"])
    lo, hi = boot(t["0"], t["1"])
    half = float(np.abs(ref).max()) * 2 ** -8
    return {"keys": int(keys), "q_heads": int(qh), "kv_heads": int(kvh),
            "arms_maxabs_diff": float(np.abs(outs["0"] - outs["1"]).max()),
            "f64_err_max_stock": float(e0.max()), "f64_err_max_variant": float(e1.max()),
            "f64_err_mean_stock": float(e0.mean()), "f64_err_mean_variant": float(e1.mean()),
            "bf16_half_step": half,
            "f64_gate_pass": bool(e1.max() <= e0.max() + 2 * half and e1.mean() <= e0.mean() + half / 2 and e1.max() <= 3 * half),
            "stock_us_per_call": round(m0, 2), "variant_us_per_call": round(m1, 2),
            "gain_pct": round(100 * (m0 - m1) / m0, 2), "gain_ci95": [lo, hi]}

if os.environ.get("SDPA_ENGAGE_ONLY") == "1":
    print("REAL-CALL " + json.dumps(real_call_check()))
    print("FUSE-DONE")
    sys.exit(0)

# engagement: same prefill, one step, flag 0 vs 1
logits = {}
for arm in ("0", "1"):
    os.environ[FLAG] = arm
    cache, _ = prefill()
    logits[arm] = step(cache, ids[CTX]).astype(mx.float32)
    mx.eval(logits[arm])
diff = float(mx.max(mx.abs(logits["0"] - logits["1"])))
same_argmax = bool(mx.array_equal(mx.argmax(logits["0"], axis=-1), mx.argmax(logits["1"], axis=-1)))

cache, _ = prefill()
for arm in ("0", "1"):  # warm both arms
    os.environ[FLAG] = arm
    step(cache, ids[CTX])
times = {"0": [], "1": []}
for i in range(NTOK):
    arm = "01"[i % 2] if (i // 2) % 2 == 0 else "10"[i % 2]  # ABBA blocks cancel a linear drift
    os.environ[FLAG] = arm
    t0 = time.perf_counter()
    step(cache, ids[CTX + 1 + i])
    times[arm].append((time.perf_counter() - t0) * 1e3)

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
lo, hi = boot(times["0"], times["1"])
print(json.dumps({"ctx": CTX, "keys_max": CTX + NTOK, "n_per_arm": len(times["0"]), "engaged_logit_maxabs_diff": diff,
                  "same_argmax": same_argmax, "stock_ms_med": round(m0, 3), "variant_ms_med": round(m1, 3),
                  "gain_pct": round(100 * (m0 - m1) / m0, 2), "gain_ci95": [lo, hi]}))
print("REAL-CALL " + json.dumps(real_call_check()))
print("FUSE-DONE")
