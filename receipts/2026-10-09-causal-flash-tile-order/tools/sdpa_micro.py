"""SDPA prefill micro-benchmark on the Qwen3-4B attention shape: B=1, Hq=32, Hkv=8 (GQA x4), hd=128, bf16, L in {512, 1024, 2048}.
mode: causal | full ; the route (composed vs flash) is chosen by the environment of THIS process (MLX_OMARCHY_SDPA_CAUSAL_FLASH=1 selects the causal flash route; unset is the composed route).
Prints one JSON line: ms per call (median of 5 reps x 20 calls), plus the output hash and max|diff| vs an fp32 numpy reference for L=512 (one head)."""
import json
import os
import sys
import time

import mlx.core as mx
import numpy as np

mode = sys.argv[1]
Ls = [int(x) for x in sys.argv[2].split(",")]
SUBMIT_LIMIT_S = 20.0  # fleet rule: no single GPU submit above 20 s
Hq, Hkv, HD = 32, 8, 128
res = {"mode": mode, "flash_min_l": os.environ.get("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L"), "flash_env": os.environ.get("MLX_OMARCHY_SDPA_PREFILL_FLASH"), "L": {}}
rng = np.random.default_rng(0)
for L in Ls:
    q = mx.array(rng.standard_normal((1, Hq, L, HD)).astype(np.float32)).astype(mx.bfloat16)
    k = mx.array(rng.standard_normal((1, Hkv, L, HD)).astype(np.float32)).astype(mx.bfloat16)
    v = mx.array(rng.standard_normal((1, Hkv, L, HD)).astype(np.float32)).astype(mx.bfloat16)
    mx.eval(q, k, v)
    scale = HD ** -0.5
    mask = "causal" if mode == "causal" else None

    def run():
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
    for _ in range(3):
        mx.eval(run())
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(20):
            t1 = time.perf_counter()
            o = run()
            mx.eval(o)
            if time.perf_counter() - t1 > SUBMIT_LIMIT_S:
                raise SystemExit(f'ABORT L={L}: one eval over the {SUBMIT_LIMIT_S:.0f} s submit limit')
        times.append((time.perf_counter() - t0) / 20 * 1000)
    o = run()
    mx.eval(o)
    on = np.array(o.astype(mx.float32))
    # fp32 numpy reference for head 0 (kv head 0), all rows
    qn = np.array(q.astype(mx.float32))[0, 0]
    kn = np.array(k.astype(mx.float32))[0, 0]
    vn = np.array(v.astype(mx.float32))[0, 0]
    s = (qn @ kn.T) * scale
    if mode == "causal":
        s = np.where(np.tril(np.ones((L, L), bool)), s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    ref = p @ vn
    err = np.abs(on[0, 0] - ref)
    res["L"][str(L)] = {"ms_median": round(float(np.median(times)), 4), "ms_min": round(float(min(times)), 4), "ms_all": [round(float(x), 3) for x in times],
                        "max_err_head0": float(err.max()), "mean_err_head0": float(err.mean())}
print(json.dumps(res))
