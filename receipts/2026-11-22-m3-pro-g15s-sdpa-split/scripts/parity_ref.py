"""Teacher-forced decode log-probs of every bf16 attention route against a float32 reference, one model, one process.

usage: parity_ref.py MODEL ARMS OUT.json EXPECT_WHEEL REPO_ROOT TEXT_FILE

ARMS is a comma-separated list of: default (SDPA variables unset), composition (MLX_OMARCHY_SDPA_DECODE_NATIVE=0: the unfused
matmul-softmax-matmul decode), split, pv, bk64 (the split-K route forced, plain, value-phase, 64-key blocks). The routes read their
variables on every call, so one process runs every arm. For each context length in CTX (environment CTX; default 443,1024,2048,4000), the
prompt is the first CTX tokens of TEXT_FILE and the next FORCE tokens are fed one at a time on the prompt cache (one query token per
step). Reference: the same model with its floating parameters cast to float32 (model.set_dtype), so attention and every activation run in
float32. Per arm: mean and max KL(reference || arm) over the steps, the steps whose argmax equals the reference argmax, and the largest
|dlogprob| of the reference's top token; and the same between every pair of bf16 arms (KL(default || arm)).
Prints the scripts/mlx_provenance.py line and refuses (exit 3) unless the installed wheel matches EXPECT_WHEEL.
"""
import gc
import json
import os
import sys
from pathlib import Path

mdl, arms_arg, out_json, expect_wheel, repo_root, text_file = sys.argv[1:7]
sys.path.insert(0, str(Path(repo_root) / "scripts"))
import mlx_provenance as mp  # noqa: E402

prov = mp.installed_provenance(expect_wheel=Path(expect_wheel))
pline = mp.provenance_line(prov)
print(pline, flush=True)
if prov["verified"] != "match" or prov["version_match"] is not True:
    sys.exit(3)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

ARMS = {
    "default": {},
    "composition": {"MLX_OMARCHY_SDPA_DECODE_NATIVE": "0"},
    "split": {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "0", "MLX_OMARCHY_SDPA_SPLIT_BK64": "0"},
    "pv": {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1", "MLX_OMARCHY_SDPA_SPLIT_BK64": "0"},
    "bk64": {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1", "MLX_OMARCHY_SDPA_SPLIT_BK64": "1"},
}
arms = arms_arg.split(",")
assert not any(k.startswith("MLX_OMARCHY_SDPA") for k in os.environ), "start with the SDPA variables unset"
KEYS = sorted({k for a in ARMS.values() for k in a})


def arm(name):
    for k in KEYS:
        os.environ.pop(k, None)
    os.environ.update(ARMS[name])


CTX = tuple(int(x) for x in os.environ.get("CTX", "443,1024,2048,4000").split(","))
FORCE = 32
model, tok = load(mdl)
text_ids = tok.encode(open(text_file).read())
assert len(text_ids) >= max(CTX) + FORCE


def forced(ctx):
    cache = make_prompt_cache(model)
    for i in range(0, ctx, 512):  # chunked prefill: the float32 reference does not fit 4000 tokens in one pass
        logits = model(mx.array(text_ids[i:min(i + 512, ctx)])[None], cache=cache)
        mx.eval(logits)
    lps = []
    for t in text_ids[ctx:ctx + FORCE]:
        lp = logits[0, -1].astype(mx.float32)
        lp = lp - mx.logsumexp(lp)
        mx.eval(lp)
        lps.append(np.array(lp).astype(np.float64))
        logits = model(mx.array([[t]]), cache=cache)
    mx.eval(logits)
    del cache, logits
    gc.collect()
    mx.clear_cache()
    return np.stack(lps)


def cmp(a, b):
    kl = (np.exp(a) * (a - b)).sum(-1)
    top = a.argmax(-1)
    return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()), "argmax_equal": int((top == b.argmax(-1)).sum()),
            "max_abs_dlogprob_top1": float(np.abs(a - b)[np.arange(len(top)), top].max())}


lp = {a: {} for a in arms}
for a in arms:
    arm(a)
    for ctx in CTX:
        lp[a][ctx] = forced(ctx)
arm("default")
model.set_dtype(mx.float32)
mx.eval(model.parameters())
ref = {ctx: forced(ctx) for ctx in CTX}
rows = []
for ctx in CTX:
    row = {"ctx": ctx, "steps": FORCE, "vs_float32": {a: cmp(ref[ctx], lp[a][ctx]) for a in arms},
           "vs_default": {a: cmp(lp["default"][ctx], lp[a][ctx]) for a in arms if a != "default"} if "default" in arms else {}}
    rows.append(row)
    print("REF", json.dumps(row), flush=True)
json.dump({"provenance": pline, "model": Path(mdl).name, "arms": {a: ARMS[a] for a in arms}, "rows": rows,
           "load1_end": os.getloadavg()[0]}, open(out_json, "w"), indent=1)
print("DONE", flush=True)
