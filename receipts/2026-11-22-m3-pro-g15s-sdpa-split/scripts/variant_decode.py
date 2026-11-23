"""Long-context greedy decode, the default route against one forced split-K variant, in one process.

usage: variant_decode.py MODEL VARIANT OUT.json EXPECT_WHEEL REPO_ROOT TEXT_FILE

VARIANT is split, pv or bk64 (the MLX_OMARCHY_SDPA_* values below). The decode routes read their variables on every call, so one
process switches between the default (variables unset) and the variant. For each context length in CTX (environment CTX,
comma-separated; default 443,1024,2048,4000), with the prompt the first CTX tokens of TEXT_FILE:
  - speed and tokens: one warm-up generation, then ROUNDS rounds of default then variant, GEN greedy tokens each
    (mlx_lm.stream_generate, generation_tps, prompt_tps); token-id sha per generation; Vulkan compute dispatches per token per arm;
  - logits: teacher-forced decode of the next FORCE tokens of the text on one prompt cache per arm (one query token per step);
    per arm the full-vocabulary log-softmax of every step, then largest |dlogprob|, KL(default || variant) and argmax agreement.
Prints the scripts/mlx_provenance.py line and refuses (exit 3) unless the installed wheel matches EXPECT_WHEEL.
"""
import ctypes
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

mdl, variant, out_json, expect_wheel, repo_root, text_file = sys.argv[1:7]
sys.path.insert(0, str(Path(repo_root) / "scripts"))
import mlx_provenance as mp  # noqa: E402

prov = mp.installed_provenance(expect_wheel=Path(expect_wheel))
pline = mp.provenance_line(prov)
print(pline, flush=True)
if prov["verified"] != "match" or prov["version_match"] is not True:
    sys.exit(3)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

VARIANTS = {
    "split": {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "0", "MLX_OMARCHY_SDPA_SPLIT_BK64": "0"},
    "pv": {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1", "MLX_OMARCHY_SDPA_SPLIT_BK64": "0"},
    "bk64": {"MLX_OMARCHY_SDPA_DECODE_SPLIT": "1", "MLX_OMARCHY_SDPA_SPLIT_PV": "1", "MLX_OMARCHY_SDPA_SPLIT_BK64": "1"},
}
var_env = VARIANTS[variant]
assert not any(k.startswith("MLX_OMARCHY_SDPA") for k in os.environ), "start with the SDPA variables unset"


def arm(name):
    for k, v in var_env.items():
        if name == "default":
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


lib = ctypes.CDLL(os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so"))


class Snap(ctypes.Structure):
    _fields_ = [(n, ctypes.c_uint64) for n in (
        "primitives", "submissions", "copies", "fills", "compute", "finalize",
        "commit_work", "commit_noop", "barriers", "barriers_skipped")]


lib.mlx_omarchy_trace_snapshot_abi_size.restype = ctypes.c_uint64
assert lib.mlx_omarchy_trace_snapshot_abi_size() == ctypes.sizeof(Snap)


def compute_dispatches():
    s = Snap()
    lib.mlx_omarchy_trace_snapshot(ctypes.byref(s))
    return s.compute


CTX = tuple(int(x) for x in os.environ.get("CTX", "443,1024,2048,4000").split(","))
GEN, ROUNDS, FORCE = 128, 2, 32

model, tok = load(mdl)
text_ids = tok.encode(open(text_file).read())
assert len(text_ids) >= max(CTX) + FORCE, len(text_ids)
greedy = make_sampler(temp=0.0)


def generate(prompt):
    ids, last = [], None
    d0 = compute_dispatches()
    for resp in stream_generate(model, tok, prompt, max_tokens=GEN, sampler=greedy):
        ids.append(int(resp.token))
        last = resp
    mx.synchronize()
    return {"sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16], "tps": round(last.generation_tps, 3),
            "prefill_tps": round(last.prompt_tps, 1), "dispatches_per_token": round((compute_dispatches() - d0) / len(ids), 2),
            "ids": ids}


def forced(prompt, ctx):
    cache = make_prompt_cache(model)
    logits = model(mx.array(prompt)[None], cache=cache)
    lps = []
    for t in text_ids[ctx:ctx + FORCE]:
        lp = logits[0, -1].astype(mx.float32)
        lp = lp - mx.logsumexp(lp)
        mx.eval(lp)
        lps.append(np.array(lp).astype(np.float64))
        logits = model(mx.array([[t]]), cache=cache)
    mx.eval(logits)
    return np.stack(lps)


rows = []
for ctx in CTX:
    prompt = text_ids[:ctx]
    arm(variant)
    generate(prompt)  # warm-up
    gens = {"default": [], variant: []}
    for _ in range(ROUNDS):
        for a in ("default", variant):
            arm(a)
            gens[a].append(generate(prompt))
    lp = {}
    for a in ("default", variant):
        arm(a)
        lp[a] = forced(prompt, ctx)
    arm("default")
    a, b = lp["default"], lp[variant]
    kl = (np.exp(a) * (a - b)).sum(-1)
    first_diff = None
    ia, ib = gens["default"][0]["ids"], gens[variant][0]["ids"]
    if ia != ib:
        first_diff = next(i for i, (x, y) in enumerate(zip(ia, ib)) if x != y)
    row = {"ctx": ctx, "gen": GEN,
           "default": {"tps": [g["tps"] for g in gens["default"]], "prefill_tps": [g["prefill_tps"] for g in gens["default"]],
                       "sha": sorted({g["sha"] for g in gens["default"]}), "dispatches_per_token": gens["default"][0]["dispatches_per_token"]},
           variant: {"tps": [g["tps"] for g in gens[variant]], "prefill_tps": [g["prefill_tps"] for g in gens[variant]],
                     "sha": sorted({g["sha"] for g in gens[variant]}), "dispatches_per_token": gens[variant][0]["dispatches_per_token"]},
           "tokens_identical": ia == ib, "first_differing_token": first_diff,
           "forced_steps": FORCE, "max_abs_dlogprob": float(np.abs(a - b).max()),
           "max_abs_dlogprob_top1": float(np.abs(a - b)[np.arange(FORCE), a.argmax(-1)].max()),
           "kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
           "argmax_equal_steps": int((a.argmax(-1) == b.argmax(-1)).sum())}
    row["speedup_pct"] = round((statistics.median(row[variant]["tps"]) / statistics.median(row["default"]["tps"]) - 1) * 100, 2)
    rows.append(row)
    print("VAR", variant, json.dumps(row), flush=True)

json.dump({"provenance": pline, "model": Path(mdl).name, "variant": variant, "variant_env": var_env, "rows": rows,
           "load1_end": os.getloadavg()[0]}, open(out_json, "w"), indent=1)
print("DONE", variant, flush=True)
