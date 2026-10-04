#!/usr/bin/env python3
"""Free-run route comparison with per-divergence composed top-2 gap logging.
Route OFF (composed) vs route ON (fused). For each prompt: identity, first
divergence position, and the COMPOSED path's top-2 logit gap at that
position (H257 near-tie rule: flag if gap >= 0.05).
usage: free_run_gaps.py <model-dir> <out.json> [n-prompts=10] [tokens=512]
"""
import glob
import json
import os
import sys

import mlx.core as mx

model_dir = sys.argv[1]
out_path = sys.argv[2]
nprompts = int(sys.argv[3]) if len(sys.argv) > 3 else 10
ntoks = int(sys.argv[4]) if len(sys.argv) > 4 else 512

prompts_path = os.environ.get("H257_PROMPTS10", "/var/tmp/Jw16NumericsGpu/qwen38-2b-prompts-10.jsonl")
texts = [json.loads(l)["text"] for l in open(prompts_path) if l.strip()][:nprompts]

from mlx_lm import load  # noqa: E402

model, tokenizer = load(model_dir)


def generate(env_on):
    if env_on:
        os.environ.pop("MLX_OMARCHY_GDN_RAW_REPEAT", None)
    else:
        os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
    from mlx_lm.models.cache import make_prompt_cache
    rows = []
    for text in texts:
        ids = tokenizer.encode(text)
        cache = make_prompt_cache(model)
        logits = model(mx.array([ids], dtype=mx.int32), cache=cache)
        toks = []
        gaps = []
        for _ in range(ntoks):
            last = logits[:, -1, :].astype(mx.float32)
            top = mx.sort(mx.flatten(last))[-2:]
            gaps.append(float(top[1] - top[0]))
            t = int(mx.argmax(last, axis=-1).item())
            toks.append(t)
            logits = model(mx.array([[t]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        rows.append({"tokens": toks, "composed_top2_gaps": gaps})
        print(".", end="", flush=True)
    print(flush=True)
    return rows


off = generate(False)
on = generate(True)
res = {"prompts": nprompts, "tokens": ntoks, "rows": []}
flags = 0
for i, (a, b) in enumerate(zip(off, on)):
    first = next((j for j, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None)
    row = {"prompt_index": i, "identical": 512 if first is None else first,
           "pct": round(100.0 * (512 if first is None else first) / 512, 2),
           "first_divergence": first}
    if first is not None:
        g = a["composed_top2_gaps"][first]
        row["composed_gap_at_divergence"] = g
        row["near_tie_lt_0.05"] = bool(g < 0.05)
        if g >= 0.05:
            flags += 1
    res["rows"].append(row)
identities = [r["pct"] for r in res["rows"]]
res["mean_identity_pct"] = round(sum(identities) / len(identities), 2)
res["diverging_prompts"] = sum(1 for r in res["rows"] if r["first_divergence"] is not None)
res["divergences_with_gap_ge_0.05"] = flags
greedy_pass = all(r["pct"] >= 95.0 for r in res["rows"])
res["greedy_95pct_bar"] = greedy_pass
near_tie_only = all(r.get("near_tie_lt_0.05", True) for r in res["rows"])
res["all_divergences_near_tie"] = near_tie_only
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps({k: res[k] for k in ["mean_identity_pct", "diverging_prompts",
                                      "divergences_with_gap_ge_0.05",
                                      "greedy_95pct_bar", "all_divergences_near_tie"]}, indent=1))
