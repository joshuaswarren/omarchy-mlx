#!/usr/bin/env python3
"""Free-running greedy identity for the order-matched fused GDU: several
prompts at d512, route OFF (composed, =0) vs route ON (default). Writes a
per-prompt identity table. Runs on ANY wheel with both routes.
usage: free_run_identity.py <model-dir> <out.json> <n-prompts>
"""
import glob
import json
import os
import sys

import mlx.core as mx

model_dir = sys.argv[1]
out_path = sys.argv[2]
nprompts = int(sys.argv[3]) if len(sys.argv) > 3 else 5

prompts_path = "/var/tmp/Jw16NumericsGpu/qwen38-2b-prompts-10.jsonl"
texts = [json.loads(l)["text"] for l in open(prompts_path) if l.strip()][:nprompts]


def generate(env_on, tokenizer, model):
    if env_on:
        os.environ.pop("MLX_OMARCHY_GDN_RAW_REPEAT", None)  # default ON
    else:
        os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
    from mlx_lm.models.cache import make_prompt_cache
    rows = []
    for text in texts:
        ids = tokenizer.encode(text)
        cache = make_prompt_cache(model)
        logits = model(mx.array([ids], dtype=mx.int32), cache=cache)
        toks = []
        for _ in range(512):
            t = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            toks.append(t)
            logits = model(mx.array([[t]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        rows.append(toks)
        print(".", end="", flush=True)
    print(flush=True)
    return rows


from mlx_lm import load  # noqa: E402
model, tokenizer = load(model_dir)
off = generate(False, tokenizer, model)
on = generate(True, tokenizer, model)
res = {"prompts": nprompts, "tokens": 512, "rows": []}
for i, (a, b) in enumerate(zip(off, on)):
    same = sum(1 for x, y in zip(a, b) if x == y)
    first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), None)
    res["rows"].append({"prompt_index": i, "identical": same, "of": len(a),
                        "pct": round(100.0 * same / len(a), 2), "first_divergence": first})
res["mean_identity_pct"] = round(sum(r["pct"] for r in res["rows"]) / len(res["rows"]), 2)
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res["rows"], indent=1))
print("mean identity:", res["mean_identity_pct"], "%")
