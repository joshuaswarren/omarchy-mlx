#!/usr/bin/env python3
"""Compare composed and fused free runs and measure first-divergence gap in bf16 ULPs.
usage: free_run_gaps.py <model-dir> <out.json> [n-prompts=10] [tokens=512]
"""
import json
import os
import sys

import mlx.core as mx

from bf16_ulp import bf16_ulp, within_one_bf16_ulp

model_dir = sys.argv[1]
out_path = sys.argv[2]
nprompts = int(sys.argv[3]) if len(sys.argv) > 3 else 10
ntoks = int(sys.argv[4]) if len(sys.argv) > 4 else 512
prompts_path = os.environ.get("H257_PROMPTS10", "/var/tmp/Jw16NumericsGpu/qwen38-2b-prompts-10.jsonl")
texts = [json.loads(line)["text"] for line in open(prompts_path) if line.strip()][:nprompts]

from mlx_lm import load  # noqa: E402

model, tokenizer = load(model_dir)


def generate(fused):
    if fused:
        os.environ.pop("MLX_OMARCHY_GDN_RAW_REPEAT", None)
    else:
        os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
    from mlx_lm.models.cache import make_prompt_cache

    rows = []
    for text in texts:
        ids = tokenizer.encode(text)
        cache = make_prompt_cache(model)
        logits = model(mx.array([ids], dtype=mx.int32), cache=cache)
        tokens, gaps, magnitudes = [], [], []
        for _ in range(ntoks):
            last = logits[:, -1, :].astype(mx.float32)
            top = mx.sort(mx.flatten(last))[-2:]
            gaps.append(float(top[1] - top[0]))
            magnitudes.append(abs(float(top[1])))
            token = int(mx.argmax(last, axis=-1).item())
            tokens.append(token)
            logits = model(mx.array([[token]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        rows.append({"tokens": tokens, "composed_top2_gaps": gaps,
                     "composed_top1_magnitudes": magnitudes})
        print(".", end="", flush=True)
    print(flush=True)
    return rows


composed = generate(False)
fused = generate(True)
res = {"prompts": nprompts, "tokens": ntoks,
       "composed_rows": [{"prompt_index": i, "composed_tokens": row["tokens"]}
                         for i, row in enumerate(composed)], "rows": []}
matched_total = 0
for i, (base, candidate) in enumerate(zip(composed, fused)):
    first = next((j for j, (x, y) in enumerate(zip(base["tokens"], candidate["tokens"]))
                  if x != y), None)
    matches = sum(x == y for x, y in zip(base["tokens"], candidate["tokens"]))
    matched_total += matches
    prefix = ntoks if first is None else first
    row = {"prompt_index": i, "identical": prefix,
           "pct": round(100.0 * prefix / ntoks, 2),
           "exact_match_pct": round(100.0 * matches / ntoks, 2),
           "first_divergence": first}
    if first is not None:
        gap = base["composed_top2_gaps"][first]
        magnitude = base["composed_top1_magnitudes"][first]
        ulp = bf16_ulp(magnitude)
        row.update({"composed_gap_at_divergence": gap,
                    "composed_top1_magnitude_at_divergence": magnitude,
                    "fused_gap_at_divergence": candidate["composed_top2_gaps"][first],
                    "composed_gap_bf16_ulp": ulp,
                    "composed_gap_in_bf16_ulps": gap / ulp,
                    "within_one_bf16_ulp": within_one_bf16_ulp(gap, magnitude)})
    res["rows"].append(row)
identities = [row["pct"] for row in res["rows"]]
res["mean_identity_pct"] = round(sum(identities) / len(identities), 2) if identities else 0.0
res["exact_match_pct_overall"] = round(100.0 * matched_total / (ntoks * len(composed)), 3) if ntoks and composed else 0.0
res["diverging_prompts"] = sum(row["first_divergence"] is not None for row in res["rows"])
res["all_divergences_within_one_bf16_ulp"] = all(
    row.get("within_one_bf16_ulp", True) for row in res["rows"])
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps({key: res[key] for key in ("mean_identity_pct", "diverging_prompts",
                                           "exact_match_pct_overall",
                                           "all_divergences_within_one_bf16_ulp")}, indent=1))
