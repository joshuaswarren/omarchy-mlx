#!/usr/bin/env python3
"""S=1 decode-route-sensitive PPL probe: feeds tokens ONE at a time so the
GDN decode kernel runs per token (the prefill-length pass is blind to the
decode route). Reports mean NLL per token for route OFF (composed) and ON
(fused). usage: ppl_s1.py <model-dir> <out.json> [n-prompts=4] [max-tokens=192]
"""
import json
import os
import sys

import mlx.core as mx

model_dir = sys.argv[1]
out_path = sys.argv[2]
nprompts = int(sys.argv[3]) if len(sys.argv) > 3 else 4
maxtok = int(sys.argv[4]) if len(sys.argv) > 4 else 192

prompts_path = os.environ.get("H257_PROMPTS10", "/var/tmp/Jw16NumericsGpu/qwen38-2b-prompts-10.jsonl")
texts = [json.loads(l)["text"] for l in open(prompts_path) if l.strip()][:nprompts]

from mlx_lm import load  # noqa: E402

model, tokenizer = load(model_dir)


def score(env_on):
    if env_on:
        os.environ.pop("MLX_OMARCHY_GDN_RAW_REPEAT", None)
    else:
        os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
    from mlx_lm.models.cache import make_prompt_cache
    total = 0.0
    count = 0
    for text in texts:
        ids = tokenizer.encode(text)[:maxtok]
        if len(ids) < 8:
            continue
        cache = make_prompt_cache(model)
        # S=1 decode scoring: feed every token as its own single-token step
        # so each GDN layer runs the decode kernel (route-sensitive).
        prev = None
        for t in ids:
            if prev is None:
                logits = model(mx.array([[t]], dtype=mx.int32), cache=cache)
            else:
                logits = model(mx.array([[prev]], dtype=mx.int32), cache=cache)
            last = logits[:, -1, :].astype(mx.float32)
            lp = last[0, t] - mx.logsumexp(last, axis=-1)[0]
            mx.eval(lp)
            total += -float(lp)
            count += 1
            prev = t
    mean_nll = total / max(count, 1)
    return {"mean_nll_per_token": mean_nll, "tokens": count,
            "exp_mean_nll": float(mx.exp(mx.array(mean_nll)))}


off = score(False)
on = score(True)
delta_pct = 100.0 * (on["mean_nll_per_token"] - off["mean_nll_per_token"]) / abs(off["mean_nll_per_token"])
res = {"composed": off, "fused": on, "mean_nll_delta_pct": round(delta_pct, 4),
       "ppl_bar_0.1pct_pass": bool(abs(delta_pct) < 0.1)}
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res, indent=1))
