#!/usr/bin/env python3
"""Token-316 near-tie evidence: teacher-force the free-running ctl prefix
(tokens up to the first divergence) through both routes and compare the
top-2 logit gap at the divergence position.
usage: tf316.py W9D_CTL_JSON OUT.json
"""
import glob
import json
import os
import sys

import mlx.core as mx

os.environ.pop("MLX_OMARCHY_GDN_RAW_REPEAT", None)

from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

ctl_json, out_path = sys.argv[1], sys.argv[2]
rec = json.load(open(ctl_json))["per_prompt"][0]
gen = rec["output_ids"][:316]  # tokens 0..315: the last common token is gen[315]
model_dir = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/"))[0]

model, tokenizer = load(model_dir)
prompt_ids = tokenizer.encode(rec.get("prompt") or json.load(open(ctl_json))["meta"].get("prompt", "")) \
    if False else None

# Rebuild the exact free-run context: the bench fed the tokenized prompt text.
# The ctl record stores input_ids - use them verbatim.
prompt_ids = rec["input_ids"]
# The free-run fed prompt -> first token; the first generated token is gen[0].
# Divergence at gen index 315 means the argmax for that position came from the
# state after 315 generated tokens. Feed prompt + gen[:315] and compare the
# next-token distribution both routes.
prefix = prompt_ids + gen[:316]


def run(env_on):
    if env_on:
        os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "1"
    else:
        os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
    cache = make_prompt_cache(model)
    logits = model(mx.array([prefix], dtype=mx.int32), cache=cache)
    last = logits[:, -1, :].astype(mx.float32)
    top = mx.topk(last, k=3, axis=-1)
    argmax = int(mx.argmax(last, axis=-1).item())
    vals = sorted((float(v) for v in mx.flatten(top)), reverse=True)
    mx.eval(last)
    return {"argmax": argmax, "top3": vals,
            "gap12": vals[0] - vals[1], "gap23": vals[1] - vals[2]}


off = run(False)
on = run(True)
mag = abs(off["top3"][0])
ulp = 2.0 ** (__import__("math").floor(__import__("math").log2(mag)) - 7) if mag else 2.0 ** -133
res = {"composed": off, "fused": on, "logit_magnitude": mag, "bf16_ulp": ulp,
       "argmax_agree": off["argmax"] == on["argmax"],
       "composed_gap12_in_ulps": off["gap12"] / ulp,
       "gap12_in_ulps": off["gap12"] / ulp,
       "within_tolerance": off["gap12"] <= ulp or off["gap12"] < 0.05}
json.dump(res, open(out_path, "w"), indent=1)
print(json.dumps(res, indent=1))
