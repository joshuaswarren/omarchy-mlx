#!/usr/bin/env python3
"""Teacher-forced-style greedy logits gate for the qwen3.8-2B contract.

Runs plain greedy decode (the exact bench trajectory, one host sync per
step) on the 10 corpus prompts and records, per step: the argmax token id
and its top-1 logit. Comparing two arms (--compare <other.json>) then
reports: prompts with any token flip, the first flip position, and the
max |delta top-1 logit| over the matched prefix (before the first flip) —
the cascade after a flip is uninformative by construction.

Gate (contract + kernel-flags.md): 0 flips AND max prefix delta within
2 bf16 quanta of the top-1 logit scale.
"""
import argparse
import json
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import generate_step

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompts", required=True)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--limit", type=int, default=10)
ap.add_argument("--compare", default="")
ap.add_argument("--out", required=True)
a = ap.parse_args()

model, tok = load(a.model)
prompts = [json.loads(l)["text"] for l in open(a.prompts) if l.strip()][: a.limit]

records = []
for pi, text in enumerate(prompts):
    ids = tok.encode(text)
    cache = make_prompt_cache(model)
    it = generate_step(mx.array(ids), model, prompt_cache=cache)
    steps = []
    for _ in range(a.steps):
        tokv, logprobs = next(it)
        steps.append({"tok": int(tokv)})
    # Second pass teacher-forced on that trajectory to capture the top-1
    # logit per step without sampling feedback (identical context chain).
    cache = make_prompt_cache(model)
    x = mx.array(ids)[None]
    for si in range(a.steps):
        logits = model(x, cache=cache)[:, -1, :]
        top = mx.argmax(logits, -1)
        steps[si]["logit"] = float(
            mx.take_along_axis(logits, top[None, None], -1).item())
        assert int(top) == steps[si]["tok"], (pi, si, int(top), steps[si]["tok"])
        x = mx.array([steps[si]["tok"]])
    records.append({"prompt_idx": pi, "prompt_len": len(ids), "steps": steps})

result = {"model": a.model, "limit": a.limit, "steps": a.steps,
          "records": records}
if a.compare:
    other = json.load(open(a.compare))["records"]
    flips = 0
    first_flips = []
    max_delta = 0.0
    for mine, theirs in zip(records, other):
        ff = None
        for si, (s1, s2) in enumerate(zip(mine["steps"], theirs["steps"])):
            if s1["tok"] != s2["tok"]:
                ff = si
                break
            max_delta = max(max_delta, abs(s1["logit"] - s2["logit"]))
        if ff is not None:
            flips += 1
            first_flips.append({"prompt_idx": mine["prompt_idx"], "step": ff})
    result["compare"] = {"flipped_prompts": flips,
                         "first_flips": first_flips,
                         "max_abs_logit_delta_prefix": max_delta}
json.dump(result, open(a.out, "w"))
print(json.dumps(result.get("compare", {"wrote": a.out}), indent=1))
