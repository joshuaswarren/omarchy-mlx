#!/usr/bin/env python3
"""Token-316 near-tie evidence: compare both routes at the first divergence.
usage: tf316.py W9D_CTL_JSON OUT.json
"""
import glob
import json
import os
import sys

import mlx.core as mx

from bf16_ulp import bf16_ulp, within_one_bf16_ulp

os.environ.pop("MLX_OMARCHY_GDN_RAW_REPEAT", None)

from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

ctl_json, out_path = sys.argv[1], sys.argv[2]
record = json.load(open(ctl_json, encoding="utf-8"))["per_prompt"][0]
generated = record["output_ids"][:316]
model_dir = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/"))[0]
model, _ = load(model_dir)
prefix = record["input_ids"] + generated


def run(fused):
    os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "1" if fused else "0"
    cache = make_prompt_cache(model)
    logits = model(mx.array([prefix], dtype=mx.int32), cache=cache)
    last = logits[:, -1, :].astype(mx.float32)
    top = mx.topk(last, k=3, axis=-1)
    values = sorted((float(value) for value in mx.flatten(top)), reverse=True)
    argmax = int(mx.argmax(last, axis=-1).item())
    mx.eval(last)
    return {"argmax": argmax, "top3": values,
            "gap12": values[0] - values[1], "gap23": values[1] - values[2]}


composed = run(False)
fused = run(True)
magnitude = abs(composed["top3"][0])
ulp = bf16_ulp(magnitude)
gap = composed["gap12"]
result = {"composed": composed, "fused": fused, "logit_magnitude": magnitude,
          "bf16_ulp": ulp, "argmax_agree": composed["argmax"] == fused["argmax"],
          "composed_gap12_in_ulps": gap / ulp,
          "within_one_bf16_ulp": within_one_bf16_ulp(gap, magnitude)}
json.dump(result, open(out_path, "w"), indent=1)
print(json.dumps(result, indent=1))
