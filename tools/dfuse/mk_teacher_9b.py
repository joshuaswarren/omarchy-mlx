#!/usr/bin/env python3
"""Generate the 9B teacher corpus: 10 prompts x 512 free-running greedy
tokens on the route-OFF (composed) path. Mirrors the NormApple teacher
format: {"max_tokens":512,"prompts":[{prompt_index,prompt,tokens}]}.
usage: mk_teacher_9b.py OUT.json
"""
import glob
import json
import os
import sys

import mlx.core as mx

os.environ.pop("MLX_OMARCHY_GDN_RAW_REPEAT", None)

from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

out_path = sys.argv[1]
prompts_path = "/var/tmp/Jw16NumericsGpu/qwen38-2b-prompts-10.jsonl"
model_dir = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/*/"))[0]

texts = [json.loads(l)["text"] for l in open(prompts_path, encoding="utf-8") if l.strip()]
assert len(texts) == 10, len(texts)
model, tokenizer = load(model_dir)
rows = []
for i, text in enumerate(texts):
    ids = tokenizer.encode(text)
    cache = make_prompt_cache(model)
    logits = model(mx.array([ids], dtype=mx.int32), cache=cache)
    tokens = []
    for _ in range(512):
        tok = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        tokens.append(tok)
        logits = model(mx.array([[tok]], dtype=mx.int32), cache=cache)
    mx.eval(logits)
    rows.append({"prompt_index": i, "prompt": text, "tokens": tokens})
    print(f"teacher prompt {i + 1}/10", flush=True)
payload = {"model": model_dir, "mx_version": mx.__version__, "max_tokens": 512,
           "limit": 10, "prompts": rows}
json.dump(payload, open(out_path, "w"))
print(json.dumps({"output": out_path, "prompts": len(rows)}))
