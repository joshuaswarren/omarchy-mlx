#!/usr/bin/env python3
"""Three-arm kernel experiment: route OFF (composed) vs ON-tiled (TILE=1)
vs ON-legacy-perrow (TILE=0, PF default). Pairwise token identity.
usage: kernel3arm.py <model-dir> <out.json> [n-prompts=2] [tokens=256]
"""
import json
import os
import sys

import mlx.core as mx

model_dir = sys.argv[1]
out_path = sys.argv[2]
nprompts = int(sys.argv[3]) if len(sys.argv) > 3 else 2
ntoks = int(sys.argv[4]) if len(sys.argv) > 4 else 256

prompts_path = "/var/tmp/Jw16NumericsGpu/qwen38-2b-prompts-10.jsonl"
texts = [json.loads(l)["text"] for l in open(prompts_path) if l.strip()][:nprompts]

from mlx_lm import load  # noqa: E402

model, tokenizer = load(model_dir)


def generate(tile):
    os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "1"
    if tile is None:
        os.environ["MLX_OMARCHY_GDN_DECODE_TILE"] = "1"
    elif tile:
        os.environ["MLX_OMARCHY_GDN_DECODE_TILE"] = "1"
    else:
        os.environ["MLX_OMARCHY_GDN_DECODE_TILE"] = "0"
    from mlx_lm.models.cache import make_prompt_cache
    rows = []
    for text in texts:
        ids = tokenizer.encode(text)
        cache = make_prompt_cache(model)
        logits = model(mx.array([ids], dtype=mx.int32), cache=cache)
        toks = []
        for _ in range(ntoks):
            t = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            toks.append(t)
            logits = model(mx.array([[t]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        rows.append(toks)
    return rows


# Route OFF first (composed reference).
os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
off = generate.__wrapped__ if False else None
# inline composed
def gen_composed():
    os.environ["MLX_OMARCHY_GDN_RAW_REPEAT"] = "0"
    from mlx_lm.models.cache import make_prompt_cache
    rows = []
    for text in texts:
        ids = tokenizer.encode(text)
        cache = make_prompt_cache(model)
        logits = model(mx.array([ids], dtype=mx.int32), cache=cache)
        toks = []
        for _ in range(ntoks):
            t = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            toks.append(t)
            logits = model(mx.array([[t]], dtype=mx.int32), cache=cache)
        mx.eval(logits)
        rows.append(toks)
    return rows


off = gen_composed()
tile1 = generate(True)
tile0 = generate(False)

res = {"rows": []}
for i, (o, t1, t0) in enumerate(zip(off, tile1, tile0)):
    same_1 = sum(1 for x, y in zip(o, t1) if x == y)
    same_0 = sum(1 for x, y in zip(o, t0) if x == y)
    fd1 = next((j for j, (x, y) in enumerate(zip(o, t1)) if x != y), None)
    fd0 = next((j for j, (x, y) in enumerate(zip(o, t0)) if x != y), None)
    res["rows"].append({"prompt_index": i, "tiled_vs_composed_pct": round(100 * same_1 / ntoks, 2),
                        "tiled_first_div": fd1,
                        "perrow_vs_composed_pct": round(100 * same_0 / ntoks, 2),
                        "perrow_first_div": fd0})
    print(f"prompt {i}: tiled {same_1}/{ntoks} (fd {fd1}) | perrow {same_0}/{ntoks} (fd {fd0})", flush=True)
json.dump(res, open(out_path, "w"), indent=1)
