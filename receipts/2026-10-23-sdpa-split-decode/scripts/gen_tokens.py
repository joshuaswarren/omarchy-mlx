"""Greedy generation of the ledger step prompt, token ids and the top-2 log-probability gap at every step, written as JSON.
usage: gen_tokens.py MODEL PROMPT_TOKENS GEN_TOKENS OUT.json [float32]   The prompt is built exactly as llmstep.py builds it."""
import json
import sys

import mlx.core as mx
from mlx_lm import load, stream_generate

model_dir, n_prompt, n_gen, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
model, tok = load(model_dir)
if len(sys.argv) > 5 and sys.argv[5] == "float32":
    model.set_dtype(mx.float32)
text = "Apple Silicon runs local language models on Linux through Vulkan and a reverse engineered GPU driver. "
ids = tok.encode(text * max(1, round(n_prompt / 18)))
for _ in stream_generate(model, tok, tok.encode("Warm up the GPU."), max_tokens=8):
    pass
tokens, gaps = [], []
for r in stream_generate(model, tok, ids, max_tokens=n_gen):
    tokens.append(int(r.token))
    top = sorted(mx.topk(r.logprobs.astype(mx.float32), 2).tolist())
    gaps.append(top[1] - top[0])
with open(out, "w") as f:
    json.dump({"prompt_tokens": len(ids), "tokens": tokens, "top2_gap": gaps}, f)
print("GEN done", len(ids), len(tokens), flush=True)
