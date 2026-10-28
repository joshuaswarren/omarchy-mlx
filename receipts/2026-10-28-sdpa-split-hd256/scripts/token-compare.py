"""Greedy decode of a fixed real prompt; prints the generated token ids and text. Run once per arm (env decides the route).
Usage: token-compare.py MODEL_REPO N_TOKENS
The prompt is a fixed paragraph repeated to about 700 tokens so the decode cache is past the route's minimum length.
"""
import os
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

repo, n_new = sys.argv[1], int(sys.argv[2])
model, tok = load(repo)
para = (
    "The harbour town woke slowly. Fishing boats rocked against their ropes while gulls argued over scraps on the quay. "
    "A baker carried trays of warm bread along the cobbles, and the clock above the customs house struck seven. "
    "Nobody hurried; the tide would not turn for another hour, and the ferry captain was still finishing his tea. "
)
text = para * 6 + "Question: describe in two sentences what the town does first each morning.\nAnswer:"
ids = mx.array(tok.encode(text))[None]
print("PROMPT_TOKENS", ids.shape[1], "mlx", mx.__file__, "SPLIT", os.environ.get("MLX_OMARCHY_SDPA_DECODE_SPLIT"), flush=True)
cache = make_prompt_cache(model)
out = model(ids, cache=cache)
cur = mx.argmax(out[:, -1, :], axis=-1)[None]
mx.eval(cur)
gen = [int(cur.item())]
for _ in range(n_new - 1):
    logits = model(cur, cache=cache)
    cur = mx.argmax(logits[:, -1, :], axis=-1)[None]
    mx.eval(cur)
    gen.append(int(cur.item()))
print("TOKENS", " ".join(str(t) for t in gen), flush=True)
print("TEXT", repr(tok.decode(gen)), flush=True)
