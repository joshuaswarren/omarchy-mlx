"""Prompt and generated token ids of one greedy generation, with the gap
between the two most likely tokens at every step (log-probability units).

usage: tokens.py <repo> <max_tokens>
Same chat prompt and sampler as decode_bench.py. Prints one JSON line.
"""
import json
import sys

import prov

prov.check()
import mlx.core as mx  # noqa: E402
from mlx_lm import load, stream_generate  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

PROMPT = "Explain in plain language how a rainbow forms. Give four distinct sentences."
repo, n = sys.argv[1], int(sys.argv[2])
model, tok = load(repo)
msgs = [{"role": "user", "content": PROMPT}]
try:
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
except TypeError:
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
ids, gaps = [], []
for r in stream_generate(model, tok, prompt, max_tokens=n, sampler=make_sampler(temp=0.0)):
    ids.append(int(r.token))
    lp = r.logprobs.astype(mx.float32)
    top = mx.sort(lp)[-2:]
    gaps.append(round(float((top[1] - top[0]).item()), 4))
print(json.dumps({"model": repo.split("/")[-1], "prompt_ids": tok.encode(prompt), "ids": ids,
                  "top2_gap": gaps, "text": tok.decode(ids)}), flush=True)
