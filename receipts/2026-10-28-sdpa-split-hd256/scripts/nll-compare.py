"""Teacher-forced decode numerics: mean negative log-likelihood per token over decode steps, for one arm (the env decides the attention route).
Usage: nll-compare.py MODEL_REPO PROMPT_TOKENS STEPS TEXT_GLOB
Real English text (the repository's own docs) is tokenized; the first PROMPT_TOKENS are prefilled in one pass, then the next STEPS tokens are fed
one at a time (the decode path: query length 1, cache past the route's minimum) and each next-token log-probability is scored.
Prints NLL (nats per token, float64 accumulation of float32 log-softmax) and the top-1 agreement with the text.
"""
import glob
import os
import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

repo, n_prompt, n_steps, pattern = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
model, tok = load(repo)
text = "\n\n".join(open(p, encoding="utf-8").read() for p in sorted(glob.glob(os.path.expanduser(pattern))))
ids = tok.encode(text)[: n_prompt + n_steps + 1]
assert len(ids) == n_prompt + n_steps + 1, f"text too short: {len(ids)} tokens"
print("NLL_SETUP", repo, "prompt", n_prompt, "steps", n_steps, "mlx", mx.__file__, "SPLIT", os.environ.get("MLX_OMARCHY_SDPA_DECODE_SPLIT"), flush=True)
cache = make_prompt_cache(model)
x = mx.array(ids[:n_prompt])[None]
step = 512
for s in range(0, n_prompt, step):
    out = model(x[:, s:s + step], cache=cache)
    mx.eval(out[:, -1, :])
logits = out[:, -1, :]
total = 0.0
hits = 0
nll = []
for i in range(n_steps):
    target = ids[n_prompt + i]
    lp = logits[0].astype(mx.float32)
    lp = lp - mx.logsumexp(lp)
    mx.eval(lp)
    value = -float(lp[target].item())
    total += value
    nll.append(value)
    hits += int(int(mx.argmax(lp).item()) == target)
    logits = model(mx.array([[target]]), cache=cache)[:, -1, :]
    mx.eval(logits)
print(f"NLL {total / n_steps:.6f} nats/token over {n_steps} steps, top-1 agreement with the text {hits}/{n_steps}", flush=True)
print("NLLVEC", " ".join(f"{v:.5f}" for v in nll), flush=True)
