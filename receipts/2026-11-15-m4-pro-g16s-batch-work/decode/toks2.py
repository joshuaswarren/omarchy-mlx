"""toks2.py <repo> <out.json>: greedy tokens (256 per prompt) plus the top1-top2 logprob margin at every step."""
import sys, json, mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
m, tok = load(sys.argv[1])
res = []
for p in ["The history of France begins in antiquity.", "Write a Python function that merges two sorted lists." * 30,
          "Explain in detail how a transformer language model computes attention, step by step."]:
    ids = tok.encode(p); toks = []; marg = []
    for (t, lp), n in zip(generate_step(mx.array(ids), m), range(256)):
        toks.append(int(t)); top = mx.topk(lp.astype(mx.float32), 2); mx.eval(top)
        a = sorted(top.tolist()); marg.append(round(a[1] - a[0], 5))
    res.append({"n_prompt": len(ids), "toks": toks, "margin": marg})
json.dump(res, open(sys.argv[2], "w"))
print("TOKS_OK", [r["toks"][:4] for r in res])
