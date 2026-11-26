import os, sys, json
sys.path.insert(0, "/tmp/vp/applytest")
import numpy as np
import mlx.core as mx
import mlx_lm
assert mlx_lm.__file__.startswith("/tmp/vp/applytest"), mlx_lm.__file__
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import generate_step

calls = {"n": 0, "high": None}


def fake(x, w, scales, biases, high, group_size, bits):
    calls["n"] += 1
    calls["high"] = high
    logits = mx.quantized_matmul(x, w, scales, biases, transpose=True, group_size=group_size, bits=bits)
    lp = logits - mx.logsumexp(logits, -1, keepdims=True)
    return [mx.argmax(lp, -1), mx.array([w.shape[0], 1], mx.uint32)]


mx.fast.greedy_quantized_argmax = fake
model, tok = load("/tmp/vp/model")
prompts = [json.loads(l)["text"] for l in open("/tmp/vp/qwen38-2b-prompts.jsonl")][:2]


def run(text, steps=6):
    out, lps = [], []
    it = generate_step(mx.array(tok.encode(text)), model, prompt_cache=make_prompt_cache(model))
    for _ in range(steps):
        t, lp = next(it)
        out.append(int(t))
        lps.append(np.array(lp.astype(mx.float32)))
    return out, lps


for text in prompts:
    os.environ["MLX_OMARCHY_NO_GREEDY_PRUNE"] = "1"
    base, base_lp = run(text)
    del os.environ["MLX_OMARCHY_NO_GREEDY_PRUNE"]
    n0 = calls["n"]
    new, new_lp = run(text)
    print("tokens base", base, "greedy", new, "op calls", calls["n"] - n0, flush=True)
    assert base == new
    assert calls["n"] - n0 >= 6
    assert all(np.array_equal(a, b) for a, b in zip(base_lp, new_lp)), "lazy logprobs differ"
emb = model.language_model.model.embed_tokens
q = np.array(emb.weight[:64])
a = (((q[:, :, None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15) >> 1).reshape(64, 32, 64)
h = np.array(calls["high"][:64])
for g in range(32):
    for e in range(64):
        word, off = (e * 3) // 32, (e * 3) % 32
        pw = list(h[:, g * 6:(g + 1) * 6].T) + [np.zeros(64, np.uint32)]
        v = (pw[word] >> off) & 7 if off <= 29 else ((pw[word] >> off) | (pw[word + 1] << ((32 - off) & 31))) & 7
        assert np.array_equal(v, a[:, g, e]), (g, e)
print("plumbing OK: tokens identical, lazy logprobs identical, high_bits layout verified on 64 rows")
