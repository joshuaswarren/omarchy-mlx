"""VocabPrune on-device qualification (jwm1). Compares
mx.fast.greedy_quantized_argmax against the composed upstream step
(argmax(logits - logsumexp(logits))) token by token and bit by bit.

usage: greedy_qual.py MODEL PROMPTS OUT.json
MLX_OMARCHY_GREEDY_PRUNE_TEST=keep|full selects the qualification modes.
"""
import json, os, sys, time
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import _greedy_head

MODEL, PROMPTS, OUT = sys.argv[1:4]
mode = os.environ.get("MLX_OMARCHY_GREEDY_PRUNE_TEST", "prune")
model, tok = load(MODEL)
text = model.language_model
emb = text.model.embed_tokens
t0 = time.perf_counter()
head = _greedy_head(model)
build_s = time.perf_counter() - t0
assert head is not None, "greedy head not routed"
high = emb._greedy_high_bits
V = emb.weight.shape[0]

# high_bits layout on the device-built array, first/last 256 rows.
for sl in (slice(0, 256), slice(V - 256, V)):
    q = np.array(emb.weight[sl])
    a = (((q[:, :, None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15) >> 1).reshape(-1, 32, 64)
    h = np.array(high[sl]).astype(np.uint64)
    for e in range(64):
        word, off = (e * 3) // 32, (e * 3) % 32
        cols = np.arange(32) * 6 + word
        v = (h[:, cols] >> off) & 7
        if off > 29:
            v = ((h[:, cols] >> off) | (h[:, cols + 1] << (32 - off))) & 7
        assert np.array_equal(v.astype(np.uint8), a[:, :, e]), ("high_bits layout", e)


def f32bits(a):
    return int(np.array(a.astype(mx.float32)).reshape(-1)[0].view(np.uint32))


def check(hidden, tag, cases):
    logits = emb.as_linear(hidden)
    lse = mx.logsumexp(logits, keepdims=True)
    ref = mx.argmax(logits - lse, -1)
    tokv, stats = mx.fast.greedy_quantized_argmax(
        hidden, emb.weight, emb.scales, emb.biases, high, group_size=64, bits=4)
    top = mx.max(logits)
    mx.eval(ref, tokv, stats, top, lse)
    st = [int(v) for v in np.array(stats)]
    full = st[1] == 1
    want = f32bits(lse) if full else f32bits(top)
    c = {"tag": tag, "ref": int(ref.item()), "tok": int(tokv.item()),
         "survivors": st[0], "full": full, "bits_ok": st[2] == want}
    cases.append(c)
    return int(ref.item())


prompts = [json.loads(l)["text"] for l in open(PROMPTS) if l.strip()][:10]
cases, hiddens = [], []
for pi, p in enumerate(prompts):
    ids = tok.encode(p)
    cache = make_prompt_cache(model)
    if len(ids) > 1:
        model(mx.array(ids[:-1])[None], cache=cache)
        mx.eval([c.state for c in cache])
    cur = ids[-1]
    for step in range(32):
        hidden = text.model(mx.array([[cur]]), cache=cache)[:, -1, :]
        mx.eval(hidden)
        hiddens.append(hidden)
        cur = check(hidden, f"p{pi}s{step}", cases)

# Scaled and synthetic inputs: small scales push the max logit toward the
# logprob-merge regime (full path), large ones stress the bound slack.
rng = np.random.default_rng(7)
rms = float(np.sqrt(np.mean(np.array(hiddens[0].astype(mx.float32)) ** 2)))
for i in range(40):
    base = hiddens[(i * 7) % len(hiddens)]
    for scale in (0.02, 0.1, 0.3, 0.6, 1.5, 3.0):
        check((base.astype(mx.float32) * scale).astype(mx.bfloat16), f"scaled{scale}", cases)
    x = mx.array((rng.standard_normal((1, 2048)) * rms * rng.choice([0.05, 0.3, 1.0, 3.0])).astype(np.float32)).astype(mx.bfloat16)
    check(x, "gauss", cases)
check(mx.zeros((1, 2048), mx.bfloat16), "zeros", cases)

# Timing: full composed step vs the greedy op on one real hidden state.
h = hiddens[5]
def full_fn():
    logits = emb.as_linear(h)
    return mx.argmax(logits - mx.logsumexp(logits, keepdims=True), -1)


op_fn = lambda: mx.fast.greedy_quantized_argmax(h, emb.weight, emb.scales, emb.biases, high, group_size=64, bits=4)[0]
timing = {}
for name, fn in (("full", full_fn), ("greedy", op_fn), ("full2", full_fn), ("greedy2", op_fn)):
    for _ in range(3):
        mx.eval(fn())
    best = 1e9
    for _ in range(7):
        t0 = time.perf_counter()
        mx.eval([fn() for _ in range(20)])
        best = min(best, (time.perf_counter() - t0) / 20)
    timing[name] = round(best * 1e3, 4)

corpus = [c for c in cases if c["tag"].startswith("p")]
summary = {
    "mode": mode, "build_high_bits_s": round(build_s, 3), "V": V,
    "cases": len(cases), "token_mismatches": sum(c["tok"] != c["ref"] for c in cases),
    "bits_mismatches": sum(not c["bits_ok"] for c in cases),
    "full_path_cases": sum(c["full"] for c in cases),
    "corpus_steps": len(corpus),
    "corpus_full_path": sum(c["full"] for c in corpus),
    "corpus_survivors_median": float(np.median([c["survivors"] for c in corpus])),
    "corpus_survivors_p90": float(np.percentile([c["survivors"] for c in corpus], 90)),
    "corpus_survivors_max": max(c["survivors"] for c in corpus),
    "corpus_pruned_fraction_mean": 1 - float(np.mean([c["survivors"] for c in corpus])) / V,
    "timing_ms": timing,
}
json.dump({"summary": summary, "cases": cases}, open(OUT, "w"))
print(json.dumps(summary, indent=1))
