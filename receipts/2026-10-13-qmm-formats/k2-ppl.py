"""K2 perplexity on a fixed text (A/B across builds). Args: model text [step=128] [limit]; step 1 = teacher-forced decode (m == 1 kernels)."""
import json, math, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
mx.set_cache_limit(0)
model, tok = load(sys.argv[1])
ids = mx.array(tok.encode(open(sys.argv[2]).read())[:2049])[None]
cache = make_prompt_cache(model)
nll, n, same = 0.0, 0, []
STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 128
LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else ids.shape[1] - 1
for i in range(0, LIMIT, STEP):
    inp, tgt = ids[:, i:i + STEP], ids[:, i + 1:i + 1 + STEP]
    logits = model(inp, cache=cache)[:, :tgt.shape[1]].astype(mx.float32)
    lse = mx.logsumexp(logits, axis=-1)
    tl = mx.take_along_axis(logits, tgt[..., None], axis=-1)[..., 0]
    am = mx.argmax(logits, axis=-1)[0]
    mx.eval(lse, tl, am); del logits
    nll += float((lse - tl).sum()); n += tgt.shape[1]; same.append(am)
print(json.dumps({"tokens": n, "ppl": math.exp(nll / n)}))
