import json, sys
import numpy as np
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import generate_step

pi = int(sys.argv[1])
model, tok = load("/tmp/vp/model")
emb = model.language_model.model.embed_tokens
if pi < 0:
    np.save("/tmp/vp/scales.npy", np.array(emb.scales.astype(mx.float32)))
    np.save("/tmp/vp/biases.npy", np.array(emb.biases.astype(mx.float32)))
    np.save("/tmp/vp/q.npy", np.array(emb.weight))
    print("dtypes", emb.weight.dtype, emb.scales.dtype, emb.biases.dtype, emb.weight.shape)
    sys.exit(0)
orig = emb.as_linear
captured = []

def as_linear(x):
    captured.append(x[:, -1, :])
    return orig(x)

emb.as_linear = as_linear
text = [json.loads(l)["text"] for l in open("/tmp/vp/qwen38-2b-prompts.jsonl") if l.strip()][pi]
ids = tok.encode(text)
toks = []
for (t, _), _ in zip(generate_step(mx.array(ids), model, prompt_cache=make_prompt_cache(model)), range(32)):
    toks.append(int(t))
H = np.stack([np.array(h.astype(mx.float32))[0] for h in captured[:32]])
np.save(f"/tmp/vp/hidden-{pi}.npy", H)
np.save(f"/tmp/vp/tokens-{pi}.npy", np.array(toks))
print(pi, len(ids), repr(tok.decode(toks)[:80]), flush=True)
