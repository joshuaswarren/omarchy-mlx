# yardstick: width-4096 subgroup-tree RMSNorm (default on G16S) vs the deployed shared-memory tree (MLX_OMARCHY_NORM_APPLE_4096=0).
# The route only serves rows <= 64, so the check runs in decode mode: each window prefills a short prompt, then feeds the true next token one at a
# time through two KV caches (one per arm), toggling the switch per forward (it is read live). Qwen3-8B-4bit, wikitext-2 test.
import os, sys, math, subprocess, time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
D = os.path.dirname(os.path.abspath(__file__))
WINDOWS = int(os.environ.get("WINDOWS", "8")); LEN = int(os.environ.get("LEN", "1024")); PRE = 64
model, tok = load("mlx-community/Qwen3-8B-4bit")
ids = tok.encode(open(os.environ.get("WIKITEXT", os.path.join(D, "wikitext-2-test.txt"))).read())
print("mlx", mx.__version__, "tokens", len(ids), flush=True)
def fwd(arm, x, cache):
    os.environ["MLX_OMARCHY_NORM_APPLE_4096"] = "0" if arm == "old" else "1"
    lg = model(x, cache=cache).astype(mx.float32)[0, -1]
    return lg, lg - mx.logsumexp(lg, axis=-1)
nll = {"old": 0.0, "new": 0.0}; n = 0; agree = 0; kl = 0.0; maxabs = 0.0
for w in range(WINDOWS):
    seg = ids[w * LEN:(w + 1) * LEN + 1]
    caches = {a: make_prompt_cache(model) for a in ("old", "new")}
    for a in caches:
        os.environ["MLX_OMARCHY_NORM_APPLE_4096"] = "0" if a == "old" else "1"
        mx.eval(model(mx.array(seg[:PRE])[None], cache=caches[a]))
    for i in range(PRE, len(seg) - 1):
        x = mx.array([seg[i]])[None]; t = seg[i + 1]
        la, pa = fwd("old", x, caches["old"]); lb, pb = fwd("new", x, caches["new"])
        s = mx.sum(mx.exp(pa) * (pa - pb)); m = mx.max(mx.abs(la - lb))
        mx.eval(pa, pb, s, m)
        nll["old"] -= pa[t].item(); nll["new"] -= pb[t].item()
        agree += int(mx.argmax(la).item() == mx.argmax(lb).item()); kl += s.item(); maxabs = max(maxabs, m.item()); n += 1
    print(f"WIN {w} n={n} old_nll={nll['old']:.2f} new_nll={nll['new']:.2f}", flush=True)
print(f"YARDSTICK width-4096 norm, decode mode, {WINDOWS} windows x {LEN - PRE} steps")
print(f"PPL old={math.exp(nll['old']/n):.4f} new={math.exp(nll['new']/n):.4f} tokens={n}")
print(f"LOGITS argmax_agree={agree}/{n} ({agree/n*100:.2f}%) mean_KL(old||new)={kl/n:.3e} max_abs_logit_diff={maxabs:.3f}")
print("ACC_DONE")
