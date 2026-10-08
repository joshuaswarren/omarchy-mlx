"""Where does the [4, T] prefill lose 5-7x to four [1, T] passes (pfbatch: T 128/454/1024 ratios 5.6/6.7/7.6)? One
prefill pass per row count (1 and 4, fresh cache, warm-up pass first) with every sublayer of every decoder layer timed
between an eval and a stream sync: linear_attn (the GDN layers) and its children and mx.fast ops, self_attn (full
attention), mlp, the two norms; then the tied head. Per type: total ms over all layers. B=1 is multiplied by 4 in the
ratio column (four sequential passes). Args: model_dir prompt_file T."""
import json
import sys
import time
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tok = load(sys.argv[1])
base = tok.encode(open(sys.argv[2]).read())
T = int(sys.argv[3])
text = getattr(model, "language_model", model)
acc = defaultdict(float)
active = [False]


def timed(name, fn):
    def run(*args, **kwargs):
        if not active[0]:
            return fn(*args, **kwargs)
        mx.eval([a for a in args if isinstance(a, mx.array)])
        mx.synchronize()
        t = time.perf_counter()
        out = fn(*args, **kwargs)
        mx.eval(out)
        mx.synchronize()
        acc[name] += time.perf_counter() - t
        return out
    return run


def wrap_module(obj, name):
    cls_call = type(obj).__call__

    class Timed(type(obj)):
        __call__ = timed(name, lambda self, *a, **k: cls_call(self, *a, **k))

    obj.__class__ = Timed


for layer in text.model.layers:
    for attr in ("linear_attn", "self_attn", "mlp", "input_layernorm", "post_attention_layernorm"):
        mod = getattr(layer, attr, None)
        if mod is None:
            continue
        wrap_module(mod, attr)
        if attr == "linear_attn":
            for child_name, child in mod.children().items():
                if isinstance(child, nn.Module):
                    wrap_module(child, f"linear_attn.{child_name}")
for fn_name in ("gated_delta_update_raw", "gated_delta_update", "rms_norm_gated", "scaled_dot_product_attention"):
    if hasattr(mx.fast, fn_name):
        setattr(mx.fast, fn_name, timed(f"mx.fast.{fn_name}", getattr(mx.fast, fn_name)))
for conv in ("conv1d",):
    pass

ids = mx.array((base * (T // len(base) + 1))[:T])[None]
out = {"mlx": mx.__version__, "T": T}
for rows in (1, 4):
    x = mx.broadcast_to(ids, (rows, T))
    mx.eval(model(x, cache=make_prompt_cache(model)))
    acc.clear()
    active[0] = True
    t = time.perf_counter()
    mx.eval(model(x, cache=make_prompt_cache(model)))
    total = time.perf_counter() - t
    active[0] = False
    out[f"B{rows}"] = {"total_s": round(total, 3), **{k: round(v * 1e3, 1) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}}
print("PFMODS " + json.dumps(out), flush=True)
