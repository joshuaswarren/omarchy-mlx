"""Where does a PADDED batched prefill (four prompts of different lengths through BatchGenerator: left padding, a real mask)
lose to the same four prompts one at a time? pfbg: 1.46-1.64x of sequential after PR 40 on both chips. Same sublayer timing as
pfmods.py (every sublayer of every decoder layer timed between an eval and a stream sync), but the passes are the real
mlx_lm.server prefill: one BatchGenerator run of the four prompts (max_tokens=1) against four runs of one prompt each; warm-up
of both first; the profiled pass is the one after. Per type: total ms over all layers, batched against the sum of the four
single runs. Rule (pre-registered with pfbg2): the op type whose batched ms exceeds its four-single sum by the most ms is
the next fix; a type within 10% of its sum is cleared. Args: model_dir prompt_file L1 L2 L3 L4."""
import json
import sys
import time
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import greedy_sampler

model, tok = load(sys.argv[1])
base = tok.encode(open(sys.argv[2]).read())
lengths = [int(a) for a in sys.argv[3:7]]
prompts = [(base * (n // len(base) + 1))[:n] for n in lengths]
prompts = [p[: n - 1] + [p[-1] + i + 1] for i, (p, n) in enumerate(zip(prompts, lengths))]
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


def run(batch):
    bg = BatchGenerator(model, max_tokens=1, sampler=greedy_sampler)
    bg.insert(batch, max_tokens=[1] * len(batch))
    while bg.next_generated():
        pass
    bg.close()


def profiled(batch):
    acc.clear()
    active[0] = True
    t = time.perf_counter()
    run(batch)
    total = time.perf_counter() - t
    active[0] = False
    return total, dict(acc)


run(prompts)
for p in prompts:
    run([p])
bt, bacc = profiled(prompts)
st, sacc = 0.0, defaultdict(float)
for p in prompts:
    t, a = profiled([p])
    st += t
    for k, v in a.items():
        sacc[k] += v
rows = {k: {"batched_ms": round(bacc.get(k, 0) * 1e3, 1), "four_single_ms": round(sacc.get(k, 0) * 1e3, 1),
            "excess_ms": round((bacc.get(k, 0) - sacc.get(k, 0)) * 1e3, 1)} for k in set(bacc) | set(sacc)}
rows = dict(sorted(rows.items(), key=lambda kv: -kv[1]["excess_ms"]))
print("PFMODSP " + json.dumps({"mlx": mx.__version__, "lengths": lengths, "batched_total_s": round(bt, 3),
                               "four_single_total_s": round(st, 3), "types": rows}), flush=True)
