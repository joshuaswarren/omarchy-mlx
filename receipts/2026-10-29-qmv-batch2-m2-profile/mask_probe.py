"""CPU probe: what does mlx_lm pass as `mask` to gated_delta_update in BatchGenerator prefill and decode, for two prompts of equal length (unpadded)
and for two prompts of 12 and 13 tokens (left padded)? Prints one line per distinct call signature (T, mask None?, shape, dtype, all true?)."""
import sys

import mlx.core as mx
import mlx_lm.models.qwen3_5 as q
from mlx_lm import load
from mlx_lm.generate import BatchGenerator
from mlx_lm.sample_utils import make_sampler

EXT = [
    "What is the capital of France? Answer in one word.",
    "Name the largest planet of the solar system in one word.",
    "Which element has the chemical symbol Fe? Answer in one word.",
]
orig = q.gated_delta_update
seen = {}


def spy(*args, **kw):
    mask = kw.get("mask", args[8] if len(args) > 8 else None)
    q_ = args[0]
    sig = (int(q_.shape[0]), int(q_.shape[1]), mask is None,
           None if mask is None else tuple(mask.shape), None if mask is None else str(mask.dtype),
           None if mask is None else bool(mx.all(mask).item()))
    seen[sig] = seen.get(sig, 0) + 1
    return orig(*args, **kw)


q.gated_delta_update = spy
model, tok = load(sys.argv[1])
for name, rows in (("E2", [0, 1]), ("P2", [0, 2])):
    seen.clear()
    prompts = [tok.encode(EXT[i]) for i in rows]
    bg = BatchGenerator(model, max_tokens=2, sampler=make_sampler(temp=0.0))
    bg.insert(prompts, max_tokens=[2] * len(prompts))
    for _ in range(3):
        bg.next()
    print(name, "lengths", [len(p) for p in prompts], flush=True)
    for sig, n in sorted(seen.items(), key=str):
        print("  B,T,mask_none,mask_shape,dtype,all_true", sig, "calls", n, flush=True)
