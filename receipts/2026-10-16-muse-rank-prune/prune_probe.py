"""prune_probe.py: four CPU ranks (mx.distributed ring on localhost, stock mlx) run the tiny 2-bit Muse model, layers split by omarchy-cluster's _pipeline_split, the token shared by omarchy-cluster's _share_token (both imported from the rank.py checkout), greedy 12 tokens.
Weights come from one safetensors file loaded lazily (load_weights), then the pipeline split, then mx.eval(parameters), the order mlx_lm.sharded_load uses.
Run under mlx.launch with MLX_OMARCHY_PIPELINE_PRUNE=0 or 1, MUSE_SERVE_DIR (the repo's serve directory) and OMARCHY_CLUSTER_DIR (an omarchy-cluster checkout) set. Each rank prints active memory after the eval and after the token-sharing swap, the pruned tensor bytes, and its tokens."""
import gc
import importlib
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

sys.path.insert(0, os.environ["MUSE_SERVE_DIR"])
sys.path.insert(0, os.environ["OMARCHY_CLUSTER_DIR"])
import mlx_omarchy_muse  # noqa: E402

mlx_omarchy_muse.register()
module = importlib.import_module(mlx_omarchy_muse.MODEL_NAME)
from mlx_lm.models.pipeline import PipelineMixin  # noqa: E402
from omarchy_cluster.rank import _pipeline_split, _share_token  # noqa: E402

CKPT = sys.argv[1]
COUNTS = [2, 2, 2, 2]
VOCAB = int(os.environ.get("PROBE_VOCAB", "32000"))
CFG = {"model_type": "muse_spark", "hidden_size": 256, "num_hidden_layers": 8, "intermediate_size": 512, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64, "vocab_size": VOCAB, "sliding_window": 16, "moe_hidden_size": 128, "moe_intermediate_size": 192, "num_local_experts": 8, "num_experts_per_tok": 2, "max_position_embeddings": 256}


def build():
    model = module.Model(module.ModelArgs.from_dict(CFG))
    nn.quantize(model, group_size=64, bits=2, class_predicate=lambda p, m: hasattr(m, "to_quantized") and "switch_mlp" in p)
    return model


group = mx.distributed.init(backend="ring")
rank, size = group.rank(), group.size()
if not os.path.exists(CKPT) and rank == 0:
    m = build()
    rs = np.random.RandomState(17)
    flat = tree_flatten(m.parameters())
    out = []
    for k, v in flat:
        if v.dtype == mx.float32:
            v = mx.array((rs.standard_normal(v.shape) * 0.1).astype(np.float16))
        out.append((k, v))
    mx.save_safetensors(CKPT, dict(out))
    print("rank 0 wrote checkpoint", flush=True)
    del m, flat, out, v
    gc.collect()
mx.eval(mx.distributed.all_sum(mx.array(1.0), stream=mx.cpu))  # checkpoint exists before any rank reads it

model = build()
model.load_weights(CKPT, strict=False)
PipelineMixin.pipeline = lambda self, g: _pipeline_split(self, g, COUNTS)
model.model.pipeline(group)
mx.eval(model.parameters())
gc.collect()
a = mx.get_active_memory()
_share_token(model)
gc.collect()
b = mx.get_active_memory()
pruned = isinstance(model.lm_head, type(model.lm_head)) and ("NoHead" in type(model.lm_head).__name__)
print(f"rank {rank} PRUNE={os.environ.get('MLX_OMARCHY_PIPELINE_PRUNE')} layers {model.model.start_idx}-{model.model.end_idx - 1} embed={type(model.model.embed_tokens).__name__} active_after_eval={a} active_after_share={b}", flush=True)

from mlx_lm.generate import generate_step  # noqa: E402

kw = {"sampler": getattr(model, "omarchy_sampler", None)}
import hashlib  # noqa: E402

toks, digs = [], []
for t, lp in generate_step(mx.array([1, 2, 3, 4, 5, 6, 7, 8]), model, max_tokens=12, **kw):
    toks.append(int(t))
    digs.append(lp)  # no evaluation inside the loop: every rank must reach its collectives in the same order
digs = [hashlib.sha256(np.array(lp.astype(mx.float32)).tobytes()).hexdigest()[:8] for lp in digs]
print(f"rank {rank} logprob-sha8 {' '.join(digs)}", flush=True)
print(f"rank {rank} tokens {toks}", flush=True)
