"""Equivalence probe: vendored layer forward vs upstream.

Run on the M2 under gpu-turn. Compares the vendored fused-pass output
against the upstream layer output for layer 0, 13, 27 at seq_len 1 and
4, plus the full 28-layer prefill pass on a fixed input. Reports max
abs diff and P99.99 abs diff.
"""
import json, sys
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, "<home>/voice-site")
sys.path.insert(0, str(Path(__file__).resolve().parent / "serve"))

from mlx_audio.tts.utils import load_model
model = load_model("<home>/mlx-tts-home/voice/qwen3-tts-0.6b-customvoice-4bit")
print("loaded", flush=True)

from mlx_omarchy_assistant._vendored import qwen3_tts_step as V
V.vendorize_talker(model)
V.vendorize_cp(model)

cfg = model.config.talker_config
head_dim = cfg.head_dim


def diff(a, b):
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    return float(mx.max(d).item()), float(mx.quantile(d, 0.9999).item())


results = {}

# Per-layer equivalence: layer 0, 13, 27, at seq_len 1 and 4.
for layer_idx in [0, 13, 27]:
    layer = model.talker.model.layers[layer_idx]
    for seq_len in [1, 4]:
        x = mx.random.normal((1, seq_len, cfg.hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        pos = mx.broadcast_to(mx.arange(seq_len)[None, :], (1, seq_len))
        pos3 = mx.stack([pos, pos, pos], axis=0)
        cos, sin = model.talker.model.rotary_emb(x, pos3)
        rope = (cos, sin)
        # Upstream: a fresh standard KVCache per call
        from mlx_audio.lm.models.cache import KVCache
        u_cache = KVCache()
        a = layer(x, rope, None, u_cache)
        mx.eval(a)
        # Vendored: preallocated contiguous cache
        v_cache = V.StaticKVCache(
            keys=[mx.zeros((1, cfg.num_key_value_heads, 16, head_dim), dtype=mx.bfloat16)],
            values=[mx.zeros((1, cfg.num_key_value_heads, 16, head_dim), dtype=mx.bfloat16)],
            pos=0,
        )
        b = V.talker_layer_forward(
            layer, layer.self_attn.qkv_fused, 0, x, rope, v_cache,
            head_dim=head_dim,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            scale=head_dim ** -0.5,
        )
        mx.eval(b)
        d_max, d_p9999 = diff(a, b)
        results[f"L{layer_idx}_S{seq_len}_max"] = d_max
        results[f"L{layer_idx}_S{seq_len}_p9999"] = d_p9999

# Full prefill pass comparison
P = 10
x_prefill = mx.random.normal((1, P, cfg.hidden_size)).astype(mx.bfloat16)
mx.eval(x_prefill)
pos = mx.broadcast_to(mx.arange(P)[None, :], (1, P))
pos3 = mx.stack([pos, pos, pos], axis=0)
cos, sin = model.talker.model.rotary_emb(x_prefill, pos3)
rope = (cos, sin)

# Upstream prefill: per-layer standard KVCache.
from mlx_audio.lm.models.cache import KVCache
u_caches = [KVCache() for _ in range(cfg.num_hidden_layers)]
h_u = x_prefill
for i, layer in enumerate(model.talker.model.layers):
    h_u = layer(h_u, rope, None, u_caches[i])
mx.eval(h_u)

# Vendored prefill: shared StaticKVCache.
v_cache = V.StaticKVCache(
    keys=[mx.zeros((1, cfg.num_key_value_heads, 16, head_dim), dtype=mx.bfloat16) for _ in range(cfg.num_hidden_layers)],
    values=[mx.zeros((1, cfg.num_key_value_heads, 16, head_dim), dtype=mx.bfloat16) for _ in range(cfg.num_hidden_layers)],
    pos=0,
)
h_v = x_prefill
for i, layer in enumerate(model.talker.model.layers):
    h_v = V.talker_layer_forward(
        layer, layer.self_attn.qkv_fused, i, h_v, rope, v_cache,
        head_dim=head_dim,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        scale=head_dim ** -0.5,
    )
mx.eval(h_v)

d_max, d_p9999 = diff(h_u, h_v)
results["full_prefill_max"] = d_max
results["full_prefill_p9999"] = d_p9999

print("EQUIV_RESULTS=" + json.dumps(results), flush=True)
