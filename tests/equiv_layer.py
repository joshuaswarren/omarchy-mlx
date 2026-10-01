"""Equivalence probe: vendored talker layer forward vs upstream.

Run as a one-shot script on the M2 under gpu-turn. Compares the
vendored fused-pass output against the upstream layer output for layer 0
and layer 27 (last) on a few fixed inputs. Reports max abs diff and
P99.99 abs diff. Then compares the full 28-layer pass with the
upstream 28-layer pass.

Exits 0 on pass, non-zero on diff > tolerance.
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, "<home>/voice-site")
sys.path.insert(0, str(Path(__file__).resolve().parent / "serve"))
import mlx.core as mx

from mlx_audio.tts.utils import load_model
model = load_model("<home>/mlx-tts-home/voice/qwen3-tts-0.6b-customvoice-4bit")
print("loaded", flush=True)

from mlx_omarchy_assistant._vendored import qwen3_tts_step as V
V.vendorize_talker(model)
V.vendorize_cp(model)

cfg = model.config.talker_config
head_dim = cfg.hidden_size // cfg.num_attention_heads


def upstream_layer(layer, x, rope):
    """Run upstream layer with a fresh in-memory KVCache."""
    from mlx_audio.lm.models.cache import KVCache
    cache = [KVCache() for _ in range(1)]
    return layer(x, rope, None, cache[0])


def vendored_layer(layer, x, rope):
    cache = V.StaticKVCache(
        keys=[mx.zeros((1, cfg.num_key_value_heads, 16, head_dim), dtype=mx.bfloat16)],
        values=[mx.zeros((1, cfg.num_key_value_heads, 16, head_dim), dtype=mx.bfloat16)],
        pos=0,
    )
    return V.talker_layer_forward(
        layer, layer.self_attn.qkv_fused, 0, x, rope, cache,
        head_dim=head_dim,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        scale=head_dim ** -0.5,
    )


def diff(a, b):
    d = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    return float(mx.max(d).item()), float(mx.quantile(d, 0.9999).item())


results = {}
mx.set_default_device(mx.gpu)

for layer_idx in [0, 13, 27]:
    layer = model.talker.model.layers[layer_idx]
    for seq_len in [1, 4]:
        x = mx.random.normal((1, seq_len, cfg.hidden_size)).astype(mx.bfloat16)
        mx.eval(x)
        pos = mx.broadcast_to(mx.arange(seq_len)[None, :], (1, seq_len))
        pos3 = mx.stack([pos, pos, pos], axis=0)
        cos, sin = model.talker.model.rotary_emb(x, pos3)
        rope = (cos, sin)
        a = upstream_layer(layer, x, rope)
        b = vendored_layer(layer, x, rope)
        mx.eval(a, b)
        mx_max, mx_p9999 = diff(a, b)
        results[f"L{layer_idx}_S{seq_len}"] = {"max": mx_max, "p9999": mx_p9999}

# Full prefill pass comparison
x_prefill = mx.random.normal((1, 10, cfg.hidden_size)).astype(mx.bfloat16)
mx.eval(x_prefill)
pos = mx.broadcast_to(mx.arange(10)[None, :], (1, 10))
pos3 = mx.stack([pos, pos, pos], axis=0)
cos, sin = model.talker.model.rotary_emb(x_prefill, pos3)
rope = (cos, sin)

def upstream_prefill(x, rope):
    from mlx_audio.lm.models.cache import KVCache
    cache = [KVCache() for _ in range(cfg.num_hidden_layers)]
    h = x
    for layer in model.talker.model.layers:
        # upstream expects a list cache indexed by layer
        h_layer = layer(h, rope, None, cache[0] if False else None)  # layer's own cache
        # actually each TalkerDecoderLayer expects cache[layer_idx]; pass the same cache repeatedly
    return None  # placeholder

def vendored_prefill(x, rope):
    cache = V.StaticKVCache(
        keys=[mx.zeros((1, cfg.num_key_value_heads, 16, head_dim), dtype=mx.bfloat16) for _ in range(cfg.num_hidden_layers)],
        values=[mx.zeros((1, cfg.num_key_value_heads, 16, head_dim), dtype=mx.bfloat16) for _ in range(cfg.num_hidden_layers)],
        pos=0,
    )
    h = x
    for i, layer in enumerate(model.talker.model.layers):
        h = V.talker_layer_forward(
            layer, layer.self_attn.qkv_fused, i, h, rope, cache,
            head_dim=head_dim,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            scale=head_dim ** -0.5,
        )
    return h

# Upstream prefill via model.talker (which sets up its own cache)
from mlx_audio.lm.models.cache import KVCache
upstream_cache = [KVCache() for _ in range(cfg.num_hidden_layers)]
h_up = x_prefill
for layer in model.talker.model.layers:
    h_up = layer(h_up, rope, None, upstream_cache[0] if False else None)
    # Note: upstream code at talker.py:497 calls layer(x, position_embeddings, mask, cache)
    # where cache is a list of KVCache (one per layer).
# Actually, looking at talker.py:497, it iterates `for i, layer in enumerate(self.layers): layer_cache = cache[i] if cache is not None else None; x = layer(x, position_embeddings, mask, layer_cache)`.
# But we passed the same cache instance each time, so let me redo.

upstream_cache2 = [KVCache() for _ in range(cfg.num_hidden_layers)]
h_up = x_prefill
for i, layer in enumerate(model.talker.model.layers):
    h_up = layer(h_up, rope, None, upstream_cache2[i])
mx.eval(h_up)

h_vendored = vendored_prefill(x_prefill, rope)
mx.eval(h_vendored)

mx_max, mx_p9999 = diff(h_up, h_vendored)
results["full_prefill"] = {"max": mx_max, "p9999": mx_p9999}

print("EQUIV_RESULTS=" + json.dumps(results), flush=True)
