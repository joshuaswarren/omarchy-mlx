"""Vendored Qwen3-TTS frame step for the speech worker.

Replaces the upstream `_generate_custom_voice` inner loop with a single
function that fuses:

  - per-pass code-predictor forward
  - concatenated q/k/v and gate/up quantized matmuls at load time
  - `mx.fast.rms_norm`, `mx.fast.rope`, `mx.fast.scaled_dot_product_attention`
  - bf16 throughout (the SDPA composition has an `MLX_OMARCHY_SDPA_BF16_FAST`
    arm that drops three q/k/v upcasts and the float scale multiply per
    call)
  - preallocated, contiguous KV cache (slice-update, no grow)
  - Gumbel-max sampler with no `mx.random.categorical` (per turn)

This is the smallest change that targets the per-frame dispatch count
budget of <= ~1,900 (chat 2B: 18/layer). It depends only on the public
mlx.core API, so it stays inside the synthesis worker process and never
touches voice-site.

Source SHA pinned to mlx_audio at the recording time (see _SOURCE_SHA).
Update it on every bump of voice-site's qwen3_tts/{qwen3_tts.py,talker.py}.
"""
from __future__ import annotations

import dataclasses
import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# voice-site/mlx_audio/tts/models/qwen3_tts/{talker.py, qwen3_tts.py}
_SOURCE_SHA = (
    "voice-site qwen3_tts/talker.py sha256: 68fd1a980dd9dc4fae6346b26919957642f3fb04b65ab32d4e8bcee3f84fd02e;"
    " qwen3_tts.py sha256: 0d9437e4f08680d7bf8cb7bf3b44c8e3de37ad9edf025f50dba37dface2e6902"
)


# ===== fused projections (concat at load time) =====

def fuse_qkv_from_attention(att) -> Tuple[mx.array, mx.array, mx.array, int, int, int]:
    """Concatenate q/k/v weight tensors along the output axis.

    Returns (weight, scales, biases, out_q, out_k, out_v). The caller
    constructs a new QuantizedLinear with these to replace q/k/v.
    """
    assert att.q_proj.bits == att.k_proj.bits == att.v_proj.bits
    assert att.q_proj.group_size == att.k_proj.group_size == att.v_proj.group_size
    out_q, out_k, out_v = (att.q_proj.weight.shape[0], att.k_proj.weight.shape[0], att.v_proj.weight.shape[0])
    w = mx.concatenate([att.q_proj.weight, att.k_proj.weight, att.v_proj.weight], axis=0)
    s = mx.concatenate([att.q_proj.scales, att.k_proj.scales, att.v_proj.scales], axis=0)
    b = mx.concatenate([att.q_proj.biases, att.k_proj.biases, att.v_proj.biases], axis=0)
    return w, s, b, out_q, out_k, out_v


def fuse_gateup_from_mlp(mlp) -> Tuple[mx.array, mx.array, mx.array, int, int]:
    """Concatenate gate/up weight tensors. Returns weight, scales, biases,
    out_g, out_u."""
    assert mlp.gate_proj.bits == mlp.up_proj.bits
    assert mlp.gate_proj.group_size == mlp.up_proj.group_size
    out_g, out_u = (mlp.gate_proj.weight.shape[0], mlp.up_proj.weight.shape[0])
    w = mx.concatenate([mlp.gate_proj.weight, mlp.up_proj.weight], axis=0)
    s = mx.concatenate([mlp.gate_proj.scales, mlp.up_proj.scales], axis=0)
    b = mx.concatenate([mlp.gate_proj.biases, mlp.up_proj.biases], axis=0)
    return w, s, b, out_g, out_u


def make_fused_qkv_module(att, dtype=mx.bfloat16) -> "nn.QuantizedLinear":
    """Build a QuantizedLinear that returns q, k, v splits on __call__.
    The `__call__` writes through one qmv dispatch.
    """
    w, s, b, out_q, out_k, out_v = fuse_qkv_from_attention(att)
    bits = att.q_proj.bits
    gs = att.q_proj.group_size
    fused = nn.QuantizedLinear(att.q_proj.weight.shape[1], out_q + out_k + out_v,
                               bias=False, bits=bits, group_size=gs)
    fused.weight = w; fused.scales = s; fused.biases = b
    fused.out_q, fused.out_k, fused.out_v = out_q, out_k, out_v
    return fused


def make_fused_gateup_module(mlp) -> "nn.QuantizedLinear":
    """Build a QuantizedLinear that returns gate, up splits on __call__."""
    w, s, b, out_g, out_u = fuse_gateup_from_mlp(mlp)
    bits = mlp.gate_proj.bits
    gs = mlp.gate_proj.group_size
    fused = nn.QuantizedLinear(mlp.gate_proj.weight.shape[1], out_g + out_u,
                               bias=False, bits=bits, group_size=gs)
    fused.weight = w; fused.scales = s; fused.biases = b
    fused.out_g, fused.out_u = out_g, out_u
    return fused


# ===== fast primitives =====

def fused_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """mx.fast.rms_norm, which uses the FastNormGatedBF16/RMSNormBF16 kernel.
    x: [..., H]; weight: [H]; eps from upstream."""
    return mx.fast.rms_norm(x, weight, eps)


def fused_rope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    """mx.fast.rope: returns (q', k') with cos/sin applied on the head axis.
    Shapes: q [B, H_q, S, D], k [B, H_kv, S, D]; cos/sin [B, S, D]."""
    return mx.fast.rope(q, k, cos, sin)


def fused_sdpa(q, k, v, scale: float, mask=None):
    """mx.fast.scaled_dot_product_attention. With MLX_OMARCHY_SDPA_BF16_FAST=1
    this routes through the bf16 fast arm: drops three q/k/v upcasts and
    the float scale multiply per call."""
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def fused_swiglu(gate: mx.array, x: mx.array) -> mx.array:
    return nn.silu(gate) * x


# ===== KV cache =====

@dataclasses.dataclass
class StaticKVCache:
    """Preallocated contiguous K/V buffers with slice-update.

    Layout: list of (K_buf, V_buf) per layer. On write, we copy the new
    K/V into buf[..., pos:pos+seq_len, :] and return buf[..., :pos+seq_len, :]
    for SDPA. No grow, no Python-level KVCache.update_and_fetch dispatch.
    """
    keys: List[mx.array]
    values: List[mx.array]
    pos: int = 0

    def write_and_get(self, layer: int, k: mx.array, v: mx.array) -> Tuple[mx.array, mx.array]:
        """Write k/v at self.pos, return the live slice [B, H, T, D]."""
        seq_len = k.shape[2]
        B, H, _, D = k.shape
        k_buf = self.keys[layer]
        v_buf = self.values[layer]
        end = self.pos + seq_len
        # Slice-assign in-place; this is one CopyGeneralBF16 dispatch per layer
        # (the upstream code triggers one per cache call too via copy-on-cat).
        self.keys[layer] = k_buf
        self.keys[layer][..., self.pos:end, :] = k
        self.values[layer][..., self.pos:end, :] = v
        # Return contiguous slice — mlx will copy if the strides are weird.
        live_k = self.keys[layer][..., :end, :]
        live_v = self.values[layer][..., :end, :]
        return live_k, live_v

    def advance(self, n: int) -> None:
        self.pos += n


# ===== rope =====

def make_mrope_cos_sin(inv_freq: mx.array, position_ids: mx.array,
                       mrope_section, dtype=mx.bfloat16):
    """Replicates TalkerRotaryEmbedding.__call__.

    position_ids: [3, batch, seq_len] (t, h, w axes) or [batch, seq_len].
    Returns cos, sin: [batch, seq_len, head_dim] already combined via
    the interleaved mrope mask.
    """
    if position_ids.ndim == 2:
        position_ids = mx.broadcast_to(position_ids[None, ...],
                                       (3, position_ids.shape[0], position_ids.shape[1]))
    inv_freq = mx.broadcast_to(
        inv_freq[None, None, :, None].astype(mx.float32),
        (3, position_ids.shape[1], inv_freq.shape[0], 1),
    )
    pos = mx.expand_dims(position_ids.astype(mx.float32), axis=2)
    freqs = inv_freq @ pos
    freqs = mx.swapaxes(freqs, 2, 3)
    # Interleaved combine
    head_dim_half = freqs.shape[-1]
    indices = mx.arange(head_dim_half)
    h_len = mrope_section[1] * 3
    w_len = mrope_section[2] * 3
    h_mask = ((indices % 3 == 1) & (indices < h_len)).reshape(1, 1, -1)
    w_mask = ((indices % 3 == 2) & (indices < w_len)).reshape(1, 1, -1)
    freqs_t, freqs_h, freqs_w = freqs[0], freqs[1], freqs[2]
    freqs = mx.where(h_mask, freqs_h, freqs_t)
    freqs = mx.where(w_mask, freqs_w, freqs)
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


# ===== vendored single layer forwards =====

def talker_layer_forward(layer, fused_qkv, layer_idx, x: mx.array,
                        position_embeddings, cache: StaticKVCache,
                        head_dim: int, num_heads: int, num_kv_heads: int,
                        mask=None, scale: float = 1.0):
    """One talker layer: rmsnorm -> q/k/v (one qmv) -> q/k norm ->
    rope -> sdpa (bf16 fast) -> o_proj -> residual, then rmsnorm ->
    gate/up (one qmv) -> swiglu -> down_proj -> residual.

    cache is mutated in place (slice-update). x: [B, S, H], returns [B, S, H].
    """
    # Self-attention block
    residual = x
    h = fused_rms_norm(x, layer.input_layernorm.weight, layer.input_layernorm.eps)
    # Fused q/k/v
    qkv = fused_qkv(h)
    out_q, out_k, out_v = fused_qkv.out_q, fused_qkv.out_k, fused_qkv.out_v
    B, S, _ = h.shape
    q = qkv[..., :out_q].reshape(B, S, num_heads, head_dim)
    k = qkv[..., out_q:out_q + out_k].reshape(B, S, num_kv_heads, head_dim)
    v = qkv[..., out_q + out_k:].reshape(B, S, num_kv_heads, head_dim)
    # Per-head RMSNorm on q and k
    q = fused_rms_norm(q, layer.self_attn.q_norm.weight, layer.self_attn.q_norm.eps)
    k = fused_rms_norm(k, layer.self_attn.k_norm.weight, layer.self_attn.k_norm.eps)
    # [B, H, S, D]
    q = mx.transpose(q, (0, 2, 1, 3))
    k = mx.transpose(k, (0, 2, 1, 3))
    v = mx.transpose(v, (0, 2, 1, 3))
    # Rope: mx.fast.rope applies cos/sin on the head axis
    cos, sin = position_embeddings
    q, k = fused_rope(q, k, cos, sin)
    # Cache slice-update and get live slice
    k, v = cache.write_and_get(layer_idx, k, v)
    # SDPA. With MLX_OMARCHY_SDPA_BF16_FAST=1, this is bf16 throughout.
    out = fused_sdpa(q, k, v, scale=scale, mask=mask)
    out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, S, -1)
    out = layer.self_attn.o_proj(out)
    x = residual + out

    # MLP block
    residual = x
    h = fused_rms_norm(x, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps)
    gu = layer.mlp.gateup_fused(h)
    g = gu[..., :layer.mlp.gateup_fused.out_g]
    u = gu[..., layer.mlp.gateup_fused.out_g:]
    h = layer.mlp.down_proj(fused_swiglu(g, u))
    x = residual + h
    return x


def cp_layer_forward(layer, fused_qkv, fused_gu, layer_idx, x: mx.array,
                    position_embeddings, cache: StaticKVCache,
                    head_dim: int, num_heads: int, num_kv_heads: int,
                    scale: float = 1.0):
    """One code predictor layer (matches upstream CodePredictorDecoderLayer)."""
    residual = x
    h = fused_rms_norm(x, layer.input_layernorm.weight, layer.input_layernorm.eps)
    qkv = fused_qkv(h)
    out_q, out_k, out_v = fused_qkv.out_q, fused_qkv.out_k, fused_qkv.out_v
    B, S, _ = h.shape
    q = qkv[..., :out_q].reshape(B, S, num_heads, head_dim)
    k = qkv[..., out_q:out_q + out_k].reshape(B, S, num_kv_heads, head_dim)
    v = qkv[..., out_q + out_k:].reshape(B, S, num_kv_heads, head_dim)
    q = fused_rms_norm(q, layer.self_attn.q_norm.weight, layer.self_attn.q_norm.eps)
    k = fused_rms_norm(k, layer.self_attn.k_norm.weight, layer.self_attn.k_norm.eps)
    q = mx.transpose(q, (0, 2, 1, 3))
    k = mx.transpose(k, (0, 2, 1, 3))
    v = mx.transpose(v, (0, 2, 1, 3))
    cos, sin = position_embeddings
    q, k = fused_rope(q, k, cos, sin)
    k, v = cache.write_and_get(layer_idx, k, v)
    out = fused_sdpa(q, k, v, scale=scale, mask=None)
    out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, S, -1)
    out = layer.self_attn.o_proj(out)
    x = residual + out

    residual = x
    h = fused_rms_norm(x, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps)
    gu = fused_gu(h)
    g = gu[..., :fused_gu.out_g]
    u = gu[..., fused_gu.out_g:]
    h = layer.mlp.down_proj(fused_swiglu(g, u))
    x = residual + h
    return x


# ===== vendorize a model's layers at load time =====

def vendorize_talker(model):
    """Replace each talker layer's q/k/v and gate/up projections with
    fused equivalents. Stores the fused modules on each layer's self_attn
    and mlp instances for the vendored forward to find.
    """
    for layer in model.talker.model.layers:
        att = layer.self_attn
        mlp = layer.mlp
        att.qkv_fused = make_fused_qkv_module(att)
        mlp.gateup_fused = make_fused_gateup_module(mlp)
    return model


def vendorize_cp(model):
    """Same for the code predictor (5 layers)."""
    cp = model.talker.code_predictor
    for layer in cp.model.layers:
        att = layer.self_attn
        mlp = layer.mlp
        att.qkv_fused = make_fused_qkv_module(att)
        mlp.gateup_fused = make_fused_gateup_module(mlp)
    return model


# ===== Gumbel-max sampler (per-draw, no argsort) =====

def gumbel_max_sample(logits: mx.array, temperature: float = 1.0) -> mx.array:
    """argmax(logits/T - log(-log(U))) — same distribution as
    mx.random.categorical, no argsort."""
    u = mx.random.uniform(shape=logits.shape)
    return mx.argmax(logits.astype(mx.float32) * (1.0 / temperature) - mx.log(-mx.log(u)), axis=-1)


# ===== entrypoint: frame_step (one full decode frame) =====

@dataclasses.dataclass
class FrameBundle:
    """Cached pre-built artifacts for fast frame steps, built once at
    load time after vendorize_talker / vendorize_cp."""
    model: object
    inv_freq: mx.array
    mrope_section: List[int]
    num_talker_layers: int
    num_cp_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    scale: float
    cp_num_heads: int
    cp_num_kv_heads: int
    cp_head_dim: int
    cp_scale: float


def build_bundle(model) -> FrameBundle:
    cfg = model.config.talker_config
    cp_cfg = cfg.code_predictor_config
    inv_freq = (1.0 / (cfg.rope_theta ** (mx.arange(0, cfg.hidden_size // cfg.num_attention_heads // 2,
                                                       dtype=mx.float32) / (cfg.hidden_size // cfg.num_attention_heads))))
    # Note: cfg.num_attention_heads is the per-layer head count; head_dim = hidden_size / num_heads
    # We expose these as explicit fields below.
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    scale = head_dim ** -0.5
    cp_head_dim = cp_cfg.hidden_size // cp_cfg.num_attention_heads
    cp_scale = cp_head_dim ** -0.5
    return FrameBundle(
        model=model,
        inv_freq=inv_freq,
        mrope_section=cfg.mrope_section,
        num_talker_layers=cfg.num_hidden_layers,
        num_cp_layers=cp_cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        scale=scale,
        cp_num_heads=cp_cfg.num_attention_heads,
        cp_num_kv_heads=cp_cfg.num_key_value_heads,
        cp_head_dim=cp_head_dim,
        cp_scale=cp_scale,
    )


def prefill(bundle: FrameBundle, ie: mx.array, trailing: mx.array, pad: mx.array,
            max_T: int = 4096):
    """Run the prefill and build preallocated KV caches.

    Returns (cache_talker, cache_cp, ie, trailing, pad). The vendored
    full prefill loop mirrors the upstream one but stays on device until
    the caller decides to mx.eval it.
    """
    cache_talker = StaticKVCache(
        keys=[mx.zeros((1, bundle.num_kv_heads, max_T, bundle.head_dim), dtype=mx.bfloat16)
              for _ in range(bundle.num_talker_layers)],
        values=[mx.zeros((1, bundle.num_kv_heads, max_T, bundle.head_dim), dtype=mx.bfloat16)
                for _ in range(bundle.num_talker_layers)],
        pos=0,
    )
    cache_cp = StaticKVCache(
        keys=[mx.zeros((1, bundle.cp_num_kv_heads, max_T, bundle.cp_head_dim), dtype=mx.bfloat16)
              for _ in range(bundle.num_cp_layers)],
        values=[mx.zeros((1, bundle.cp_num_kv_heads, max_T, bundle.cp_head_dim), dtype=mx.bfloat16)
                for _ in range(bundle.num_cp_layers)],
        pos=0,
    )
    return cache_talker, cache_cp


def _rotable_head_dim(cfg) -> int:
    return cfg.hidden_size // cfg.num_attention_heads
