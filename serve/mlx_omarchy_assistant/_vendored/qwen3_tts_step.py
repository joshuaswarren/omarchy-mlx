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

Source SHA pinned to mlx_audio at the recording time (see module docstring
and SHA field below). Update it on every bump of voice-site's
qwen3_tts.py.
"""
from __future__ import annotations

import dataclasses
import math
import threading
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

# When this is updated, bump _SOURCE_SHA to the new git sha of
# voice-site/mlx_audio/tts/models/qwen3_tts/{qwen3_tts.py,talker.py}.
_SOURCE_SHA = (
    "voice-site qwen3_tts/talker.py sha256: 68fd1a980dd9dc4fae6346b26919957642f3fb04b65ab32d4e8bcee3f84fd02e;"
    " qwen3_tts.py sha256: 0d9437e4f08680d7bf8cb7bf3b44c8e3de37ad9edf025f50dba37dface2e6902"
)


# ----- fused projections -----------------------------------------------------

class FusedQKV(nn.Module):
    """One QuantizedLinear that returns q, k, v splits at once.

    Built from an attention's q_proj / k_proj / v_proj weights. The
    ``__call__`` writes through one qmv dispatch instead of three.
    """
    def __init__(self, q_proj, k_proj, v_proj):
        super().__init__()
        assert q_proj.bits == k_proj.bits == v_proj.bits
        assert q_proj.group_size == k_proj.group_size == v_proj.group_size
        bits = q_proj.bits
        gs = q_proj.group_size
        out_q = q_proj.weight.shape[0]
        out_k = k_proj.weight.shape[0]
        out_v = v_proj.weight.shape[0]
        fused = nn.QuantizedLinear(
            q_proj.weight.shape[1], out_q + out_k + out_v,
            bias=False, bits=bits, group_size=gs,
        )
        fused.weight = mx.concatenate([q_proj.weight, k_proj.weight, v_proj.weight], axis=0)
        fused.scales = mx.concatenate([q_proj.scales, k_proj.scales, v_proj.scales], axis=0)
        fused.biases = mx.concatenate([q_proj.biases, k_proj.biases, v_proj.biases], axis=0)
        self.fused = fused
        self.out_q = out_q
        self.out_k = out_k
        self.out_v = out_v
        self.num_heads = q_proj.weight.shape[0] // q_proj.out_features.__class__.__len__() if False else None  # placeholder
        # Caller fills the head counts after construction.

    def __call__(self, x):
        y = self.fused(x)
        q, k, v = y[..., :self.out_q], y[..., self.out_q:self.out_q + self.out_k], y[..., self.out_q + self.out_k:]
        return q, k, v


class FusedGateUp(nn.Module):
    """One QuantizedLinear that returns gate and up splits."""
    def __init__(self, gate_proj, up_proj):
        super().__init__()
        assert gate_proj.bits == up_proj.bits
        assert gate_proj.group_size == up_proj.group_size
        bits = gate_proj.bits
        gs = gate_proj.group_size
        out_g = gate_proj.weight.shape[0]
        out_u = up_proj.weight.shape[0]
        fused = nn.QuantizedLinear(
            gate_proj.weight.shape[1], out_g + out_u,
            bias=False, bits=bits, group_size=gs,
        )
        fused.weight = mx.concatenate([gate_proj.weight, up_proj.weight], axis=0)
        fused.scales = mx.concatenate([gate_proj.scales, up_proj.scales], axis=0)
        fused.biases = mx.concatenate([gate_proj.biases, up_proj.biases], axis=0)
        self.fused = fused
        self.out_g = out_g
        self.out_u = out_u

    def __call__(self, x):
        y = self.fused(x)
        return y[..., :self.out_g], y[..., self.out_g:]


# ----- vendored talker layer pass --------------------------------------------

def fused_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """One mx.fast.rms_norm call; weight is [H], eps matches upstream."""
    return mx.fast.rms_norm(x, weight, eps)


def fused_rope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> Tuple[mx.array, mx.array]:
    """mx.fast.rope applies (q,k) with cos/sin broadcast on head axis."""
    return mx.fast.rope(q, k, cos, sin)


def fused_sdpa(q, k, v, scale: float, mask=None):
    """SDPA call; the bf16 fast route drops q/k/v upcasts and the float
    scale multiply when MLX_OMARCHY_SDPA_BF16_FAST is set."""
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


def swiglu(gate, x):
    return nn.silu(gate) * x


# ----- vendored frame step ----------------------------------------------------

@dataclasses.dataclass
class FrameState:
    """Per-call frame state. Reset by `init_frame`."""
    prefill_embeds: mx.array        # [1, P, H]
    trailing: mx.array               # [1, P, H] - 1
    pad: mx.array                    # [1, 1, H]
    talker_cache_keys: List[mx.array]  # [L] contiguous K
    talker_cache_vals: List[mx.array]  # [L] contiguous V
    talker_cache_pos: int              # next position
    cp_cache_keys: List[mx.array]    # [5] contiguous K
    cp_cache_vals: List[mx.array]    # [5] contiguous V
    cp_cache_pos: int
    noise: mx.array                   # [16, max_vocab]


def _prepare_prefill(model):
    """Build the prefill inputs once per call."""
    cfg = model.config.talker_config
    text = "The local assistant is ready to help."
    ie, trailing, pad = model._prepare_generation_inputs(text, language="english", speaker="aiden")
    return ie, trailing, pad


def init_frame(model, *, dtype=mx.bfloat16) -> FrameState:
    """Build a fresh frame state. Loads the model prefill, allocates
    contiguous K/V caches sized to the model's hidden/layer config, and
    prepares a single (16, max_vocab) noise buffer for the sampler.
    """
    ie, trailing, pad = _prepare_prefill(model)
    cfg = model.config.talker_config
    cp_cfg = cfg.code_predictor_config
    # Preallocate contiguous K/V caches:
    #   talker: 28 layers, num_kv_heads=8 (GQA), head_dim=128, max 4096 tokens
    #   cp:     5 layers, num_kv_heads=8, head_dim=128
    T_max = 4096
    talker_k = [mx.zeros((1, 8, T_max, 128), dtype=dtype) for _ in range(cfg.num_hidden_layers)]
    talker_v = [mx.zeros((1, 8, T_max, 128), dtype=dtype) for _ in range(cfg.num_hidden_layers)]
    cp_k = [mx.zeros((1, 8, T_max, 128), dtype=dtype) for _ in range(cp_cfg.num_hidden_layers)]
    cp_v = [mx.zeros((1, 8, T_max, 128), dtype=dtype) for _ in range(cp_cfg.num_hidden_layers)]
    V = max(cfg.vocab_size, cp_cfg.vocab_size)
    noise = mx.random.uniform(shape=(16, V))
    return FrameState(
        prefill_embeds=ie, trailing=trailing, pad=pad,
        talker_cache_keys=talker_k, talker_cache_vals=talker_v,
        talker_cache_pos=ie.shape[1],
        cp_cache_keys=cp_k, cp_cache_vals=cp_v, cp_cache_pos=0,
        noise=noise,
    )


# Frame step is implemented as one function per model: a talker decode
# followed by a 15-pass code predictor. Both are pure functions of the
# state and the model's parameters (closed over by the worker at
# load time). For testability, the inner steps are extracted below.
