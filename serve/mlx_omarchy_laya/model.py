"""Laya decision model in pure mlx.core ops.

Port of the pinned upstream DecisionModel
(convaiinnovations/laya revision 1c5edc17a7acd8701df6fc341c0d179f1c62c982,
files rl_common.py + transformers modeling_modernbert.py): a ModernBERT-large
encoder (28 layers, alternating full/sliding attention with per-type RoPE)
followed by a pre-norm transformer decision head that scores per-option
[MASK] markers, plus an escalate/answer action head.

Every op runs on the default MLX device. There is no CPU fallback and no
model-name branching inside the math: the caller selects the device.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx


# --------------------------------------------------------------------------- config


@dataclass
class EncoderConfig:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    norm_eps: float
    local_attention: int
    global_attn_every_n_layers: int
    layer_types: list
    global_rope_theta: float
    local_rope_theta: float

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def is_local(self) -> list:
        """True for sliding_attention layers, False for full_attention layers."""
        return [t != "full_attention" for t in self.layer_types]


def load_encoder_config(encoder_dir: str | Path) -> EncoderConfig:
    with open(Path(encoder_dir) / "config.json") as f:
        c = json.load(f)
    rope = c.get("rope_parameters") or {}
    global_theta = rope.get("full_attention", {}).get("rope_theta", c.get("global_rope_theta", 160000.0))
    local_theta = rope.get("sliding_attention", {}).get("rope_theta", c.get("local_rope_theta", 10000.0))
    return EncoderConfig(
        hidden_size=c["hidden_size"],
        num_attention_heads=c["num_attention_heads"],
        num_hidden_layers=c["num_hidden_layers"],
        intermediate_size=c["intermediate_size"],
        norm_eps=c.get("norm_eps", 1e-5),
        local_attention=c.get("local_attention", 128),
        global_attn_every_n_layers=c.get("global_attn_every_n_layers", 3),
        layer_types=c["layer_types"],
        global_rope_theta=global_theta,
        local_rope_theta=local_theta,
    )


def _canonical_key(name: str) -> str:
    """Map checkpoint key variants to the canonical (upstream source) names.

    The aac6fef/laya-mlx pack renames some decision-head keys relative to the
    source checkpoint (in_proj_weight -> in_proj.weight, scorer.N ->
    scorer.layers.N, act_head.N -> act_head.layers.N). Tensor shapes and
    semantics are identical; canonical form here is the source layout.
    """
    if ".self_attn.in_proj.weight" in name:
        name = name.replace(".self_attn.in_proj.weight", ".self_attn.in_proj_weight")
    if ".self_attn.in_proj.bias" in name:
        name = name.replace(".self_attn.in_proj.bias", ".self_attn.in_proj_bias")
    if name.startswith("scorer.layers."):
        rest = name[len("scorer.layers."):]
        name = "scorer." + rest
    if name.startswith("act_head.layers."):
        rest = name[len("act_head.layers."):]
        name = "act_head." + rest
    return name


def load_weights(model_dir: str | Path, dtype) -> dict:
    """Load the converted safetensors, normalize keys, cast to the run dtype.

    The checkpoint is produced by mlx_omarchy_laya.convert (which already
    emits canonical names) or is an aac6fef/laya-mlx pack dir (aliases are
    normalized here). Safetensors only — no pickle, no trust_remote_code.
    """
    path = Path(model_dir) / "model.safetensors"
    weights = mx.load(str(path))
    return {_canonical_key(k): v.astype(dtype) for k, v in weights.items()}


# --------------------------------------------------------------------------- primitives


def _layer_norm(x, weight, bias, eps):
    if bias is None:
        bias = mx.zeros(x.shape[-1], dtype=x.dtype)
    return mx.fast.layer_norm(x, weight, bias, eps)


def _linear(x, w, b=None):
    return x @ w.T + (b if b is not None else 0)


def rope_tables(cfg: EncoderConfig, max_len: int) -> dict:
    """HF ModernBERT rotary tables, float32, positions 0..max_len-1, one
    (cos, sin) pair per theta: inv_freq = theta**-(2i/d); emb = [freqs, freqs].

    Built once at engine load. Row t depends only on t and theta, so a
    gathered row is bit-identical to recomputing it for a shorter sequence,
    and a request never evaluates cos/sin (each evaluation is a trig-argument
    readback that stalls the GPU).
    """
    hd = cfg.head_dim
    tables = {}
    for theta in {cfg.local_rope_theta, cfg.global_rope_theta}:
        inv_freq = mx.power(theta, -(mx.arange(0, hd, 2, dtype=mx.float32) / hd))
        freqs = mx.outer(mx.arange(max_len, dtype=mx.float32), inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        tables[theta] = (emb.cos(), emb.sin())
    return tables


def _rotate_half(x):
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(q, k, cos, sin):
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


def _additive_pad_mask(attention_mask, dtype):
    """[B, T] 0/1 -> additive [B, 1, 1, T]: 0 keep, dtype-min masked."""
    keep = attention_mask.astype(dtype)
    return (1.0 - keep) * mx.finfo(dtype).min


# --------------------------------------------------------------------------- encoder


def _encoder_attention(x, w, prefix, cfg, is_local, positions, pad_bias, rope):
    B, T, D = x.shape
    H, hd = cfg.num_attention_heads, cfg.head_dim
    qkv = _linear(x, w[f"{prefix}.attn.Wqkv.weight"])
    qkv = qkv.reshape(B, T, 3, H, hd)
    q = qkv[:, :, 0].transpose(0, 2, 1, 3)
    k = qkv[:, :, 1].transpose(0, 2, 1, 3)
    v = qkv[:, :, 2].transpose(0, 2, 1, 3)
    theta = cfg.local_rope_theta if is_local else cfg.global_rope_theta
    cos, sin = rope[theta]
    cos = cos[positions].astype(x.dtype)
    sin = sin[positions].astype(x.dtype)
    q, k = _apply_rope(q, k, cos, sin)
    scores = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5)
    if is_local:
        window = cfg.local_attention // 2
        rows = mx.arange(T)
        near = mx.abs(rows[:, None] - rows[None, :]) <= window  # [T, T]
        # pad mask everywhere, plus window restriction on top
        scores = scores + pad_bias[:, None, None, :]
        win_bias = mx.where(near, 0.0, mx.finfo(x.dtype).min).astype(x.dtype)
        scores = scores + win_bias
    else:
        scores = scores + pad_bias[:, None, None, :]
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
    out = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return _linear(out, w[f"{prefix}.attn.Wo.weight"])


def _encoder_mlp(x, w, prefix):
    wi = _linear(x, w[f"{prefix}.mlp.Wi.weight"])
    gate, up = wi[..., : wi.shape[-1] // 2], wi[..., wi.shape[-1] // 2 :]
    hidden = _gelu(gate) * up
    return _linear(hidden, w[f"{prefix}.mlp.Wo.weight"])


def _gelu(x):
    """Exact GELU (erf), matching nn.GELU()."""
    return 0.5 * x * (1.0 + mx.erf(x / mx.sqrt(2.0)))


def encoder_forward(w, cfg, input_ids, attention_mask, rope):
    """ModernBERT body. input_ids [B, T] int32, attention_mask [B, T] 0/1,
    rope from rope_tables() covering at least T positions.

    Returns last_hidden_state [B, T, D] after final_norm.
    """
    x = mx.take(w["encoder.embeddings.tok_embeddings.weight"], input_ids, axis=0)
    x = _layer_norm(x, w["encoder.embeddings.norm.weight"], None, cfg.norm_eps)
    pad_bias = _additive_pad_mask(attention_mask, x.dtype)
    positions = mx.arange(input_ids.shape[1])
    is_local = cfg.is_local
    for i in range(cfg.num_hidden_layers):
        p = f"encoder.layers.{i}"
        if i == 0:  # upstream: layer 0 has attn_norm = Identity
            h = x
        else:
            h = _layer_norm(x, w[f"{p}.attn_norm.weight"], None, cfg.norm_eps)
        x = x + _encoder_attention(h, w, p, cfg, is_local[i], positions, pad_bias, rope)
        x = x + _encoder_mlp(_layer_norm(x, w[f"{p}.mlp_norm.weight"], None, cfg.norm_eps), w, p)
    return _layer_norm(x, w["encoder.final_norm.weight"], None, cfg.norm_eps)


# --------------------------------------------------------------------------- decision head


def _head_self_attention(x, w, prefix, pad_bias):
    B, T, D = x.shape
    n_heads = max(1, D // 64)  # upstream: nhead = max(1, d // 64)
    hd = D // n_heads
    qkv = _linear(x, w[f"{prefix}.self_attn.in_proj_weight"], w[f"{prefix}.self_attn.in_proj_bias"])
    q, k, v = [t.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3) for t in mx.split(qkv, 3, axis=-1)]
    scores = (q @ k.transpose(0, 1, 3, 2)) * (hd ** -0.5) + pad_bias[:, None, None, :]
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
    out = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return _linear(out, w[f"{prefix}.self_attn.out_proj.weight"], w[f"{prefix}.self_attn.out_proj.bias"])


def _head_layer(x, w, prefix, pad_bias):
    h = _layer_norm(x, w[f"{prefix}.norm1.weight"], w[f"{prefix}.norm1.bias"], 1e-5)
    x = x + _head_self_attention(h, w, prefix, pad_bias)
    h = _layer_norm(x, w[f"{prefix}.norm2.weight"], w[f"{prefix}.norm2.bias"], 1e-5)
    h = _linear(h, w[f"{prefix}.linear1.weight"], w[f"{prefix}.linear1.bias"])
    h = mx.maximum(h, 0)  # nn.TransformerEncoderLayer default activation (ReLU)
    h = _linear(h, w[f"{prefix}.linear2.weight"], w[f"{prefix}.linear2.bias"])
    return x + h


def decision_head(w, hidden, attention_mask, marker_pos, marker_mask, qtype, head_layers: int):
    """Returns (logits [B, M] float32, act_logits [B, 2] float32)."""
    B, T, D = hidden.shape
    h = hidden + mx.take(w["type_emb.weight"], qtype, axis=0)[:, None, :]
    pad = attention_mask == 0
    pad_bias = _additive_pad_mask(attention_mask, h.dtype)
    for i in range(head_layers):
        h = _head_layer(h, w, f"head.layers.{i}", pad_bias)
    idx = mx.maximum(marker_pos, 0)[:, :, None]
    idx = mx.broadcast_to(idx, (B, marker_pos.shape[1], D))
    m = mx.take_along_axis(h, idx, axis=1)
    s = _layer_norm(m, w["scorer.0.weight"], w["scorer.0.bias"], 1e-5)
    s = _linear(s, w["scorer.1.weight"], w["scorer.1.bias"])
    s = _gelu(s)
    logits = _linear(s, w["scorer.3.weight"], w["scorer.3.bias"])[..., 0].astype(mx.float32)
    logits = mx.where(marker_mask, logits, -1e4)
    # action head sees the pooled CLS plus a detached summary of its own answer distribution
    p = mx.softmax(logits, axis=-1)
    k = mx.maximum(marker_mask.astype(mx.float32).sum(axis=-1), 2.0)
    ent = -(p * mx.log(mx.maximum(p, 1e-9))).sum(axis=-1) / mx.log(k)
    top2 = mx.sort(p, axis=-1)[..., -2:]  # ascending -> [second, first]
    feats = mx.concatenate(
        [top2[:, 1:2], (top2[:, 1:2] - top2[:, 0:1]), ent[:, None], (k / 255.0)[:, None]], axis=-1
    )
    pooled = h[:, 0, :].astype(mx.float32)
    a = _linear(mx.concatenate([pooled, feats], axis=-1),
                w["act_head.0.weight"].astype(mx.float32), w["act_head.0.bias"].astype(mx.float32))
    a = mx.maximum(a, 0)  # upstream nn.GELU-free ReLU in act head
    act_logits = _linear(a, w["act_head.2.weight"].astype(mx.float32), w["act_head.2.bias"].astype(mx.float32))
    return logits, act_logits


def forward(w, cfg, rope, input_ids, attention_mask, marker_pos, marker_mask, qtype,
            head_layers: int):
    hidden = encoder_forward(w, cfg, input_ids, attention_mask, rope)
    return decision_head(w, hidden, attention_mask, marker_pos, marker_mask, qtype, head_layers)
