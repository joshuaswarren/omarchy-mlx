"""Native omarchy implementation of the GLM DSA fused decode indexer
scan (upstream ``dsa_decode_scores``, metallib
``dsa_decode_scores_<t>_o<out>_h32_d128_t256``).

Contract (DSAIndexerScoresPrimitive / dsa_indexer.cpp:605-700 and
dsa_indexer.metal:72-137):
- q [B, 32, 1, 128] (post-RoPE), contiguous
- k [B, 1, S, 128], last-dim contiguous (cache slices allowed;
  row stride a multiple of 8 elements for the vectorized loads)
- w [B, 32] (scaled weights)
- out [B, 1, 1, S]; fp32 accumulation throughout, stored to the
  output dtype (input dtype by default, fp32 when fp32_scores=True)
- S >= 1024

Math per key position:
    out[b, 0, 0, s] = sum_h max(sum_d q[b,h,0,d] * k[b,0,s,d], 0) * w[b,h]

Two omarchy variants:
- ``dsa_decode_scores_omarchy``: fp32 chain end to end (f32 matmul,
  f32 relu/mul/sum), matching the metal kernel's fp32-accumulate
  contract, then one cast to the output dtype. Strictly tighter
  numerics than the deployed fallback, whose [B,32,1,S] relu/weight
  intermediates round to 16-bit storage three extra times.
- ``dsa_decode_scores_composed``: the deployed fallback chain in the
  input dtype (the graph that runs on Linux today).

Both run entirely on omarchy primitives (MatmulF32/MatmulBF16,
elementwise max/mul, axis-sum reduce); no Metal anywhere.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

_HEADS = 32
_DIM = 128


def _check(q, k, w):
    if q.ndim != 4 or k.ndim != 4 or w.ndim != 2:
        raise ValueError(
            f"[dsa_decode_scores] bad ranks: q={q.ndim} k={k.ndim} w={w.ndim}"
        )
    B = q.shape[0]
    if q.shape[1:] != (_HEADS, 1, _DIM):
        raise ValueError(
            f"[dsa_decode_scores] q must be [B,{_HEADS},1,{_DIM}], got {q.shape}"
        )
    if k.shape[0] != B or k.shape[1] != 1 or k.shape[3] != _DIM:
        raise ValueError(
            f"[dsa_decode_scores] k must be [B,1,S,{_DIM}], got {k.shape}"
        )
    # (The C++ primitive additionally requires a contiguous last dim with
    # 16B-aligned rows for its vectorized loads; the composed omarchy
    # path has no such constraint - the omarchy matmul handles arbitrary
    # layouts.)
    if k.shape[2] < 1024:
        raise ValueError(
            f"[dsa_decode_scores] S must be >= 1024, got {k.shape[2]}"
        )
    if w.shape != (B, _HEADS):
        raise ValueError(
            f"[dsa_decode_scores] w must be [B,{_HEADS}], got {w.shape}"
        )
    if q.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError(
            f"[dsa_decode_scores] q dtype must be fp16/bf16, got {q.dtype}"
        )
    if k.dtype != q.dtype or w.dtype != q.dtype:
        raise ValueError("[dsa_decode_scores] dtype mismatch across inputs")


def dsa_decode_scores_omarchy(
    q: mx.array,
    k: mx.array,
    w: mx.array,
    *,
    fp32_scores: bool = False,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """fp32-chain decode indexer scan on the omarchy backend."""
    _check(q, k, w)
    S = k.shape[2]
    # fp32 chain: f32 matmul (MatmulF32Coopmat on omarchy) accumulates
    # and stores in fp32; relu, weight mul, and the head sum run in
    # fp32 storage - the same accumulation precision as the metal
    # kernel, with the only new rounding at the final store.
    q32 = q.astype(mx.float32)
    k32 = k.astype(mx.float32)
    w32 = w.astype(mx.float32)
    # [B, 32, 1, 128] @ [B, 1, 128, S] -> [B, 32, 1, S]
    acc = mx.matmul(q32, mx.transpose(k32, (0, 1, 3, 2)))
    acc = mx.maximum(acc, 0.0)
    acc = acc * w32[:, :, None, None]
    out32 = mx.sum(acc, axis=1, keepdims=True)  # [B, 1, 1, S]
    out_dtype = mx.float32 if fp32_scores else q.dtype
    return out32.astype(out_dtype)


def dsa_decode_scores_composed(
    q: mx.array,
    k: mx.array,
    w: mx.array,
    *,
    fp32_scores: bool = False,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """The deployed fallback chain: input-dtype intermediates."""
    _check(q, k, w)
    # [B, 32, 1, 128] @ [B, 1, 128, S] -> [B, 32, 1, S] in q.dtype
    acc = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
    acc = mx.maximum(acc, 0.0)
    acc = acc * w[:, :, None, None]
    out = mx.sum(acc, axis=1, keepdims=True)
    out_dtype = mx.float32 if fp32_scores else q.dtype
    return out.astype(out_dtype)


def dsa_decode_scores_reference(
    q: mx.array,
    k: mx.array,
    w: mx.array,
    fp32_scores: bool = False,
) -> mx.array:
    """fp32 ground truth (GPU has no fp64; CPU-stream fp32 eager with
    a per-head accumulation order different from both ports)."""
    def _ref():
        q32 = q.astype(mx.float32)
        k32 = k.astype(mx.float32)
        w32 = w.astype(mx.float32)
        # Per-head loop: different reduction order than the batched
        # matmul + sum the ports use.
        B, H = w32.shape
        acc = None
        for h in range(H):
            qh = q32[:, h, :, :]  # [B, 1, 128]
            kh = k32[:, 0, :, :]  # [B, S, 128]
            s = mx.matmul(qh, mx.transpose(kh, (0, 2, 1)))  # [B, 1, S]
            s = mx.maximum(s, 0.0) * w32[:, h][:, None, None]
            acc = s if acc is None else acc + s
        acc = mx.expand_dims(acc, axis=1)  # [B, 1, 1, S]
        out_dtype = mx.float32 if fp32_scores else q.dtype
        return acc.astype(out_dtype)

    with mx.stream(mx.cpu):
        return _ref()
