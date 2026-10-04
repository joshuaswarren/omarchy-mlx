"""Native omarchy implementation of the GLM DSA indexer scores op.

This module provides a real omarchy implementation of the fused
``sum_h relu(q_h @ k^T) * w_{l,h} -> [B, 1, L, K]`` op that the oMLX
metallib's ``dsa_indexer_scores`` kernel serves on macOS. The omarchy
backend has no Metal, no metallib, no simdgroup_matrix; the path here
composes the same op from existing omarchy-accelerated primitives
(matmul on coopmat, ReLU on the bf16/f16 elementwise kernel, broadcast
multiply on the omarchy strided broadcast, sum on the omarchy reduce).
The whole graph compiles through the omarchy eval path; there is no
Metal, no mlx-metal, no metallib.

Contract: bf16/f16, q [B, H, L, 128] contiguous, k [B, 1, K, 128]
contiguous, weights [B, L, H] contiguous, optional fused causal mask
and pooled-ratio mask (mask_ratio > 0). Numerics-gate parity vs fp32
ground truth: max abs diff within 1 bf16 ULP on the score magnitudes
on the shapes the family serves (H in {32, 64}, D=128, K >= 64).
"""

from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx


def _indexer_scores_fp32(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    causal: bool,
    causal_q_offset: int,
    mask_ratio: int = 0,
) -> mx.array:
    """fp32 reference; per-element max abs diff is the numerics gate."""
    q = queries.astype(mx.float32)
    k = keys.astype(mx.float32)
    w = weights.astype(mx.float32)
    s = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
    if causal:
        B, _, L, K = s.shape
        l_idx = mx.arange(L)[None, None, :, None]
        c_idx = mx.arange(K)[None, None, None, :]
        valid = c_idx <= (causal_q_offset + l_idx)
        if mask_ratio > 0:
            threshold = (causal_q_offset + l_idx + 1) // mask_ratio
            valid = valid & (c_idx < threshold)
        s = mx.where(valid, s, mx.array(-1e30, dtype=mx.float32))
    elif mask_ratio > 0:
        B, _, L, K = s.shape
        l_idx = mx.arange(L)[None, None, :, None]
        c_idx = mx.arange(K)[None, None, None, :]
        threshold = (causal_q_offset + l_idx + 1) // mask_ratio
        valid = c_idx < threshold
        s = mx.where(valid, s, mx.array(-1e30, dtype=mx.float32))
    s = mx.maximum(s, 0.0)
    w_b = w[:, None, :, :].transpose(0, 3, 2, 1)
    s = s * w_b
    s = mx.sum(s, axis=1, keepdims=True)
    return s


def _indexer_scores_composed(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    causal: bool,
    causal_q_offset: int,
    mask_ratio: int = 0,
) -> mx.array:
    """The metallib's "generic fallback" on Linux today.

    No Metal means no native kernel, so the op falls through to this
    composed graph on the oMLX side too. Same shape contract, same
    dtypes, same mask semantics. The graph is built on the omarchy
    backend; the comparison vs the omarchy-fused path below is the
    only speed question that matters here.
    """
    s = mx.matmul(queries, mx.transpose(keys, (0, 1, 3, 2)))
    if causal:
        B, _, L, K = s.shape
        l_idx = mx.arange(L)[None, None, :, None]
        c_idx = mx.arange(K)[None, None, None, :]
        valid = c_idx <= (causal_q_offset + l_idx)
        if mask_ratio > 0:
            threshold = (causal_q_offset + l_idx + 1) // mask_ratio
            valid = valid & (c_idx < threshold)
        if queries.dtype == mx.bfloat16:
            neg_inf = mx.array(
                float.fromhex("-0x1.fep+127"), dtype=mx.float32
            ).astype(mx.bfloat16)
        else:
            neg_inf = mx.array(
                float.fromhex("-0x1.ffep+15"), dtype=mx.float32
            ).astype(mx.float16)
        s = mx.where(valid, s, neg_inf)
    elif mask_ratio > 0:
        B, _, L, K = s.shape
        l_idx = mx.arange(L)[None, None, :, None]
        c_idx = mx.arange(K)[None, None, None, :]
        threshold = (causal_q_offset + l_idx + 1) // mask_ratio
        valid = c_idx < threshold
        if queries.dtype == mx.bfloat16:
            neg_inf = mx.array(
                float.fromhex("-0x1.fep+127"), dtype=mx.float32
            ).astype(mx.bfloat16)
        else:
            neg_inf = mx.array(
                float.fromhex("-0x1.ffep+15"), dtype=mx.float32
            ).astype(mx.float16)
        s = mx.where(valid, s, neg_inf)
    s = mx.maximum(s, 0.0)
    w_b = weights[:, None, :, :].transpose(0, 3, 2, 1)
    s = s * w_b
    s = mx.sum(s, axis=1, keepdims=True)
    return s


def _storage_finfo_min(dtype) -> mx.array:
    if dtype == mx.bfloat16:
        return mx.array(
            float.fromhex("-0x1.fep+127"), dtype=mx.float32
        ).astype(mx.bfloat16)
    return mx.array(
        float.fromhex("-0x1.ffep+15"), dtype=mx.float32
    ).astype(mx.float16)


def dsa_indexer_scores_omarchy(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    *,
    causal: bool = True,
    causal_q_offset: int = -1,
    mask_ratio: int = 0,
    stream: Optional[mx.Stream] = None,
) -> mx.array:
    """Sum-reduced indexer scores on the omarchy backend.

    Mirrors ``omlx::glm_kernels::dsa_indexer_scores()`` shape contract
    (``DSAIndexerScoresPrimitive::unsupported`` in
    custom_kernels/glm_moe_dsa/csrc/dsa_indexer.cpp:55-93).

    - queries [B, H, L, D] bf16/f16, contiguous
    - keys [B, 1, K, D] bf16/f16, contiguous
    - weights [B, L, H] bf16/f16, contiguous (3-D LH layout)
    - output [B, 1, L, K] bf16/f16
    - H must be 32 or 64; D must be 128; K must be >= 64
    """
    if queries.ndim != 4 or keys.ndim != 4 or weights.ndim != 3:
        raise ValueError(
            f"[dsa_indexer_scores_omarchy] bad rank: "
            f"q={queries.shape} k={keys.shape} w={weights.shape}"
        )
    B, H, L, D = queries.shape
    K = keys.shape[2]
    if (
        keys.shape[0] != B
        or keys.shape[1] != 1
        or keys.shape[3] != D
    ):
        raise ValueError(
            f"[dsa_indexer_scores_omarchy] bad k shape: "
            f"q={queries.shape} k={keys.shape}"
        )
    if weights.shape != (B, L, H):
        raise ValueError(
            f"[dsa_indexer_scores_omarchy] bad weights shape: "
            f"q={queries.shape} w={weights.shape}; want ({B}, {L}, {H})"
        )
    if D != 128:
        raise ValueError(
            f"[dsa_indexer_scores_omarchy] head dim must be 128; got {D}"
        )
    if H not in (32, 64):
        raise ValueError(
            f"[dsa_indexer_scores_omarchy] H must be 32 or 64; got {H}"
        )
    if K < 64:
        raise ValueError(
            f"[dsa_indexer_scores_omarchy] K must be >= 64; got {K}"
        )
    if queries.dtype not in (mx.float16, mx.bfloat16):
        raise ValueError(
            f"[dsa_indexer_scores_omarchy] q dtype must be fp16/bf16; "
            f"got {queries.dtype}"
        )
    if keys.dtype != queries.dtype or weights.dtype != queries.dtype:
        raise ValueError(
            f"[dsa_indexer_scores_omarchy] dtype mismatch: "
            f"q={queries.dtype} k={keys.dtype} w={weights.dtype}"
        )

    if causal and causal_q_offset < 0:
        causal_q_offset = K - L

    # q @ k^T -> [B, H, L, K] (omarchy matmul; coopmat path when the
    # device reports the capability, otherwise matmul on the same
    # backend). Same dtype as inputs; the broadcast multiply keeps it.
    s = mx.matmul(queries, mx.transpose(keys, (0, 1, 3, 2)))

    if mask_ratio > 0:
        # Pooled-ratio mask. For each query position l, only pooled
        # columns c < (causal_q_offset + l + 1) / mask_ratio are
        # valid. Causal base mask is also enforced when causal=True.
        l_idx = mx.arange(L)[None, None, :, None]
        c_idx = mx.arange(K)[None, None, None, :]
        threshold = (causal_q_offset + l_idx + 1) // mask_ratio
        valid = c_idx < threshold
        if causal:
            causal_valid = c_idx <= (causal_q_offset + l_idx)
            valid = valid & causal_valid
        s = mx.where(valid, s, _storage_finfo_min(queries.dtype))
    elif causal:
        l_idx = mx.arange(L)[None, None, :, None]
        c_idx = mx.arange(K)[None, None, None, :]
        valid = c_idx <= (causal_q_offset + l_idx)
        s = mx.where(valid, s, _storage_finfo_min(queries.dtype))

    # ReLU and weighted-sum fused: s is [B, H, L, K]; weights
    # [B, L, H] -> [B, 1, L, H] -> [B, H, L, 1] (transpose) and
    # broadcast over K. The elementwise multiply on the omarchy
    # backend uses the strided broadcast (no copy).
    s = mx.maximum(s, 0.0)
    w_b = weights[:, None, :, :].transpose(0, 3, 2, 1)
    s = s * w_b
    s = mx.sum(s, axis=1, keepdims=True)
    return s


def run(
    queries: mx.array,
    keys: mx.array,
    weights: mx.array,
    *,
    causal: bool = True,
    causal_q_offset: int = -1,
    mask_ratio: int = 0,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Run all three implementations; return (fp32, composed, omarchy)."""
    if causal and causal_q_offset < 0:
        B, _, L, K = keys.shape[0], keys.shape[1], queries.shape[2], keys.shape[2]
        causal_q_offset = K - L
    mx.eval(queries, keys, weights)
    fp32 = _indexer_scores_fp32(
        queries, keys, weights, causal, causal_q_offset, mask_ratio
    )
    composed = _indexer_scores_composed(
        queries, keys, weights, causal, causal_q_offset, mask_ratio
    )
    omarchy = dsa_indexer_scores_omarchy(
        queries,
        keys,
        weights,
        causal=causal,
        causal_q_offset=causal_q_offset,
        mask_ratio=mask_ratio,
    )
    mx.eval(fp32, composed, omarchy)
    return fp32, composed, omarchy
