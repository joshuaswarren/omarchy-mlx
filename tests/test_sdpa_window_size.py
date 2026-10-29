# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Sliding-window attention must reach the Omarchy backend with its window.

Upstream MLX (ml-explore/mlx#4734) adds ``window_size`` to
``mx.fast.scaled_dot_product_attention`` and to ``ScaledDotProductAttention::use_fallback``.
Our ``use_fallback`` returns false for every inference call, so the fused
kernels would run on a windowed call as if there were no window: the output is
finite and plausible, and wrong. The backend must either serve the window or
return true so that MLX builds the composed graph (which applies it).

Each case compares the windowed call against the same attention written as an
explicit boolean mask, at shapes that reach the decode kernel (one query row,
head dim 128), the prefill kernels (several query rows) and the sliced-cache
shape (more keys than queries). It also asserts that the window changes the
answer for the inputs used, so a backend that ignores the window cannot pass
by comparing two unwindowed results.

The tests skip on an MLX that does not accept ``window_size`` (the baseline
before the sync that brings in #4734) and on a machine without a GPU device.
"""

import pytest

mx = pytest.importorskip("mlx.core")

try:
    mx.fast.scaled_dot_product_attention(
        mx.zeros((1, 1, 1, 8)), mx.zeros((1, 1, 1, 8)), mx.zeros((1, 1, 1, 8)),
        scale=1.0, window_size=-1)
    HAS_WINDOW = True
except TypeError:
    HAS_WINDOW = False

pytestmark = [
    pytest.mark.skipif(not HAS_WINDOW, reason="MLX baseline without sdpa window_size (ml-explore/mlx#4734)"),
    pytest.mark.skipif(not mx.is_available(mx.gpu), reason="needs the GPU device"),
]


def _window_mask(q_len, k_len, window, causal):
    q_idx = mx.arange(q_len)[:, None] + (k_len - q_len)
    k_idx = mx.arange(k_len)[None, :]
    mask = mx.abs(q_idx - k_idx) < window
    return mask & (q_idx >= k_idx) if causal else mask


def _reference(q, k, v, scale, mask, sinks):
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k = mx.repeat(k, rep, axis=1)
        v = mx.repeat(v, rep, axis=1)
    scores = (q * scale) @ k.swapaxes(-1, -2)
    scores = mx.where(mask, scores, -mx.inf)
    if sinks is not None:
        sink = mx.broadcast_to(sinks[None, :, None, None], scores.shape[:-1] + (1,))
        scores = mx.concatenate([sink, scores], axis=-1)
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    if sinks is not None:
        weights = weights[..., 1:]
    return weights @ v


# (query rows, key rows, head dim, window, causal, sinks)
CASES = [
    (1, 33, 128, 3, True, False),   # decode kernel, window far inside the cache
    (1, 33, 128, 3, False, True),   # decode kernel, no causal mask, with sinks
    (7, 33, 64, 16, True, False),   # prefill, sliced cache (keys > queries)
    (16, 16, 64, 3, True, True),    # prefill, square
    (16, 16, 64, 1, False, False),  # window of one key (the diagonal)
]


@pytest.mark.parametrize("q_len,k_len,dim,window,causal,sinks", CASES)
def test_window_matches_the_boolean_mask(q_len, k_len, dim, window, causal, sinks):
    mx.random.seed(q_len * 1000 + k_len)
    heads, kv_heads = 4, 2
    scale = dim ** -0.5
    q = mx.random.normal((1, heads, q_len, dim)).astype(mx.bfloat16)
    k = mx.random.normal((1, kv_heads, k_len, dim)).astype(mx.bfloat16)
    v = mx.random.normal((1, kv_heads, k_len, dim)).astype(mx.bfloat16)
    s = mx.random.normal((heads,)).astype(mx.bfloat16) if sinks else None

    kwargs = {"scale": scale, "sinks": s}
    if causal:
        kwargs["mask"] = "causal"
    out = mx.fast.scaled_dot_product_attention(q, k, v, window_size=window, **kwargs)
    unwindowed = mx.fast.scaled_dot_product_attention(q, k, v, **kwargs)
    ref = _reference(q, k, v, scale, _window_mask(q_len, k_len, window, causal), s)
    mx.eval(out, unwindowed, ref)

    out32, ref32, un32 = (a.astype(mx.float32) for a in (out, ref, unwindowed))
    assert not mx.allclose(out32, un32, atol=1e-2, rtol=1e-2).item(), \
        "the window did not change the result: the window was dropped before the kernel"
    assert mx.allclose(out32, ref32, atol=2e-2, rtol=2e-2).item(), \
        f"windowed output differs from the masked reference by {mx.max(mx.abs(out32 - ref32)).item():.4g}"


def test_invalid_windows_and_force_fused_raise():
    q = mx.random.normal((1, 2, 4, 32))
    for window in (0, -2):
        with pytest.raises(ValueError):
            mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, window_size=window)
    with pytest.raises(ValueError):
        mx.fast.scaled_dot_product_attention(q, q, q, scale=1.0, window_size=2, force_fused=True)
