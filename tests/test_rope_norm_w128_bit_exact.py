# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Element-level bit-exactness of the fused rope+norm kernel vs the
composed rms_norm -> rope chain, at the shapes the W128 builds serve.

M2D-3 adds 128-thread builds of shaders/fast_rope_norm.comp selected for
fused rows with D <= 128 (Qwen3-4B attention: B=1, T=1, 32 q heads and
8 k heads, D=128). The 128-thread reduction must produce the identical
f32 row sum the 256-thread build produces (the dropped threads load 0.0
and square to +0.0, an exact identity in f32), so the fused output must
stay bit-identical to the composed chain, per element, bf16 included.

Needs the omarchy wheel that ships mx.fast.rope_rms_norm, on hardware
(Apple GPU) or an llvmpipe dev box via MLX_OMARCHY_ALLOW_NON_APPLE=1.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mx")
fast = getattr(mx, "fast", None)
if fast is None or not hasattr(fast, "rope_rms_norm"):
    pytest.skip("wheel has no mx.fast.rope_rms_norm")
rope_rms_norm = fast.rope_rms_norm
rms_norm = fast.rms_norm
rope = fast.rope

D_CASES = [128, 256]  # 128 -> W128 builds; 256 -> the 256-thread builds


def _composed(a, dims, weight, eps, base, scale, offset):
    normed = rms_norm(a, weight, eps)
    return rope(
        normed,
        offset,
        traditional=False,
        base=base,
        scale=scale,
        dims=dims,
    )


@pytest.mark.parametrize("D", D_CASES)
@pytest.mark.parametrize("heads", [8, 32])
@pytest.mark.parametrize("passthrough", [True, False])
def test_w128_fused_bit_equal_composed(D, heads, passthrough):
    eps = 1e-6
    base = 10000.0
    scale = 1.0
    offset = 7
    full = D if passthrough else D // 2  # dims_ = half_dims semantics
    a = (np.random.default_rng(D * heads + passthrough)
         .standard_normal((1, heads, 1, D))
         .astype(np.float32))
    a = (a + np.abs(a).min() * 0.25).astype(np.float32)
    weight = (np.random.default_rng(D).standard_normal(D).astype(np.float32)
              * 0.1 + 1.0)
    a_bf = mx.array(a).astype(mx.bfloat16)
    w_bf = mx.array(weight).astype(mx.bfloat16)

    fused = rope_rms_norm(
        a_bf, full, w_bf, eps,
        traditional=False, base=base, scale=scale, offset=offset,
    )
    composed = _composed(a_bf, full, w_bf, eps, base, scale, offset)

    fused_bits = np.frombuffer(
        np.asarray(fused).astype(np.float32).tobytes(), dtype=np.uint16
    ) >> 16
    composed_bits = np.frombuffer(
        np.asarray(composed).astype(np.float32).tobytes(), dtype=np.uint16
    ) >> 16
    fa = np.asarray(fused).astype(np.float32)
    ca = np.asarray(composed).astype(np.float32)
    assert fa.shape == ca.shape
    assert (fused_bits == composed_bits).all(), (
        f"D={D} heads={heads} passthrough={passthrough}: "
        f"{int((fused_bits != composed_bits).sum())} of {fused_bits.size} "
        "bf16 outputs differ between the fused kernel and the composed chain"
    )
    assert np.isfinite(fa).all()
