"""BitNet ternary matmul must run on non-Metal builds.

Upstream mlx_lm calls `mx.fast.metal_kernel` unconditionally in
`BitLinear.execute_matmul_kernel` (ml-explore/mlx-lm #219), so a CPU-only
or non-Apple build raises `[metal_kernel] No Metal back-end.` and BitNet
cannot run at all. `patches/mlx-lm-bitlinear-cpu-fallback.patch` adds the
upstream-style gate: Metal keeps the kernel, anything else takes a
pure-MLX unpack + matmul fallback.

The check here is the gate's contract: on a build where
`mx.metal.is_available()` is false the call must produce the same values
as an independent NumPy unpacking of the packed ternary weights, in both
`invert_weight_scales` modes, without ever touching the Metal kernel
object. On unpatched upstream this test raises the Metal back-end error
— that failure is the point.

Skips where mlx.core or mlx_lm is not installed.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")


def _pack_ternary(w_np, in_features):
    """Pack ternary weights the way `BitLinear.weight` stores them.

    Byte (r, c) holds, in 2-bit fields j=0..3, the weights of outputs
    ``r + j * (out_features // 4)`` at input c — the layout the Metal
    kernel's ``(w >> (2*j)) & 3) - 1`` unpacking implies.
    """
    out_quarter = w_np.shape[0] // 4
    packed = np.zeros((out_quarter, in_features), dtype=np.uint8)
    for r in range(out_quarter):
        for j in range(4):
            packed[r] |= ((w_np[r + j * out_quarter] + 1).astype(np.uint8) & 3) << (2 * j)
    return packed


def _np_reference(x_np, w_np, scale, invert):
    s = (1.0 / scale) if invert else scale
    return x_np.astype(np.float64) @ w_np.astype(np.float64).T * s


@pytest.mark.skipif(
    mx.metal.is_available(),
    reason="fallback path is only exercised on non-Metal builds",
)
def test_bitlinear_matmul_matches_numpy_reference_without_metal():
    from mlx_lm.models.bitlinear_layers import BitLinear

    rng = np.random.default_rng(0)
    batch, in_features, out_features = 2, 8, 4
    x_np = rng.standard_normal((batch, in_features)).astype(np.float32)
    w_np = rng.integers(-1, 2, size=(out_features, in_features), dtype=np.int8)
    packed = mx.array(_pack_ternary(w_np, in_features))
    x = mx.array(x_np)

    import mlx_lm.models.bitlinear_layers as mod

    class MetalSpy:
        def __call__(self, *args, **kwargs):
            raise AssertionError("Metal kernel invoked on a non-Metal build")

    original = mod._bitlinear_kernel
    mod._bitlinear_kernel = MetalSpy()
    try:
        for invert, scale_val in ((False, 1.5), (True, 2.0)):
            layer = BitLinear(in_features, out_features, bias=False, invert_weight_scales=invert)
            layer.weight = packed
            layer.weight_scale = mx.array([scale_val], dtype=mx.float32)
            out = layer.execute_matmul_kernel(x, packed)
            mx.eval(out)
            got = np.asarray(out).astype(np.float64)
            want = _np_reference(x_np, w_np, scale_val, invert)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)

        # The >2-D reshape path must reach the fallback too.
        layer = BitLinear(in_features, out_features, bias=False, invert_weight_scales=False)
        layer.weight = packed
        layer.weight_scale = mx.array([1.5], dtype=mx.float32)
        out3 = layer.execute_matmul_kernel(x.reshape(1, batch, in_features), packed)
        mx.eval(out3)
        assert tuple(np.asarray(out3).shape) == (1, batch, out_features)
    finally:
        mod._bitlinear_kernel = original
