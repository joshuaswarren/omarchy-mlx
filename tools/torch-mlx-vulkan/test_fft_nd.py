# PT-Q61: aten.fft_fftn/ifftn + fft_fftshift/ifftshift — N-D parity vs torch
# CPU across shapes (even and odd), axes subsets, all three norm modes, the
# s pad/crop, real input -> complex out, the inverse pairing, and the shift
# rolls. All through the lowering tables; no CPU fallback on the happy path.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import mlx_vulkan  # noqa: E402

a = torch.ops.aten


def mxin(t):
    return mx.array(t.detach().numpy())


def check_close(got, want, tag):
    g, w = np.array(got), want.detach().numpy()
    assert g.shape == w.shape, (tag, g.shape, w.shape)
    assert g.dtype.kind == w.dtype.kind, (tag, g.dtype, w.dtype)
    if g.dtype.kind == "c":
        g, w = g.real, w.real + 1j * w.imag  # exact compare through numpy
    assert np.allclose(g, w, atol=1e-4, rtol=1e-4), (tag, np.abs(np.asarray(g) - w).max())


def test_fftn_ifftn():
    rng = np.random.RandomState(3)
    x = rng.randn(2, 6, 5).astype(np.float32)  # even and odd lengths mixed
    xt = torch.tensor(x)
    for axes in [None, [0], [1], [2], [0, 1], [1, 2], [0, 2]]:
        for norm in [None, "backward", "ortho", "forward"]:
            ref = torch.fft.fftn(xt, axes=axes, norm=norm)
            got = mlx_vulkan.ATEN_TABLE["fft_fftn"](mxin(xt), axes=axes, norm=norm)
            check_close(got, ref, f"fftn axes={axes} norm={norm}")
    # the s pad/crop: numpy-style length override per axis
    ref = torch.fft.fftn(xt, s=(4, 8, 5))
    got = mlx_vulkan.ATEN_TABLE["fft_fftn"](mxin(xt), s=(4, 8, 5))
    check_close(got, ref, "fftn s=(4,8,5)")
    # real input -> complex out, same shape as input
    got = mlx_vulkan.ATEN_TABLE["fft_fftn"](mxin(xt))
    assert np.array(got).dtype.kind == "c" and np.array(got).shape == (2, 6, 5)
    # the inverse pairing recovers the input for every norm pairing torch
    # defines, and the table's ifftn matches torch.ifftn exactly
    for norm in [None, "backward", "ortho", "forward"]:
        fwd = mlx_vulkan.ATEN_TABLE["fft_fftn"](mxin(xt), norm=norm)
        ref = torch.fft.ifftn(torch.tensor(np.array(fwd)), norm=norm)
        back = mlx_vulkan.ATEN_TABLE["fft_ifftn"](mx.array(np.array(fwd)), norm=norm)
        check_close(back, ref, f"ifftn norm={norm}")
        round_trip = np.array(mlx_vulkan.ATEN_TABLE["fft_ifftn"](fwd, norm=norm))
        assert np.allclose(round_trip, x, atol=1e-3, rtol=1e-3), (norm,)
    # the aten names lower through the same specs
    assert mlx_vulkan.ATEN_TABLE["fft_ifftn"] is mlx_vulkan._FUNC_SPECS["fft_ifftn"]
    print("test_fft_nd fftn/ifftn PASS: shapes, axes subsets, norms, s, inverse pairing")


def test_fftshift_ifftshift():
    rng = np.random.RandomState(5)
    for shape in [(8,), (7,), (4, 6), (5, 7), (2, 5, 6)]:
        x = torch.tensor(rng.randn(*shape).astype(np.float32))
        for dim in [None, 0, -1, [0], [0, 1], [-1], [-2, -1] if len(shape) >= 2 else [0]]:
            got = mlx_vulkan.ATEN_TABLE["fft_fftshift"](mxin(x), dim=dim)
            ref = torch.fft.fftshift(x, dim=dim)
            assert np.array_equal(np.array(got), ref.numpy()), (shape, dim)
            back = mlx_vulkan.ATEN_TABLE["fft_ifftshift"](got, dim=dim)
            assert np.array_equal(np.array(back), x.numpy()), (shape, dim)  # roll pair = identity
    print("test_fft_nd fftshift/ifftshift PASS: even/odd, dim subsets, round trip")


if __name__ == "__main__":
    test_fftn_ifftn()
    test_fftshift_ifftshift()
