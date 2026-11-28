#!/usr/bin/env python3
"""Parity tests for the aten.im2col / aten.col2im lowerings in mlx_vulkan.

Runs under plain python3 (no pytest needed) or pytest.
Requires torch + mlx on the path; run from this directory or set
PYTHONPATH to it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
import mlx_vulkan


def _np(a):
    return np.array(a)


def _check_unfold(a, kernel_size, dilation, padding, stride, rtol=1e-5, atol=1e-6):
    want = F.unfold(torch.tensor(a), kernel_size, dilation=dilation,
                    padding=padding, stride=stride)
    got = np.array(mlx_vulkan._im2col(mx.array(a), kernel_size,
                                      dilation=dilation, padding=padding,
                                      stride=stride))
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.allclose(got, want.numpy(), rtol=rtol, atol=atol), \
        (kernel_size, dilation, padding, stride,
         float(np.abs(got - want.numpy()).max()))


def _check_fold(cols, output_size, kernel_size, dilation, padding, stride,
                rtol=1e-5, atol=1e-6):
    want = F.fold(torch.tensor(cols), output_size, kernel_size,
                  dilation=dilation, padding=padding, stride=stride)
    got = np.array(mlx_vulkan._col2im(mx.array(cols), output_size, kernel_size,
                                      dilation=dilation, padding=padding,
                                      stride=stride))
    assert got.shape == want.shape, (got.shape, want.shape)
    assert np.allclose(got, want.numpy(), rtol=rtol, atol=atol), \
        (output_size, kernel_size, dilation, padding, stride,
         float(np.abs(got - want.numpy()).max()))


def test_im2col_fp32_configs():
    rng = np.random.RandomState(0)
    a = rng.randn(2, 3, 9, 11).astype(np.float32)
    for k, d, p, s in (
        ((3, 3), 1, 0, 1), ((3, 3), 1, 1, 1), ((2, 3), 1, 1, 2),
        ((1, 1), 1, 0, 1), ((3, 3), 2, 2, 1), ((4, 2), 1, (1, 2), (2, 1)),
        ((2, 2), (2, 1), (0, 1), (1, 2)),
    ):
        _check_unfold(a, k, d, p, s)
    print("test_im2col_fp32_configs PASS")


def test_im2col_unbatched():
    # aten.im2col accepts a 3-D (C,H,W) input and returns (C*kh*kw, L).
    rng = np.random.RandomState(1)
    a = rng.randn(4, 10, 12).astype(np.float32)
    for k, d, p, s in (((3, 3), 1, 0, 1), ((2, 4), 2, 1, 2)):
        _check_unfold(a, k, d, p, s)
    print("test_im2col_unbatched PASS")


def test_col2im_fp32_configs():
    rng = np.random.RandomState(2)
    # Independent random column matrices (not unfold output) so overlap sums
    # are exercised, not just roundtrips.
    for b, c, h, w, k, d, p, s in (
        (2, 3, 6, 7, (3, 3), 1, 0, 1), (2, 3, 6, 7, (3, 3), 1, 1, 1),
        (1, 2, 8, 9, (2, 3), 1, 1, 2), (2, 1, 9, 9, (4, 4), 2, 2, 2),
        (2, 2, 7, 8, (2, 2), (2, 1), (1, 0), (2, 1)),
    ):
        hp, wp = h + 2 * (p if isinstance(p, int) else p[0]), w + 2 * (p if isinstance(p, int) else p[1])
        dh, dw = (d, d) if isinstance(d, int) else d
        sh, sw = (s, s) if isinstance(s, int) else s
        kh, kw = k
        ho = (hp - dh * (kh - 1) - 1) // sh + 1
        wo = (wp - dw * (kw - 1) - 1) // sw + 1
        cols = rng.randn(b, c * kh * kw, ho * wo).astype(np.float32)
        _check_fold(cols, (h, w), k, d, p, s)
    print("test_col2im_fp32_configs PASS")


def test_col2im_unbatched():
    rng = np.random.RandomState(3)
    cols = rng.randn(2 * 3 * 3, 25).astype(np.float32)  # C=2, k=3x3, L=25
    _check_fold(cols, (7, 7), (3, 3), 1, 0, 1)
    print("test_col2im_unbatched PASS")


def test_im2col_col2im_roundtrip():
    rng = np.random.RandomState(4)
    a = rng.randn(2, 3, 8, 8).astype(np.float32)
    cols = np.array(mlx_vulkan._im2col(mx.array(a), (3, 3), padding=1))
    back = np.array(mlx_vulkan._col2im(mx.array(cols), (8, 8), (3, 3), padding=1))
    want = F.fold(F.unfold(torch.tensor(a), (3, 3), padding=1), (8, 8), (3, 3),
                  padding=1).numpy()
    assert np.allclose(back, want, rtol=1e-5, atol=1e-6)
    # stride-1 3x3 unfold+fold scales interior by 9 (patch overlap count)
    assert np.allclose(back[:, :, 1:-1, 1:-1], a[:, :, 1:-1, 1:-1] * 9.0,
                       rtol=1e-5, atol=1e-6)
    print("test_im2col_col2im_roundtrip PASS")


def test_im2col_executor():
    # End-to-end: aten.im2col + aten.col2im through the executor's mlx leg.
    rng = np.random.RandomState(5)
    a = rng.randn(1, 2, 7, 9).astype(np.float32)
    g = torch.fx.Graph()
    px = g.placeholder("x")
    n1 = g.call_function(torch.ops.aten.im2col.default,
                         (px, [3, 3], [1, 1], [1, 1], [1, 1]))
    n2 = g.call_function(torch.ops.aten.col2im.default,
                         (n1, [7, 9], [3, 3], [1, 1], [1, 1], [1, 1]))
    g.output(n2)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(torch.tensor(a))
    want = torch.ops.aten.col2im.default(
        torch.ops.aten.im2col.default(torch.tensor(a), [3, 3], [1, 1],
                                      [1, 1], [1, 1]),
        [7, 9], [3, 3], [1, 1], [1, 1], [1, 1])
    np.testing.assert_allclose(np.asarray(out), want.numpy(),
                               rtol=1e-5, atol=1e-6)
    assert "im2col" in stats["mlx"], stats
    assert "col2im" in stats["mlx"], stats
    assert "im2col" not in stats["cpu"], stats
    assert "col2im" not in stats["cpu"], stats
    print("test_im2col_executor PASS")


if __name__ == "__main__":
    test_im2col_fp32_configs()
    test_im2col_unbatched()
    test_col2im_fp32_configs()
    test_col2im_unbatched()
    test_im2col_col2im_roundtrip()
    test_im2col_executor()
