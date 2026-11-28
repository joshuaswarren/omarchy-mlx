#!/usr/bin/env python3
# Q-12: aten.cumprod/cummax/cummin scanned reductions — parity vs torch CPU
# (incl. reverse via flip+scan where the aten op supports it).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _np(a):
    return np.array(a)


def test_cumprod():
    rng = np.random.RandomState(3)
    for shape in ((6,), (3, 5)):
        x = rng.rand(*shape).astype(np.float32) + 0.5
        for dim in (0,) if len(shape) == 1 else (0, 1, -1):
            want_dim = dim
            got = mlx_vulkan._FUNC_SPECS["cumprod"](mlx_vulkan._mx(x), dim)
            wt = torch.cumprod(torch.tensor(x), dim=want_dim)
            assert np.allclose(_np(got), wt.numpy(), rtol=1e-4), (shape, dim)
    print("test_cumprod PASS")


def test_cummax_cummin():
    rng = np.random.RandomState(5)
    for shape in ((6,), (3, 5)):
        x = rng.randn(*shape).astype(np.float32)
        dims = (0,) if len(shape) == 1 else (0, 1, -1)
        for dim in dims:
            gv, gi = mlx_vulkan._FUNC_SPECS["cummax"](mlx_vulkan._mx(x), dim)
            wv, wi = torch.cummax(torch.tensor(x), dim)
            assert np.allclose(_np(gv), wv.values.numpy()), (shape, dim)
            assert np.array_equal(_np(gi), wv.indices.numpy()), (shape, dim)
            gv2, gi2 = mlx_vulkan._FUNC_SPECS["cummin"](mlx_vulkan._mx(x), dim)
            wv2, wi2 = torch.cummin(torch.tensor(x), dim)
            assert np.allclose(_np(gv2), wv2.values.numpy()), (shape, dim)
            assert np.array_equal(_np(gi2), wi2.indices.numpy()), (shape, dim)
    # torch ties adopt the latest position; each NaN adopts once then
    # blocks every later finite value - pin both rules on the index side.
    for arr in ([1.0, 3.0, 3.0, 2.0, 5.0], [1.0, np.nan, np.nan, 2.0],
                [np.nan, np.nan, 1.0], [1.0, 3.0, np.nan, 5.0, 4.0]):
        t = torch.tensor(arr)
        gv, gi = mlx_vulkan._FUNC_SPECS["cummax"](mlx_vulkan._mx(np.array(arr)), 0)
        wv, wi = torch.cummax(t, 0)
        assert np.array_equal(np.array(gv, copy=False), wv.values.numpy(),
                              equal_nan=True), arr
        assert np.array_equal(_np(gi), wi.numpy()), arr
    print("test_cummax_cummin PASS")


def test_reverse_via_flip():
    rng = np.random.RandomState(9)
    x = rng.randn(5, 6).astype(np.float32) + 0.5
    # reverse cumulative sum via flip+scan (the flip+scan technique)
    got = mx.flip(mx.cumsum(mx.flip(mlx_vulkan._mx(x)), axis=1), axis=1)
    want = np.flip(np.cumsum(np.flip(x, 1), 1), 1)
    assert np.allclose(_np(got), want, rtol=1e-4)
    print("test_reverse_via_flip PASS")


if __name__ == "__main__":
    test_cumprod()
    test_cummax_cummin()
    test_reverse_via_flip()
    print("ALL PASS")
