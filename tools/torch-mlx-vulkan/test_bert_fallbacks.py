#!/usr/bin/env python3
"""Parity tests for the BERT-train fallback lowerings in mlx_vulkan.

Covers select_backward + tanh_backward (added here) and the three ops the
BERT backward graph needs that already exist on main: bitwise_and,
new_ones, scalar_tensor. Each test compares the lowering against eager
torch. Runs under plain python3 or pytest.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _np(a):
    return np.array(a)


def test_select_backward():
    rng = np.random.RandomState(0)
    for sizes, dim, index in (((3, 4, 5), 1, 2), ((3, 4, 5), 2, 0),
                              ((3, 4, 5), -1, 4), ((6,), 0, 0)):
        dd = dim % len(sizes)
        gg = np.asarray(rng.randn(*[s for i, s in enumerate(sizes) if i != dd]),
                        dtype=np.float32)
        got = mlx_vulkan._select_backward(mx.array(gg), sizes, dim, index)
        x = torch.zeros(*sizes, requires_grad=True)
        x.clone().select(dd, index).backward(torch.tensor(gg))
        want = x.grad.numpy()
        assert got.shape == want.shape, (sizes, dim, index, got.shape)
        assert np.allclose(_np(got), want), (sizes, dim, index)
    print("test_select_backward PASS")


def test_tanh_backward():
    rng = np.random.RandomState(1)
    x = rng.randn(4, 8).astype(np.float32)
    g = rng.randn(4, 8).astype(np.float32)
    out = np.tanh(x)
    spec = mlx_vulkan._FUNC_SPECS["tanh_backward"]
    got = spec(mx.array(g), mx.array(out))
    xt = torch.tensor(x, requires_grad=True)
    torch.tanh(xt).backward(torch.tensor(g))
    assert np.allclose(_np(got), xt.grad.numpy(), atol=1e-7)
    print("test_tanh_backward PASS")


def test_bitwise_and():
    a = np.array([[0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.int64)
    b = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.int64)
    spec = mlx_vulkan._FUNC_SPECS["bitwise_and"]
    got = spec(mx.array(a), mx.array(b))
    assert np.array_equal(_np(got), np.bitwise_and(a, b))
    ab, bb = a.astype(bool), b.astype(bool)
    got = spec(mx.array(ab), mx.array(bb))
    assert np.array_equal(_np(got), np.logical_and(ab, bb))
    print("test_bitwise_and PASS")


def test_new_ones():
    x = torch.zeros(2, dtype=torch.float32)
    spec = mlx_vulkan._FUNC_SPECS["new_ones"]
    got = spec(mx.zeros(2), (2, 3))
    assert np.array_equal(_np(got), np.ones((2, 3), dtype=np.float32))
    got = spec(mx.zeros(2, dtype=mx.int64), (3,), dtype=None)
    assert np.array_equal(_np(got), np.ones((3,), dtype=np.int64))
    print("test_new_ones PASS")


def test_scalar_tensor():
    spec = mlx_vulkan._FUNC_SPECS["scalar_tensor"]
    got = spec(5)
    assert np.array_equal(_np(got), np.array(5))
    got = spec(2.5)
    assert np.allclose(_np(got), np.array(2.5))
    got = spec(3, dtype=torch.float32)
    assert np.allclose(_np(got), np.array(3.0, dtype=np.float32))
    print("test_scalar_tensor PASS")


def test_dispatch_registered():
    for name in ("select_backward", "tanh_backward", "bitwise_and",
                 "new_ones", "scalar_tensor"):
        assert name in mlx_vulkan.ATEN_TABLE, name
        assert name in mlx_vulkan._FUNC_SPECS, name
    print("test_dispatch_registered PASS")


if __name__ == "__main__":
    test_select_backward()
    test_tanh_backward()
    test_bitwise_and()
    test_new_ones()
    test_scalar_tensor()
    test_dispatch_registered()
    print("ALL PASS")
