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
                              ((3, 4, 5), -1, 4), ((6,), 0, 0),
                              ((3, 4, 5), 1, -1), ((3, 4, 5), 0, -3)):
        dd = dim % len(sizes)
        gg = np.asarray(rng.randn(*[s for i, s in enumerate(sizes) if i != dd]),
                        dtype=np.float32)
        got = mlx_vulkan._select_backward(mx.array(gg), sizes, dim, index)
        x = torch.zeros(*sizes, requires_grad=True)
        x.clone().select(dd, index).backward(torch.tensor(gg))
        want = x.grad.numpy()
        assert got.shape == want.shape, (sizes, dim, index, got.shape)
        assert np.allclose(_np(got), want), (sizes, dim, index)


def test_select_backward_nan_inf():
    # A NaN/Inf grad must land only on the selected slice; unselected
    # positions are exactly 0 (g * 0 would leak NaN/Inf, mx.where does not).
    sizes, dim, index = (2, 3), 1, 0
    gg = np.array([np.nan, np.inf], dtype=np.float32)
    got = _np(mlx_vulkan._select_backward(mx.array(gg), sizes, dim, index))
    x = torch.zeros(*sizes, requires_grad=True)
    x.clone().select(dim, index).backward(torch.tensor(gg))
    want = x.grad.numpy()
    assert np.array_equal(np.isnan(got), np.isnan(want)), (got, want)
    assert np.array_equal(np.isinf(got), np.isinf(want)), (got, want)
    assert np.all(got[:, 1:] == 0.0), got
    print("test_select_backward_nan_inf PASS")


def test_tanh_backward():
    rng = np.random.RandomState(1)
    # fp32 + bf16 through the dispatch table: aten.tanh_backward.default
    # runs grad*(1-out^2) in fp32 opmath over the saved output.
    x = rng.randn(4, 8).astype(np.float32)
    g = rng.randn(4, 8).astype(np.float32)
    out = np.tanh(x)
    for tag in ("fp32", "bf16"):
        if tag == "bf16":
            ob = torch.tensor(out, dtype=torch.bfloat16)
            gref = torch.tensor(g, dtype=torch.bfloat16)
            ref = torch.ops.aten.tanh_backward.default(gref, ob)
            mine = mlx_vulkan.ATEN_TABLE["tanh_backward"](
                mx.array(gref.float().numpy()).astype(mx.bfloat16),
                mx.array(ob.float().numpy()).astype(mx.bfloat16))
            assert mine.dtype == mx.bfloat16, mine.dtype
            got = torch.from_numpy(np.array(mine.astype(mx.float32)))
            d = float((got - ref.float()).abs().max())
            assert d < 0.02, f"bf16: {d:.3e}"
        else:
            gref = torch.tensor(g)
            oref = torch.tensor(out)
            ref = torch.ops.aten.tanh_backward.default(gref, oref)
            mine = mlx_vulkan.ATEN_TABLE["tanh_backward"](
                mx.array(g), mx.array(out))
            assert mine.dtype == mx.float32
            assert np.allclose(_np(mine), ref.numpy(), atol=1e-6)
    # executor path: the aten.tanh_backward.default node lowers to mx.
    gfx = torch.fx.Graph()
    ph_g, ph_o = gfx.placeholder("g"), gfx.placeholder("o")
    gfx.output(gfx.call_function(torch.ops.aten.tanh_backward.default, (ph_g, ph_o)))
    gm = torch.fx.GraphModule(torch.nn.Module(), gfx)
    run, stats = mlx_vulkan.make_executor(gm)
    res = run(torch.tensor(g), torch.tensor(out))
    ref = torch.ops.aten.tanh_backward.default(torch.tensor(g), torch.tensor(out))
    assert np.abs(_np(res) - ref.numpy()).max() < 1e-6
    assert "tanh_backward" in stats["mlx"], stats
    assert "tanh_backward" not in stats["cpu"], stats
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
    test_select_backward_nan_inf()
    test_tanh_backward()
    test_bitwise_and()
    test_new_ones()
    test_scalar_tensor()
    test_dispatch_registered()
    print("ALL PASS")
