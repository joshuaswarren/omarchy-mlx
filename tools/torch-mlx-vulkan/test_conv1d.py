#!/usr/bin/env python3
"""Parity tests for the conv1d lowering in mlx_vulkan.

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


def _check(a, w, b, stride, padding, dilation, dtype, rtol, atol, groups=1):
    if dtype is torch.bfloat16:
        a = torch.tensor(a, dtype=torch.bfloat16)
        w = torch.tensor(w, dtype=torch.bfloat16)
        b = torch.tensor(b, dtype=torch.bfloat16) if b is not None else None
    else:
        a = torch.tensor(a, dtype=dtype)
        w = torch.tensor(w, dtype=dtype)
        b = torch.tensor(b, dtype=dtype) if b is not None else None
    want = F.conv1d(a, w, b, stride=stride, padding=padding, dilation=dilation,
                    groups=groups)
    mxa = mx.array(np.array(a))
    mxw = mx.array(np.array(w))
    mxb = mx.array(np.array(b)) if b is not None else None
    got = mlx_vulkan._conv1d(mxa, mxw, mxb, stride=stride, padding=padding,
                             dilation=dilation, groups=groups)
    got = np.array(got)
    want_np = want.numpy()
    assert got.shape == want_np.shape, (got.shape, want_np.shape)
    if dtype is torch.bfloat16:
        ok = np.allclose(got.astype(np.float32), want_np.astype(np.float32), rtol=rtol, atol=atol)
    else:
        ok = np.allclose(got, want_np, rtol=rtol, atol=atol)
    assert ok, (dtype, stride, padding, dilation, groups,
                float(np.abs(got.astype(np.float32) - want_np.astype(np.float32)).max()))


def test_conv1d_fp32_configs():
    rng = np.random.RandomState(0)
    a = rng.randn(2, 6, 23).astype(np.float32)
    w = rng.randn(5, 6, 4).astype(np.float32)
    b = rng.randn(5).astype(np.float32)
    for stride, padding, dilation, use_b in (
        (1, 0, 1, True), (1, 0, 1, False), (2, 0, 1, True), (1, 2, 1, True),
        (2, 2, 1, False), (1, 3, 2, True), (3, 1, 2, False), (2, 4, 3, True),
    ):
        _check(a, w, b if use_b else None, stride, padding, dilation,
               torch.float32, 1e-4, 1e-5)
    print("test_conv1d_fp32_configs PASS")


def test_conv1d_fp32_shapes():
    rng = np.random.RandomState(1)
    for n, c, l, o, k in ((1, 3, 11, 4, 3), (3, 8, 32, 16, 5), (2, 1, 7, 1, 1)):
        a = rng.randn(n, c, l).astype(np.float32)
        w = rng.randn(o, c, k).astype(np.float32)
        b = rng.randn(o).astype(np.float32)
        _check(a, w, b, 1, 0, 1, torch.float32, 1e-4, 1e-5)
        _check(a, w, b, 2, 1, 1, torch.float32, 1e-4, 1e-5)
    print("test_conv1d_fp32_shapes PASS")


def test_conv1d_unbatched():
    # PAR-41 census rerun: aten.conv1d's unbatched 2-D (C, L) input crashed the
    # lowering - x.transpose(0, 2, 1) asked 3 axes of a 2-D array. Torch treats
    # a missing batch dim as N=1 and squeezes the output back.
    rng = np.random.RandomState(7)
    for c, l, o, k in ((6, 23, 5, 4), (3, 11, 8, 3), (1, 7, 1, 1)):
        a = rng.randn(c, l).astype(np.float32)
        w = rng.randn(o, c, k).astype(np.float32)
        b = rng.randn(o).astype(np.float32)
        _check(a, w, b, 1, 0, 1, torch.float32, 1e-4, 1e-5)
        _check(a, w, b, 2, 1, 1, torch.float32, 1e-4, 1e-5)
        _check(a, w, None, 1, 2, 2, torch.float32, 1e-4, 1e-5)
    print("test_conv1d_unbatched PASS")


def test_conv1d_bf16():
    rng = np.random.RandomState(2)
    a = rng.randn(2, 6, 23)
    w = rng.randn(5, 6, 4)
    b = rng.randn(5)
    for stride, padding, dilation in ((1, 0, 1), (2, 0, 1), (1, 2, 1), (1, 3, 2)):
        a_t = torch.tensor(a, dtype=torch.float32).to(torch.bfloat16)
        w_t = torch.tensor(w, dtype=torch.float32).to(torch.bfloat16)
        b_t = torch.tensor(b, dtype=torch.float32).to(torch.bfloat16)
        want = F.conv1d(a_t, w_t, b_t, stride=stride, padding=padding,
                        dilation=dilation).float().numpy()
        mx_a = mx.view(mx.array(a_t.view(torch.uint16).numpy()), mx.bfloat16)
        mx_w = mx.view(mx.array(w_t.view(torch.uint16).numpy()), mx.bfloat16)
        mx_b = mx.view(mx.array(b_t.view(torch.uint16).numpy()), mx.bfloat16)
        got = mlx_vulkan._conv1d(mx_a, mx_w, mx_b, stride=stride,
                                 padding=padding, dilation=dilation)
        got_c = mx.astype(got, mx.float32)
        got_np = np.array(got_c)
        assert got_np.shape == want.shape, (got_np.shape, want.shape)
        assert np.allclose(got_np, want, rtol=5e-2, atol=2e-1), \
            float(np.abs(got_np - want).max())
    print("test_conv1d_bf16 PASS")


def test_conv1d_groups():
    # torch w is (O, I/groups, K) under groups; mlx wants (O, K, I/groups).
    rng = np.random.RandomState(3)
    cases = (
        # n, c, l, o, k, groups, stride, padding, dilation, use_b
        (2, 6, 23, 8, 4, 2, 1, 0, 1, True),    # plain grouped
        (1, 8, 31, 8, 5, 4, 2, 2, 1, True),    # grouped + stride + pad
        (1, 8, 40, 8, 3, 8, 1, 3, 2, False),   # depthwise g==C_in
        (2, 12, 17, 12, 5, 3, 2, 1, 2, True),  # depthwise g==C_in, d2 s2
        (1, 4, 19, 6, 3, 2, 1, 4, 1, False),   # grouped, big pad, no bias
    )
    for n, c, l, o, k, g, st, pd, dl, use_b in cases:
        a = rng.randn(n, c, l).astype(np.float32)
        w = rng.randn(o, c // g, k).astype(np.float32)
        b = rng.randn(o).astype(np.float32)
        _check(a, w, b if use_b else None, st, pd, dl, torch.float32,
               1e-4, 1e-5, groups=g)
    print("test_conv1d_groups PASS")


def test_conv1d_census_lowered():
    """torch.compile on a functional conv1d call lowers conv1d (no CPU fallback).

    Uses the gen_qwen logging pattern (basicConfig to a file before the
    backend import) so the LOWERED lines land where the runtime emits them.
    """
    import logging
    import torch.nn.functional as F

    logging.basicConfig(level=logging.INFO,
                        filename="/tmp/w7y-conv1d-census.log", filemode="w",
                        force=True)
    import torch  # noqa: F401  (imported after basicConfig like gen_qwen)
    import torch.nn.functional as F
    import mlx_vulkan

    os.environ["MLX_VK_EAGER"] = "1"  # per-node census: eager walk logs LOWERED/FALLBACK

    def f(x, w, b):
        return torch.ops.aten.conv1d.default(x, w, b, [2], [2], [1], 1)

    cf = torch.compile(backend="mlx_vulkan", dynamic=False)(f)
    x = torch.zeros(1, 6, 23)
    w = torch.zeros(5, 6, 4)
    b = torch.zeros(5)
    cf(x, w, b)
    with open("/tmp/w7y-conv1d-census.log") as fh:
        out = fh.read()
    assert "unlowerable" not in out, out[-800:]
    assert "FALLBACK" not in out, out[-800:]
    assert "LOWERED\tconv1d" in out, out[-800:]
    print("test_conv1d_census_lowered PASS")


def test_conv1d_census_grouped():
    """A grouped (depthwise) conv1d through torch.compile lowers, not FALLBACK."""
    import logging
    import torch.nn.functional as F

    logging.basicConfig(level=logging.INFO,
                        filename="/tmp/w7t-conv1d-census-grouped.log",
                        filemode="w", force=True)
    import torch  # noqa: F401
    import torch.nn.functional as F
    import mlx_vulkan

    os.environ["MLX_VK_EAGER"] = "1"  # per-node census: eager walk logs LOWERED/FALLBACK

    def f(x, w):
        return torch.ops.aten.conv1d.default(x, w, None, [1], [3], [2], 8)

    cf = torch.compile(backend="mlx_vulkan", dynamic=False)(f)
    x = torch.zeros(1, 8, 40)
    w = torch.zeros(8, 1, 3)
    cf(x, w)
    with open("/tmp/w7t-conv1d-census-grouped.log") as fh:
        out = fh.read()
    assert "unlowerable" not in out, out[-800:]
    assert "FALLBACK" not in out, out[-800:]
    assert "LOWERED\tconv1d" in out, out[-800:]
    print("test_conv1d_census_grouped PASS")


def test_conv1d_unbatched_executor():
    # PT-U5 residual complement to test_conv1d_unbatched (PAR-41, #1098):
    # the unbatched (C, L) form must route through the executor's mlx leg
    # end-to-end, not just lower correctly at the table level.
    rng = np.random.RandomState(7)
    a = rng.randn(4, 16).astype(np.float32)   # (C, L) unbatched
    w = rng.randn(3, 4, 5).astype(np.float32)  # (O, I, K)
    g = torch.fx.Graph()
    px, pw = g.placeholder("x"), g.placeholder("w")
    g.output(g.call_function(torch.ops.aten.conv1d.default, (px, pw)))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(torch.tensor(a), torch.tensor(w))
    exp = torch.ops.aten.conv1d.default(torch.tensor(a), torch.tensor(w))
    np.testing.assert_allclose(np.asarray(out), exp.numpy(), rtol=1e-4, atol=1e-5)
    assert "conv1d" in stats["mlx"], stats
    assert "conv1d" not in stats["cpu"], stats
    print("test_conv1d_unbatched_executor PASS")


if __name__ == "__main__":
    test_conv1d_fp32_configs()
    test_conv1d_fp32_shapes()
    test_conv1d_bf16()
    test_conv1d_groups()
    test_conv1d_census_lowered()
    test_conv1d_census_grouped()
    test_conv1d_unbatched()
    test_conv1d_unbatched_executor()
