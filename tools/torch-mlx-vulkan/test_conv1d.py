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


def _check(a, w, b, stride, padding, dilation, dtype, rtol, atol):
    if dtype is torch.bfloat16:
        a = torch.tensor(a, dtype=torch.bfloat16)
        w = torch.tensor(w, dtype=torch.bfloat16)
        b = torch.tensor(b, dtype=torch.bfloat16) if b is not None else None
    else:
        a = torch.tensor(a, dtype=dtype)
        w = torch.tensor(w, dtype=dtype)
        b = torch.tensor(b, dtype=dtype) if b is not None else None
    want = F.conv1d(a, w, b, stride=stride, padding=padding, dilation=dilation)
    mxa = mx.array(np.array(a))
    mxw = mx.array(np.array(w))
    mxb = mx.array(np.array(b)) if b is not None else None
    got = mlx_vulkan._conv1d(mxa, mxw, mxb, stride=stride, padding=padding, dilation=dilation)
    got = np.array(got)
    want_np = want.numpy()
    assert got.shape == want_np.shape, (got.shape, want_np.shape)
    if dtype is torch.bfloat16:
        ok = np.allclose(got.astype(np.float32), want_np.astype(np.float32), rtol=rtol, atol=atol)
    else:
        ok = np.allclose(got, want_np, rtol=rtol, atol=atol)
    assert ok, (dtype, stride, padding, dilation,
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


def test_conv1d_refuses_groups():
    import mlx.core as mx
    a = mx.zeros((1, 4, 8))
    w = mx.zeros((4, 2, 3))
    try:
        mlx_vulkan._conv1d(a, w, None, groups=2)
    except Exception as e:
        assert "grouped conv1d" in str(e), e
        print("test_conv1d_refuses_groups PASS")
        return
    raise AssertionError("grouped conv1d did not refuse")


if __name__ == "__main__":
    test_conv1d_fp32_configs()
    test_conv1d_fp32_shapes()
    test_conv1d_bf16()
    test_conv1d_refuses_groups()
