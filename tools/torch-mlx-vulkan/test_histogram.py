#!/usr/bin/env python3
"""Parity tests for the aten.histc / aten.histogram lowerings in mlx_vulkan.

fp32 and int64 parity vs torch CPU for: histc (default range, explicit range,
out-of-range dropped) and histogram (counts, weights, density, int bins + range,
list-of-edges form). Runs under plain python3 or pytest; needs torch + mlx.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _np(a):
    return np.array(a)


def test_histc_fp32_default_range():
    rng = np.random.RandomState(0)
    x = (rng.randn(128) * 3).astype(np.float32)
    for bins in (1, 4, 10, 100):
        got = _np(mlx_vulkan._histc(mx.array(x), bins))
        want = torch.histc(torch.from_numpy(x), bins=bins)
        assert got.shape == (bins,), (bins, got.shape)
        assert np.array_equal(got.astype(np.float32), want.numpy().astype(np.float32)), (bins, got, want.numpy())
    print("test_histc_fp32_default_range PASS")


def test_histc_fp32_explicit_range_and_out_of_range():
    rng = np.random.RandomState(1)
    x = (rng.randn(200) * 5).astype(np.float32)  # values beyond [0, 10] exist
    lo, hi, bins = 0.0, 10.0, 8
    got = _np(mlx_vulkan._histc(mx.array(x), bins, min=lo, max=hi))
    want = torch.histc(torch.from_numpy(x), bins=bins, min=lo, max=hi)
    assert np.array_equal(got.astype(np.float32), want.numpy().astype(np.float32)), (got, want.numpy())
    total_in = int(want.sum())
    assert total_in == ((x >= lo) & (x <= hi)).sum(), "out-of-range handling changed the mass"
    print("test_histc_fp32_explicit_range_and_out_of_range PASS")


def test_histc_int64_input():
    # torch.histc rejects int64 ("not implemented for Long"), so the int64 parity
    # reference is torch.histc on the int64->float32 view: same values, same bins.
    rng = np.random.RandomState(2)
    x = rng.randint(0, 50, size=300).astype(np.int64)
    got = _np(mlx_vulkan._histc(mx.array(x), bins=5, min=0, max=49))
    want = torch.histc(torch.from_numpy(x.astype(np.float32)), bins=5, min=0, max=49)
    assert got.dtype == np.float32
    assert np.array_equal(got.astype(np.float32), want.numpy().astype(np.float32)), (got, want.numpy())
    # int64 input through the histogram route (which does accept int64) parity too
    got2, edges2 = mlx_vulkan._histogram(mx.array(x), bins=5, range=(0.0, 49.0))
    want2, edges2 = torch.histogram(torch.from_numpy(x.astype(np.float32)), bins=5, range=(0.0, 49.0))
    assert np.array_equal(_np(got2).astype(np.float32), want2.numpy().astype(np.float32))
    print("test_histc_int64_input PASS")


def test_histogram_counts_fp32_and_int64():
    rng = np.random.RandomState(3)
    x = (rng.randn(300) * 4 + 2).astype(np.float32)
    for dtype in (torch.float32, torch.int64):
        # torch.histogram also rejects int64 on CPU; the int64 parity reference is the
        # float32 view (identical values, integer-valued so binning is identical).
        xn = np.round(x).astype(np.int64) if dtype == torch.int64 else x.astype(np.float32)
        t = torch.from_numpy(xn.astype(np.float32))
        got, edges = mlx_vulkan._histogram(mx.array(xn), bins=7, range=(-5.0, 5.0))
        want, want_edges = torch.histogram(t, bins=7, range=(-5.0, 5.0))
        assert np.array_equal(_np(got).astype(np.float64), want.numpy().astype(np.float64)), dtype
        assert np.allclose(_np(edges).astype(np.float64), want_edges.numpy().astype(np.float64)), dtype
    print("test_histogram_counts_fp32_and_int64 PASS")


def test_histogram_weights_and_density():
    rng = np.random.RandomState(4)
    x = (rng.randn(120) * 2).astype(np.float32)
    w = (rng.rand(120) + 0.5).astype(np.float32)
    bins, lo, hi = 6, -3.0, 3.0
    got, edges = mlx_vulkan._histogram(mx.array(x), bins=bins, range=(lo, hi), weight=mx.array(w))
    want, want_edges = torch.histogram(torch.from_numpy(x), bins=bins, range=(lo, hi), weight=torch.from_numpy(w))
    assert np.allclose(_np(got).astype(np.float64), want.numpy().astype(np.float64)), (got, want.numpy())
    assert np.allclose(_np(edges).astype(np.float64), want_edges.numpy().astype(np.float64))
    # density=True: sum(hist * width) == 1
    got_d, _ = mlx_vulkan._histogram(mx.array(x), bins=bins, range=(lo, hi), weight=mx.array(w), density=True)
    want_d, _ = torch.histogram(torch.from_numpy(x), bins=bins, range=(lo, hi), weight=torch.from_numpy(w), density=True)
    assert np.allclose(_np(got_d).astype(np.float64), want_d.numpy().astype(np.float64)), (got_d, want_d.numpy())
    width = (hi - lo) / bins
    assert abs(float(_np(got_d).sum() * width) - 1.0) < 1e-5, "density must integrate to 1"
    print("test_histogram_weights_and_density PASS")


def test_histogram_edges_list_form():
    rng = np.random.RandomState(5)
    x = rng.randn(80).astype(np.float32)
    edges_list = [-3.0, -1.0, 0.0, 1.0, 3.0]
    got, edges = mlx_vulkan._histogram(mx.array(x), bins=edges_list)
    want, want_edges = torch.histogram(torch.from_numpy(x), bins=torch.tensor(edges_list))
    assert np.array_equal(_np(got).astype(np.float64), want.numpy().astype(np.float64)), (got, want.numpy())
    assert np.allclose(_np(edges).astype(np.float64), want_edges.numpy().astype(np.float64))
    print("test_histogram_edges_list_form PASS")


if __name__ == "__main__":
    _provenance() if False else None
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ALL PASS")
