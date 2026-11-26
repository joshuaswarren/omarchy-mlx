# SPDX-License-Identifier: MIT
# Copyright 2026 Joshua Warren
"""CHANGES findings on #773, each pinned by a failing-first test.

1. The whole-graph reshape gate caught detach / gelu_backward /
   native_layer_norm_backward / split and fed args[1] to mx.reshape before
   their dedicated handlers ran, so the first such node raised inside the
   compiled run and guarded() silently fell back to the eager executor.
2. The inline native_layer_norm_backward tested mask membership
   (1 in out_mask is True when out_mask[0] is True) instead of mask slots,
   and ignored out_mask[0] for dx.
3. The embedding lowering dropped optional aten arguments and raised
   TypeError whenever aot emitted them.
4. _opname stopped resolving torch-level call_function targets through
   __name__, so matmul/linear/sdpa nodes missed their lowerings.
5. _ln_native variance E[x^2]-mu^2 can cancel below zero and rsqrt of a
   negative variance is NaN.
"""
import logging
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")
import torch
import torch.nn.functional as F
from torch import nn


class DetachSplitGeluLN(nn.Module):
    """Forward graph carrying detach (one arg), split (int arg) and GELU,
    with LayerNorm so the aot backward adds native_layer_norm_backward.
    The norm acts on the (N, 4) chunk half, so LayerNorm(4)."""

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(4)
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        h = x.detach()
        a, b = torch.chunk(h, 2, dim=-1)
        return self.fc(F.gelu(self.ln(a))) + b.sum() * 0


def _compiled(mlx_vulkan, module):
    def backend(gm, example_inputs):
        return mlx_vulkan.mlx_vulkan_backend(gm, example_inputs)

    return torch.compile(module, backend=backend)


def test_detach_split_gelu_graph_lowers_in_the_compiled_executor(
        mlx_vulkan, caplog):
    module = DetachSplitGeluLN()
    x = torch.randn(2, 8)
    with caplog.at_level(logging.WARNING, logger="mlx_vulkan"):
        out = _compiled(mlx_vulkan, module)(x)
    assert not any("falling back to eager" in r.message
                   for r in caplog.records), \
        "the compiled executor fell back to eager on detach/split/gelu"
    assert torch.allclose(out, module(x), rtol=1e-4, atol=1e-6)


def test_multi_output_graph_returns_all_outputs(mlx_vulkan):
    """run() must return one result per graph output. The pre-fix code
    returned the bare first output, so aot iterated it row-wise: x*2 on a
    (2, 8) input came back as (8,) and a two-output module yielded two ROWS
    of output 0 (w7S comment 7183883072, verified live on torch 2.15)."""
    class TwoOut(nn.Module):
        def forward(self, x):
            return x * 2, x + 1

    x = torch.randn(2, 8)
    a, b = _compiled(mlx_vulkan, TwoOut())(x)
    assert a.shape == x.shape and b.shape == x.shape, \
        "the compiled run returned rows of output 0 instead of both outputs"
    assert torch.allclose(a, x * 2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(b, x + 1, rtol=1e-5, atol=1e-6)


def test_layer_norm_backward_obeys_the_output_mask(mlx_vulkan):
    torch.manual_seed(0)
    go = mx.array(torch.randn(2, 8).numpy())
    inp = mx.array(torch.randn(2, 8).numpy())
    mean = inp.mean(axis=-1, keepdims=True)
    rstd = mx.rsqrt(
        inp.square().mean(axis=-1, keepdims=True) - mean.square() + 1e-5)
    weight = mx.array(torch.randn(8).numpy())

    dx, dw, db = mlx_vulkan._nln_backward(
        go, inp, (8,), mean, rstd, weight, [True, False, False])
    assert dw is None and db is None
    assert dx is not None and bool(mx.isfinite(dx).all())

    dx2, dw2, db2 = mlx_vulkan._nln_backward(
        go, inp, (8,), mean, rstd, weight, [False, True, True])
    assert dx2 is None
    assert dw2 is not None and db2 is not None


def test_embedding_ignores_optional_aten_arguments(mlx_vulkan):
    torch.manual_seed(0)
    weight = mx.array(torch.randn(6, 4).numpy())
    idx = mx.array(torch.tensor([1, 3]).numpy())
    out = mlx_vulkan._embedding(weight, idx, padding_idx=0,
                                scale_grad_by_freq=False, sparse=False)
    assert bool(mx.array_equal(out, mlx_vulkan._embedding(weight, idx)))


def test_opname_resolves_torch_level_targets(mlx_vulkan):
    assert mlx_vulkan._opname(
        SimpleNamespace(target=torch.matmul)) == "matmul"
    assert mlx_vulkan._opname(
        SimpleNamespace(target=F.linear)) == "linear"
    aten_mm = torch.ops.aten.mm.default
    assert mlx_vulkan._opname(SimpleNamespace(target=aten_mm)) == "mm"


def test_layer_norm_native_rstd_is_finite_on_flat_rows(mlx_vulkan):
    # E[x^2]-mu^2 in float32 cancels below zero on near-flat rows (variance
    # ~1e-6 against a float32 rounding floor of ~0.06 at magnitude 1e6), so
    # the pre-clamp code fed a negative variance to rsqrt and returned NaN.
    # The clamp trades exactness on cancellation rows for finiteness.
    bad = 0
    for seed in range(24):
        torch.manual_seed(seed)
        a = 1000.0 + 0.001 * torch.randn(2, 8)
        _, _, rstd = mlx_vulkan._ln_native(mx.array(a.numpy()), (8,), None,
                                           None, 1e-5)
        if not bool(mx.isfinite(rstd).all()):
            bad += 1
    assert bad == 0, "rstd is NaN on flat rows in %d/24 seeds" % bad


def test_layer_norm_native_matches_torch_off_the_cancellation_floor(
        mlx_vulkan):
    # Rows far from the cancellation floor (variance 2.5e-3 vs rounding
    # noise ~1e-6) must still match torch closely; the clamp never fires.
    torch.manual_seed(0)
    a = 4.0 + 0.05 * torch.randn(2, 8)
    out = mlx_vulkan._ln_native(mx.array(a.numpy()), (8,), None, None, 1e-5)[0]
    ref = torch.nn.functional.layer_norm(a, (8,), eps=1e-5)
    assert float(mx.abs(out - mx.array(ref.numpy())).max()) < 5e-3
