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
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")
import torch
import torch.nn.functional as F
from torch import nn


class DetachSplitGeluLN(nn.Module):
    """Forward graph carrying detach (one arg), split (int arg) and GELU,
    with LayerNorm so the aot backward adds native_layer_norm_backward."""

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 4)

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
    # E[x^2]-mu^2 in float32 on rows of ~1000.0 cancels below zero; the
    # pre-clamp code fed the negative variance to rsqrt and returned NaN.
    torch.manual_seed(0)
    a = 1000.0 + 0.001 * torch.randn(2, 8)
    _, _mu, rstd = mlx_vulkan._ln_native(mx.array(a.numpy()), (8,), None,
                                         None, 1e-5)
    assert bool(mx.isfinite(rstd).all()), "rstd is NaN on flat rows"
    ref = torch.nn.functional.layer_norm(a, (8,), eps=1e-5)
    out = mlx_vulkan._ln_native(mx.array(a.numpy()), (8,), None, None, 1e-5)[0]
    assert math.isclose(
        float(mx.abs(out - mx.array(ref.numpy())).max()), 0.0, abs_tol=1e-2)
