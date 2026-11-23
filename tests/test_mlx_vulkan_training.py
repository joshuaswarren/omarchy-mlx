# SPDX-License-Identifier: MIT
# Copyright 2026 Joshua Warren
"""PT-T1 regression: one training step through aot_autograd + mlx_vulkan
must produce finite gradients when the graph contains a LayerNorm.

The old _ln_native returned dummy mean/rstd, which aot_autograd saves and
feeds to aten.native_layer_norm_backward: rstd = 0 divided everywhere, and
every gradient upstream of the norm turned NaN at the first training step
(the nanoGPT phase of G708 PT-T1: step 0 bit-exact, step 1 all-NaN).
"""
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn


class TinyLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.ln(x))


def _one_step(mlx_vulkan, backend):
    torch.manual_seed(0)
    model = TinyLN()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    step_fn = torch.compile(model, backend=backend) if backend else model
    x = torch.randn(2, 8)
    target = torch.randn(2, 4)
    opt.zero_grad()
    loss = F.mse_loss(step_fn(x), target)
    loss.backward()
    grads = {n: (None if p.grad is None else p.grad.detach().clone())
             for n, p in model.named_parameters()}
    return float(loss), grads


def _aot(mlx_vulkan, gm, example_inputs):
    from torch._dynamo.backends.common import aot_autograd

    def lower(gm2, ex):
        return mlx_vulkan.mlx_vulkan_backend(gm2, ex)

    return aot_autograd(fw_compiler=lower, bw_compiler=lower)(gm, example_inputs)


def test_layer_norm_training_step_gradients_are_finite_and_match(mlx_vulkan):
    def backend(gm, example_inputs):
        return _aot(mlx_vulkan, gm, example_inputs)

    loss_gpu, grads_gpu = _one_step(mlx_vulkan, backend)
    loss_cpu, grads_cpu = _one_step(mlx_vulkan, None)

    assert math.isfinite(loss_gpu), "the loss is not finite"
    for name, grad in grads_gpu.items():
        assert grad is not None and torch.isfinite(grad).all(), \
            "the %s gradient is not finite" % name
        assert torch.allclose(grad, grads_cpu[name], rtol=1e-4, atol=1e-6), \
            "the %s gradient differs from the CPU run" % name
    assert math.isclose(loss_gpu, loss_cpu, rel_tol=1e-5)
