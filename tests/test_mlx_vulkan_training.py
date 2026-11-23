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

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")
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
        runner = mlx_vulkan.mlx_vulkan_backend(gm2, ex,
                                               return_all_outputs=True)

        def call(*inputs):
            flat = inputs
            if (len(inputs) == 1 and isinstance(inputs[0], (list, tuple))
                    and not torch.is_tensor(inputs[0])):
                flat = tuple(inputs[0])
            return runner(*flat)

        call._boxed_call = True
        return call

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


def test_whole_graph_training_step_stays_on_mx(mlx_vulkan, caplog):
    """wAX/PT-T1 speed gate: the nanoGPT-shaped step must run the whole
    graph under mx.compile - no per-node executor fallback, no torch CPU
    fallback records from the backend's own logger."""
    import logging

    torch.manual_seed(0)

    class Nano(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(65, 96)
            self.pos = nn.Parameter(torch.randn(1, 64, 96) * 0.02)
            self.blocks = nn.ModuleList()
            for _ in range(2):
                self.blocks.append(nn.ModuleDict({
                    "ln1": nn.LayerNorm(96),
                    "qkv": nn.Linear(96, 288),
                    "proj": nn.Linear(96, 96),
                    "ln2": nn.LayerNorm(96),
                    "fc1": nn.Linear(96, 384),
                    "fc2": nn.Linear(384, 96)}))

        def forward(self, idx):
            h = self.emb(idx) + self.pos
            B, T, C = h.shape
            for blk in self.blocks:
                q, k, v = blk["qkv"](blk["ln1"](h)).chunk(3, dim=2)
                q = q.view(B, T, 6, 16).transpose(1, 2)
                k = k.view(B, T, 6, 16).transpose(1, 2)
                v = v.view(B, T, 6, 16).transpose(1, 2)
                a = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
                h = h + blk["proj"](a)
                h = h + blk["fc2"](torch.nn.functional.gelu(
                    blk["fc1"](blk["ln2"](h))))
            return self.head(h) if hasattr(self, "head") else h

    model = Nano()
    model.head = nn.Linear(96, 65)
    x = torch.randint(0, 65, (2, 64))
    y = torch.randint(0, 65, (2 * 64,))

    def backend(gm, example_inputs):
        from torch._dynamo.backends.common import aot_autograd

        def lower(gm2, ex):
            runner = mlx_vulkan.mlx_vulkan_backend(gm2, ex,
                                                   return_all_outputs=True)

            def call(*inputs):
                flat = inputs
                if (len(inputs) == 1
                        and isinstance(inputs[0], (list, tuple))
                        and not torch.is_tensor(inputs[0])):
                    flat = tuple(inputs[0])
                return runner(*flat)

            call._boxed_call = True
            return call

        return aot_autograd(fw_compiler=lower, bw_compiler=lower)(gm,
                                                                  example_inputs)

    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    log = logging.getLogger("mlx_vulkan")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        step_fn = torch.compile(model, backend=backend)
        loss = torch.nn.functional.cross_entropy(
            step_fn(x).reshape(-1, 65), y)
        loss.backward()
        assert math.isfinite(float(loss))
    finally:
        log.removeHandler(handler)

    assert not [m for m in records if "FALLBACK" in m], \
        "the per-node executor fell back to torch: %s" % records[:4]
    assert not [m for m in records if "compiled executor failed" in m], \
        "the whole-graph executor refused the graph: %s" % records[:4]
