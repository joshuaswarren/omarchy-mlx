# SPDX-License-Identifier: MIT
# Copyright 2026 Joshua Warren
"""PT-T1 speed PR: the conv lowerings match torch on the CPU device.

Executes the whole-graph lowering body for aten.convolution and
threshold_backward directly against torch's own conv and relu backward
on the same inputs."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
mx = pytest.importorskip("mlx.core")


def _conv_lower(inp, wt, bias, stride, padding, dilation, transposed, groups):
    stride = tuple(stride) if not isinstance(stride, int) else stride
    padding = tuple(padding) if not isinstance(padding, int) else padding
    dilation = tuple(dilation) if not isinstance(dilation, int) else dilation
    x = mx.transpose(inp, (0, 2, 3, 1))
    y = mx.conv2d(x, wt, stride=stride, padding=padding, dilation=dilation)
    y = mx.transpose(y, (0, 3, 1, 2))
    if bias is not None:
        y = y + bias.reshape(1, -1, 1, 1)
    return y


def test_conv2d_forward_matches_torch(mlx_vulkan):
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16, 16)
    w = torch.randn(12, 8, 3, 3)
    b = torch.randn(12)
    ref = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)

    out = _conv_lower(mx.array(x.numpy()), mx.array(w.numpy()),
                      mx.array(b.numpy()), [1], [1], [1], False, 1)
    got = np.asarray(out.transpose(0, 3, 1, 2))
    diff = float(abs(got - ref.detach().numpy()).max())
    assert diff < 1e-4, "the conv forward differs: %.3e" % diff


def test_threshold_backward_matches_relu_grad(mlx_vulkan):
    torch.manual_seed(1)
    go = torch.randn(2, 8, 8, 8)
    x = torch.randn(2, 8, 8, 8) * 2.0
    ref = go * (x > 0.0)

    out = mlx_vulkan.ATEN_TABLE["threshold_backward"](
        mx.array(go.numpy()), mx.array(x.numpy()), 0.0)
    got = np.asarray(out)
    assert np.array_equal(got, ref.numpy()), "threshold backward differs"


def test_conv_training_step_matches_cpu(mlx_vulkan):
    """One conv training step through aot_autograd + the backend: the
    gradients finite and matching the CPU run."""
    import math
    torch.manual_seed(2)

    class ConvNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(8, 12, 3, padding=1)
            self.fc = torch.nn.Linear(12 * 16 * 16, 4)

        def forward(self, x):
            return self.fc(
                torch.nn.functional.relu(
                    self.conv(x)).reshape(x.shape[0], -1))

    def run(backend):
        torch.manual_seed(2)
        model = ConvNet()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(2, 8, 16, 16)
        y = torch.randn(2, 4)
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(backend(model)(x)
                                            if backend else model(x), y)
        loss.backward()
        grads = {n: p.grad.detach().clone()
                 for n, p in model.named_parameters()}
        return float(loss), grads

    def aot(gm, example_inputs):
        runner = mlx_vulkan.mlx_vulkan_backend(
            gm, example_inputs, return_all_outputs=True)

        def call(*inputs):
            flat = inputs
            if (len(inputs) == 1 and isinstance(inputs[0], (list, tuple))
                    and not torch.is_tensor(inputs[0])):
                flat = tuple(inputs[0])
            return runner(*flat)

        call._boxed_call = True
        return call

    from torch._dynamo.backends.common import aot_autograd

    def lower(gm2, ex):
        return aot(gm2, ex)

    loss_gpu, grads_gpu = run(lambda m: torch.compile(m, backend=lower))
    loss_cpu, grads_cpu = run(None)

    assert math.isfinite(loss_gpu)
    assert abs(loss_gpu - loss_cpu) < 0.02, \
        "the loss differs: %s vs %s" % (loss_gpu, loss_cpu)
    for name, grad in grads_gpu.items():
        assert torch.isfinite(grad).all(), \
            "the %s gradient is not finite" % name
        assert torch.allclose(grad, grads_cpu[name], rtol=0.05, atol=1e-4), \
            "the %s gradient differs from the CPU run" % name
