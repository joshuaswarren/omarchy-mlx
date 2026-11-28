#!/usr/bin/env python3
"""PT-AL-a: aten.elu / aten.selu / aten.celu parity (activation family).

Each op is a one-line _FUNC_SPECS entry; the cases drive the aten.*
overloads through make_executor on hand-built fx graphs (compiled path,
stats prove mlx lowering with zero CPU fallback) and call the spec entry
directly for the eager path and bf16 dtype fidelity.

fp32 compares at atol=1e-5; bf16 compares upcast at a tolerance that
covers the bf16 mantissa.

Runs under plain python3 (no pytest needed) or pytest.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _run_case(target, ins, args, ref, atol):
    g = torch.fx.Graph()
    phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
    n = g.call_function(target, (phs[0],) + args, {})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(*ins)
    ref = torch.as_tensor(ref)
    assert stats["cpu"] == [], (stats, [str(t.target) for t in gm.graph.nodes])
    got = out.float().numpy()
    assert np.allclose(got, ref.float().numpy(), atol=atol), \
        (str(target), args, got, ref.float().numpy())
    assert out.dtype == ref.dtype or (
        ref.dtype == torch.bfloat16 and out.dtype == torch.float32), \
        (str(target), out.dtype, ref.dtype)


def test_elu():
    # aten.elu.default(self, alpha=1, scale=1, input_scale=1). Non-default
    # args go positionally, matching how the dispatcher binds them.
    x = torch.tensor([[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0],
                      [-2.0, -0.25, 0.25, 2.0, -4.0, 4.0, 0.0]])
    xb = x.bfloat16()
    cases = [
        ((x,), (), torch.ops.aten.elu.default(x), 1e-5),
        ((x,), (2.0,), torch.ops.aten.elu.default(x, 2.0), 1e-5),
        ((x,), (2.0, 1.5, 0.5), torch.ops.aten.elu.default(x, 2.0, 1.5, 0.5), 1e-5),
        ((xb,), (), torch.ops.aten.elu.default(xb), 2e-2),
    ]
    for ins, args, ref, atol in cases:
        _run_case(torch.ops.aten.elu.default, ins, args, ref, atol)
    # eager spec: same formula, bf16 stays bf16 through the lowering
    low = mlx_vulkan._FUNC_SPECS["elu"](mlx_vulkan._mx(xb), 2.0, 1.5, 0.5)
    assert low.dtype == mx.bfloat16, low.dtype
    ref = torch.ops.aten.elu.default(xb, 2.0, 1.5, 0.5)
    assert np.allclose(np.array(low.astype(mx.float32)), ref.float().numpy(), atol=2e-2)
    print("test_elu PASS")


def test_selu():
    # aten.selu.default(self): fixed alpha 1.6732632423543772, scale
    # 1.0507009873554805. No parameters.
    x = torch.tensor([[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0],
                      [-2.0, -0.25, 0.25, 2.0, -4.0, 4.0, 0.0]])
    xb = x.bfloat16()
    cases = [
        ((x,), torch.ops.aten.selu.default(x), 1e-5),
        ((xb,), torch.ops.aten.selu.default(xb), 2e-2),
    ]
    for ins, ref, atol in cases:
        _run_case(torch.ops.aten.selu.default, ins, (), ref, atol)
    low = mlx_vulkan._FUNC_SPECS["selu"](mlx_vulkan._mx(xb))
    assert low.dtype == mx.bfloat16, low.dtype
    ref = torch.ops.aten.selu.default(xb)
    assert np.allclose(np.array(low.astype(mx.float32)), ref.float().numpy(), atol=2e-2)
    print("test_selu PASS")


def test_celu():
    # aten.celu.default(self, alpha=1.0): max(0,x) + min(0, alpha*expm1(x/alpha)).
    x = torch.tensor([[-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0],
                      [-2.0, -0.25, 0.25, 2.0, -4.0, 4.0, 0.0]])
    xb = x.bfloat16()
    cases = [
        ((x,), (), torch.ops.aten.celu.default(x), 1e-5),
        ((x,), (2.0,), torch.ops.aten.celu.default(x, 2.0), 1e-5),
        ((xb,), (), torch.ops.aten.celu.default(xb), 2e-2),
        ((xb,), (2.0,), torch.ops.aten.celu.default(xb, 2.0), 2e-2),
    ]
    for ins, args, ref, atol in cases:
        _run_case(torch.ops.aten.celu.default, ins, args, ref, atol)
    low = mlx_vulkan._FUNC_SPECS["celu"](mlx_vulkan._mx(xb), 2.0)
    assert low.dtype == mx.bfloat16, low.dtype
    ref = torch.ops.aten.celu.default(xb, 2.0)
    assert np.allclose(np.array(low.astype(mx.float32)), ref.float().numpy(), atol=2e-2)
    print("test_celu PASS")


if __name__ == "__main__":
    test_elu()
    test_selu()
    test_celu()
    print("ALL PASS")
