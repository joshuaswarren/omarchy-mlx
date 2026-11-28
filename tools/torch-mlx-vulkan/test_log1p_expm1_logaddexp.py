#!/usr/bin/env python3
"""PT-AL-f: aten.log1p / aten.expm1 / aten.logaddexp parity (stable log/exp family).

log1p already maps to mx.log1p; expm1 and logaddexp are new one-line
_FUNC_SPECS entries. The cases drive the aten.* overloads through
make_executor on hand-built fx graphs (compiled path, stats prove mlx
lowering with zero CPU fallback) and call the spec entries directly for
the eager path and bf16 dtype fidelity.

Near-zero inputs are the point of this family: log1p and expm1 exist
because naive log(1+x) / exp(x)-1 lose the mantissa there, and
logaddexp's near-equal-magnitude case is where naive log(exp(a)+exp(b))
overflows or underflows. fp32 compares tight at the near-zero grid and
at atol=1e-5 elsewhere; bf16 compares upcast at a tolerance that covers
the bf16 mantissa.

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
    n = g.call_function(target, (phs[0], phs[1]) if len(ins) == 2 else (phs[0],) + args, {})
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


NEAR_ZERO = [1e-8, -1e-8, 1e-6, -1e-6, 1e-3, -1e-3]


def test_log1p():
    # aten.log1p.default(self). Near-zero grid: naive log(1+x) loses the
    # mantissa below ~1e-3; log1p stays exact, and the parity must hold at
    # a tight atol there, not just at the 1e-5 general tolerance.
    x = torch.tensor(NEAR_ZERO + [-0.999, -0.9999, 0.0, 1.0, 10.0, 1e3], dtype=torch.float32)
    xb = x.bfloat16()
    ref = torch.log1p(x)
    tight = 5e-7 * max(1.0, float(ref.abs().max()))
    cases = [
        ((x.float(),), (), ref, 5e-6),
        ((xb,), (), torch.log1p(xb), 2e-2),
    ]
    for ins, args, r, atol in cases:
        _run_case(torch.ops.aten.log1p.default, ins, args, r, atol)
    # near-zero tight check on the fp32 compiled output
    _run_case(torch.ops.aten.log1p.default, (x,), (), ref, max(tight, 1e-6))
    low = mlx_vulkan._FUNC_SPECS["log1p"](mlx_vulkan._mx(xb))
    assert low.dtype == mx.bfloat16, low.dtype
    print("test_log1p PASS")


def test_expm1():
    # aten.expm1.default(self). Same near-zero grid; the large tail
    # (x = 20) overflows naive exp(x)-1 in f32 at x >= 89 only, 20 stays
    # representable and checks magnitude scaling.
    x = torch.tensor(NEAR_ZERO + [-20.0, -1.0, 0.0, 1.0, 5.0, 20.0], dtype=torch.float32)
    xb = x.bfloat16()
    ref = torch.expm1(x)
    cases = [
        ((x,), (), ref, 5e-6),
        ((xb,), (), torch.expm1(xb), 2e-2),
    ]
    for ins, args, r, atol in cases:
        _run_case(torch.ops.aten.expm1.default, ins, args, r, atol)
    near = torch.tensor(NEAR_ZERO, dtype=torch.float32)
    _run_case(torch.ops.aten.expm1.default, (near,), (), torch.expm1(near), 1e-6)
    low = mlx_vulkan._FUNC_SPECS["expm1"](mlx_vulkan._mx(xb))
    assert low.dtype == mx.bfloat16, low.dtype
    print("test_expm1 PASS")


def test_logaddexp():
    # aten.logaddexp.default(self, other). The near-equal-magnitude case
    # at large negative values is the numerically hard one: naive
    # log(exp(a)+exp(b)) underflows to -inf there; the stable identity
    # max(a,b) + log1p(exp(-|a-b|)) must match torch bit-tight in fp32.
    cases_t = [
        (torch.tensor([-1e6, -1e3, -20.0], dtype=torch.float32),
         torch.tensor([-1e6 + 1e-6, -1e3 + 1e-3, -20.0 + 1e-5], dtype=torch.float32)),
        (torch.tensor(NEAR_ZERO, dtype=torch.float32),
         torch.tensor([-n for n in NEAR_ZERO], dtype=torch.float32)),
        (torch.tensor([1.0, -1.0, 0.0, 50.0], dtype=torch.float32),
         torch.tensor([-1.0, 1.0, 0.0, -50.0], dtype=torch.float32)),
    ]
    for a, b in cases_t:
        ab, bb = a.bfloat16(), b.bfloat16()
        ref = torch.logaddexp(a, b)
        cases = [
            ((a, b), (), ref, 5e-6),
            ((ab, bb), (), torch.logaddexp(ab, bb), 2e-2),
        ]
        for ins, args, r, atol in cases:
            _run_case(torch.ops.aten.logaddexp.default, ins, args, r, atol)
    # near-equal large-negative tight check in fp32
    a = torch.tensor([-1e6, -1e4], dtype=torch.float32)
    b = torch.tensor([-1e6 + 1e-6, -1e4 + 1e-5], dtype=torch.float32)
    _run_case(torch.ops.aten.logaddexp.default, (a, b), (), torch.logaddexp(a, b), 1e-4)
    low = mlx_vulkan._FUNC_SPECS["logaddexp"](mlx_vulkan._mx(a.bfloat16()), mlx_vulkan._mx(b.bfloat16()))
    assert low.dtype == mx.bfloat16, low.dtype
    print("test_logaddexp PASS")


if __name__ == "__main__":
    test_log1p()
    test_expm1()
    test_logaddexp()
    print("PT-AL-f ALL PASS")
