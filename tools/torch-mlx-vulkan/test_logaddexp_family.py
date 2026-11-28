#!/usr/bin/env python3
"""Q-16: aten.hypot / aten.logcumsumexp / aten.logaddexp2 parity.

The three one-line-ish _FUNC_SPECS entries sit next to the wheel's own
logaddexp. Each op is driven through make_executor on a hand-built fx
graph (the compiled path; stats prove the mlx lowering with zero CPU
fallback) and called directly for the eager path, the dtype fidelity,
and the edge cases the formulas carry:

- hypot: the scaled form keeps 1e200-scale inputs from overflowing and
  hypot(0, 0) = 0.
- logcumsumexp: the all--inf row stays -inf (the shift's NaN maps back).
- logaddexp2: logaddexp2(inf, -inf) = inf, equal inputs exact.

fp32 compares against the torch CPU reference at atol=1e-5; bf16 goes
through fp32 opmath with the family tolerance.

Runs under plain python3 (no pytest needed) or pytest.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _run_case(target, ins, args, ref, atol=1e-5):
    g = torch.fx.Graph()
    phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
    n = g.call_function(target, tuple(phs) + tuple(args), {})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(*ins)
    assert stats["cpu"] == [], (stats, [str(t.target) for t in gm.graph.nodes])
    got = out.float().numpy()
    ref = torch.as_tensor(ref).float().numpy()
    assert np.allclose(got, ref, atol=atol, equal_nan=False), \
        (str(target), args, got, ref)


def test_family_parity():
    a = torch.randn(4, 6) * 3.0
    b = torch.randn(4, 6) * 3.0
    A, B = mx.array(a.numpy()), mx.array(b.numpy())
    # hypot: fp32 parity vs torch CPU, plus the overflow-scale edge the
    # scaled form exists for, plus hypot(0, 0) = 0.
    _run_case(torch.ops.aten.hypot.default, (A, B), (),
              torch.hypot(a, b))
    big = torch.tensor([[1e200, -1e200], [3e150, 0.0]])
    got = mlx_vulkan.ATEN_TABLE["hypot"](
        mx.array(big.numpy()), mx.array(big.numpy()))
    ref = torch.hypot(big, big.flip(1))
    assert np.allclose(got.astype(mx.float32).__array__(),
                       ref.numpy(), rtol=1e-6), (got, ref)
    z = mlx_vulkan.ATEN_TABLE["hypot"](mx.array(0.0), mx.array(0.0))
    assert float(z) == 0.0, z
    # hypot(inf, inf) = inf and hypot(-inf, -inf) = inf - the scaled
    # form would give inf * sqrt(1 + inf/inf) = NaN without the mask.
    ii = torch.tensor([[float("inf"), -float("inf")]])
    got = mlx_vulkan.ATEN_TABLE["hypot"](
        mx.array(ii.numpy()), mx.array(ii.numpy()))
    ref = torch.hypot(ii, ii)
    assert np.allclose(got.astype(mx.float32).__array__(),
                       ref.numpy()), (got, ref)
    # hypot_ (in-place variant maps to the same math).
    g = torch.fx.Graph()
    pa, pb = g.placeholder("a"), g.placeholder("b")
    g.output(g.call_function(torch.ops.aten.hypot_.default, (pa, pb)))
    run, stats = mlx_vulkan.make_executor(
        torch.fx.GraphModule(torch.nn.Module(), g))
    assert stats["cpu"] == [], stats
    # logcumsumexp: parity along both dims, and the all--inf row keeps
    # -inf (the shifted cumsum's NaN maps back).
    _run_case(torch.ops.aten.logcumsumexp.default, (A,), (0,),
              torch.logcumsumexp(a, dim=0))
    _run_case(torch.ops.aten.logcumsumexp.default, (A,), (1,),
              torch.logcumsumexp(a, dim=1))
    pad = torch.tensor([[-float("inf"), -float("inf")],
                        [-1.0, -2.0]])
    got = mlx_vulkan.ATEN_TABLE["logcumsumexp"](
        mx.array(pad.numpy()), 1)
    ref = torch.logcumsumexp(pad, dim=1)
    assert np.allclose(got.astype(mx.float32).__array__(),
                       ref.numpy(), equal_nan=True), (got, ref)
    # aten ends the prefix at the first non-finite entry: +inf gives
    # +inf from there on, NaN gives NaN - neither poisons the positions
    # before it.
    hit = torch.tensor([[1.0, float("inf"), 2.0],
                        [float("-inf"), float("inf"), 3.0]])
    got = mlx_vulkan.ATEN_TABLE["logcumsumexp"](
        mx.array(hit.numpy()), 1)
    ref = torch.logcumsumexp(hit, dim=1)
    assert np.allclose(got.astype(mx.float32).__array__(),
                       ref.numpy(), equal_nan=True), (got, ref)
    nanr = torch.tensor([[1.0, float("nan"), 2.0]])
    got = mlx_vulkan.ATEN_TABLE["logcumsumexp"](
        mx.array(nanr.numpy()), 1)
    ref = torch.logcumsumexp(nanr, dim=1)
    assert np.allclose(got.astype(mx.float32).__array__(),
                       ref.numpy(), equal_nan=True), (got, ref)
    # logaddexp2: parity, the inf/-inf asymmetry, and equal inputs.
    _run_case(torch.ops.aten.logaddexp2.default, (A, B), (),
              torch.logaddexp2(a, b))
    inf = torch.tensor([[float("inf"), -float("inf")],
                        [2.0, 2.0]])
    got = mlx_vulkan.ATEN_TABLE["logaddexp2"](
        mx.array(inf.numpy()), mx.array((-inf).numpy()))
    ref = torch.logaddexp2(inf, -inf)
    assert np.allclose(got.astype(mx.float32).__array__(),
                       ref.numpy()), (got, ref)
    e = mlx_vulkan.ATEN_TABLE["logaddexp2"](mx.array(3.0), mx.array(3.0))
    assert abs(float(e) - 4.0) < 1e-12, e  # log2(2*2^3) = 4
    # same-sign infinities: logaddexp2(inf, inf) = inf and
    # (-inf, -inf) = -inf - the |a-b| correction would NaN on inf-inf.
    for v, want in ((float("inf"), float("inf")),
                    (float("-inf"), float("-inf"))):
        got = mlx_vulkan.ATEN_TABLE["logaddexp2"](
            mx.array(v), mx.array(v))
        assert float(got) == want, (v, got)
    # bf16 fidelity: fp32 opmath within the family tolerance.
    for name, fn, ref_fn in (
            ("hypot", mlx_vulkan.ATEN_TABLE["hypot"],
             lambda x, y: torch.hypot(x, y)),
            ("logaddexp2", mlx_vulkan.ATEN_TABLE["logaddexp2"],
             lambda x, y: torch.logaddexp2(x, y))):
        ab, bb = a.to(torch.bfloat16), b.to(torch.bfloat16)
        mine = fn(mx.array(ab.float().numpy()).astype(mx.bfloat16),
                  mx.array(bb.float().numpy()).astype(mx.bfloat16))
        ref = ref_fn(ab, bb)
        d = float(np.abs(np.array(mine.astype(mx.float32).__array__(),
                                    dtype=np.float32)
                         - ref.float().numpy()).max())
        assert d < 0.05, (name, d)


def test_executor_path():
    # Every family node lowers to mlx, never the CPU fallback (parity is
    # re-asserted here so the route proof cannot pass on wrong values).
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    A, B = mx.array(a.numpy()), mx.array(b.numpy())
    _run_case(torch.ops.aten.hypot.default, (A, B), (),
              torch.hypot(a, b))
    _run_case(torch.ops.aten.logcumsumexp.default, (A,), (1,),
              torch.logcumsumexp(a, dim=1))
    _run_case(torch.ops.aten.logaddexp2.default, (A, B), (),
              torch.logaddexp2(a, b))
    _run_case(torch.ops.aten.logaddexp.default, (A, B), (),
              torch.logaddexp(a, b))


if __name__ == "__main__":
    test_family_parity()
    test_executor_path()
    print("Q-16 family: all checks passed")
