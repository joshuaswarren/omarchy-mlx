#!/usr/bin/env python3
"""PT-X: sub_ (in-place, alias-propagating) and rsub (scalar-reflected) parity.

sub_/rsub are handled by the compiled executor's in-place/state blocks
(the add_/mul_ pattern), so those tests build with make_compiled_executor.
The eager _FUNC_SPECS rsub entry is tested directly.

Runs under plain python3 (no pytest needed) or pytest.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _np(a):
    return np.array(a)


def _compiled_run(nodes_fn, ins, dtype=mx.float32):
    g = torch.fx.Graph()
    phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
    out = nodes_fn(g, phs)
    g.output(out)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run = mlx_vulkan.make_compiled_executor(gm, dt=mx.float32)
    return run(*ins)


def test_sub_inplace_alias():
    # sub_ mutates the operand node's value and every same-storage alias of
    # the state placeholder; the graph re-reads the operand after the sub_,
    # so both cat halves must show the subtracted value.
    z = torch.tensor([5.0, 6.0, 7.0])
    s = torch.tensor([1.5, 2.5, 3.5])

    def nodes(g, phs):
        sub = g.call_function(torch.ops.aten.sub_.Tensor, (phs[0], phs[1]), {})
        return g.call_function(torch.ops.aten.cat.default, ([sub, phs[0]], 0), {})

    out = _compiled_run(nodes, [z, s])
    o = out[0] if isinstance(out, (tuple, list)) else out
    want = torch.cat([z - s, z - s], 0)
    assert np.allclose(_np(o), want.numpy(), atol=1e-5), (_np(o), want.numpy())
    print("test_sub_inplace_alias PASS")


def test_sub_inplace_value():
    # sub_ returns the subtracted sequence
    z = torch.tensor([5.0, 6.0, 7.0])
    s = torch.tensor([1.0, 2.0, 3.0])

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.sub_.Tensor, (phs[0], phs[1]), {})

    out = _compiled_run(nodes, [z, s])
    o = out[0] if isinstance(out, (tuple, list)) else out
    want = z - s
    assert np.allclose(_np(o), want.numpy(), atol=1e-5)
    print("test_sub_inplace_value PASS")


def test_sub_inplace_alpha_kwarg():
    # aten.sub_.Tensor(self, other, *, alpha): self -= alpha * other.
    # alpha is keyword-only on this overload; the compiled executor must
    # read it from kwargs, not silently drop it.
    z = torch.tensor([1.0, 2.0])
    s = torch.tensor([2.0, 3.0])

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.sub_.Tensor,
                               (phs[0], phs[1]), {"alpha": 2})

    out = _compiled_run(nodes, [z, s])
    o = out[0] if isinstance(out, (tuple, list)) else out
    want = z - 2 * s
    assert np.allclose(_np(o), want.numpy(), atol=1e-5), (_np(o), want.numpy())
    print("test_sub_inplace_alpha_kwarg PASS")


def test_sub_inplace_rejects_broadcast_growth():
    # in-place semantics: the result must fit the destination shape.
    # (2,) -= (2,2) is a torch error, not a silent writeback of (2,2).
    z = torch.tensor([1.0, 2.0])
    s = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.sub_.Tensor, (phs[0], phs[1]), {})

    try:
        _compiled_run(nodes, [z, s])
    except RuntimeError:
        print("test_sub_inplace_rejects_broadcast_growth PASS")
        return
    raise AssertionError("sub_ accepted broadcast growth of the destination")


def test_sub_inplace_dtype_narrowing():
    # in-place writeback casts the result to the destination dtype:
    # int8 dst - int64 rhs wraps to int8 like torch ([-254,4] -> [2,4]).
    z = torch.tensor([3, 7], dtype=torch.int8)
    s = torch.tensor([257, 3], dtype=torch.int64)

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.sub_.Tensor, (phs[0], phs[1]), {})

    out = _compiled_run(nodes, [z, s])
    o = out[0] if isinstance(out, (tuple, list)) else out
    want = torch.tensor([3, 7], dtype=torch.int8).sub_(torch.tensor([257, 3]))
    assert np.array_equal(_np(o), want.numpy()), (_np(o), want.numpy())
    assert str(o.dtype) == "torch.int8", o.dtype
    print("test_sub_inplace_dtype_narrowing PASS")


def test_sub_inplace_rejects_float_to_int():
    # int64 dst -= float rhs: torch rejects the float result casting back.
    z = torch.tensor([3, 7], dtype=torch.int64)
    s = torch.tensor([1.5, 2.5], dtype=torch.float32)

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.sub_.Tensor, (phs[0], phs[1]), {})

    try:
        _compiled_run(nodes, [z, s])
    except RuntimeError:
        print("test_sub_inplace_rejects_float_to_int PASS")
        return
    raise AssertionError("sub_ cast a float result into an integral destination")


def test_sub_inplace_rejects_float_alpha_on_int():
    # integral operands + floating alpha is a torch error, not a float writeback.
    z = torch.tensor([3, 7], dtype=torch.int64)
    s = torch.tensor([1, 2], dtype=torch.int64)

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.sub_.Tensor,
                               (phs[0], phs[1]), {"alpha": 0.5})

    try:
        _compiled_run(nodes, [z, s])
    except RuntimeError:
        print("test_sub_inplace_rejects_float_alpha_on_int PASS")
        return
    raise AssertionError("sub_ accepted floating alpha on integral operands")


def test_rsub_alpha_kwarg():
    # rsub.Scalar(self, other, *, alpha=1): other - alpha * self, alpha by kwarg.
    x = torch.tensor([1.0, 2.0, 3.0])

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.rsub.Scalar,
                               (phs[0], 10), {"alpha": 2})

    out = _compiled_run(nodes, [x])
    o = out[0] if isinstance(out, (tuple, list)) else out
    want = 10 - 2 * x
    assert np.allclose(_np(o), want.numpy(), atol=1e-5), (_np(o), want.numpy())
    print("test_rsub_alpha_kwarg PASS")


def test_sub_inplace_rejects_bool_dst():
    # nothing casts into a bool destination; torch rejects bool subtraction
    # and the int->bool writeback alike.
    z = torch.tensor([True, False])
    s = torch.tensor([1, 1], dtype=torch.int64)

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.sub_.Tensor, (phs[0], phs[1]), {})

    try:
        _compiled_run(nodes, [z, s])
    except (RuntimeError, ValueError):
        print("test_sub_inplace_rejects_bool_dst PASS")
        return
    raise AssertionError("sub_ accepted an int result into a bool destination")


def test_mul_inplace_rejects_bool_dst():
    # same writeback contract on mul_: int64 result into bool dst is a
    # torch error, not a silent bool round-trip.
    z = torch.tensor([True, False])
    s = torch.tensor([2, 3], dtype=torch.int64)

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.mul_.Tensor, (phs[0], phs[1]), {})

    try:
        _compiled_run(nodes, [z, s])
    except (RuntimeError, ValueError):
        print("test_mul_inplace_rejects_bool_dst PASS")
        return
    raise AssertionError("mul_ accepted an int result into a bool destination")


def test_rsub_rejects_float_alpha_on_int():
    # integral input + floating alpha is a torch error on the rsub path too.
    x = torch.tensor([1, 2, 3], dtype=torch.int64)

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.rsub.Scalar,
                               (phs[0], 10), {"alpha": 0.5})

    try:
        _compiled_run(nodes, [x])
    except RuntimeError:
        print("test_rsub_rejects_float_alpha_on_int PASS")
        return
    raise AssertionError("rsub accepted floating alpha on integral input")


def test_rsub_float_other_alpha_ok():
    # the alpha check is on the promoted type: int64 self + float other +
    # alpha=0.5 is legal and returns float [9.5, 9.0] like torch.
    x = torch.tensor([1, 2], dtype=torch.int64)

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.rsub.Scalar,
                               (phs[0], 10.0), {"alpha": 0.5})

    out = _compiled_run(nodes, [x])
    o = out[0] if isinstance(out, (tuple, list)) else out
    want = torch.tensor([9.5, 9.0])
    assert np.allclose(_np(o), want.numpy(), atol=1e-5), (_np(o), want.numpy())
    print("test_rsub_float_other_alpha_ok PASS")


def test_rsub_eager_rejects_float_alpha_on_int():
    # same rule on the per-node eager spec: integral operands + floating
    # alpha defers to torch, which raises.
    mxa = mlx_vulkan._mx(torch.tensor([1, 2], dtype=torch.int64))
    try:
        mlx_vulkan._FUNC_SPECS["rsub"](mxa, 10, 0.5)
    except RuntimeError:
        print("test_rsub_eager_rejects_float_alpha_on_int PASS")
        return
    raise AssertionError("eager rsub accepted floating alpha on integral input")


def test_rsub_scalar_reflected():
    # 3 - x lowers through aten.rsub.Scalar(x, 3): the scalar is reflected
    # to the left of the subtraction
    x = torch.tensor([1.0, 2.0, 3.0])

    def nodes(g, phs):
        r = g.call_function(torch.ops.aten.rsub.Scalar, (phs[0], 3), {})
        return g.call_function(torch.ops.aten.mul.default, (r, phs[0]), {})

    out = _compiled_run(nodes, [x])
    o = out[0] if isinstance(out, (tuple, list)) else out
    want = (3 - x) * x
    assert np.allclose(_np(o), want.numpy(), atol=1e-5), (_np(o), want.numpy())
    print("test_rsub_scalar_reflected PASS")


def test_rsub_alpha():
    # aten.rsub.Scalar(self, other, alpha): other - alpha * self
    x = torch.tensor([1.0, 2.0, 3.0])

    def nodes(g, phs):
        return g.call_function(torch.ops.aten.rsub.Scalar, (phs[0], 10, 2.0), {})

    out = _compiled_run(nodes, [x])
    o = out[0] if isinstance(out, (tuple, list)) else out
    want = 10 - 2.0 * x
    assert np.allclose(_np(o), want.numpy(), atol=1e-5)
    print("test_rsub_alpha PASS")


def test_rsub_eager_spec():
    # the eager _FUNC_SPECS entry: reverse subtract with alpha
    mxa = mlx_vulkan._mx(torch.tensor([1.0, 2.0]))
    mxb = mlx_vulkan._mx(torch.tensor([10.0, 20.0]))
    out = mlx_vulkan._FUNC_SPECS["rsub"](mxa, mxb, 2)
    want = (10.0 - 2 * 1.0, 20.0 - 2 * 2.0)
    assert np.allclose(_np(out), want, atol=1e-5), (_np(out), want)
    # default alpha 1
    out1 = mlx_vulkan._FUNC_SPECS["rsub"](mxa, mxb)
    want1 = (10.0 - 1.0, 20.0 - 2.0)
    assert np.allclose(_np(out1), want1, atol=1e-5)
    print("test_rsub_eager_spec PASS")


if __name__ == "__main__":
    test_sub_inplace_alias()
    test_sub_inplace_value()
    test_sub_inplace_alpha_kwarg()
    test_sub_inplace_rejects_broadcast_growth()
    test_sub_inplace_dtype_narrowing()
    test_sub_inplace_rejects_float_to_int()
    test_sub_inplace_rejects_float_alpha_on_int()
    test_sub_inplace_rejects_bool_dst()
    test_mul_inplace_rejects_bool_dst()
    test_rsub_alpha_kwarg()
    test_rsub_rejects_float_alpha_on_int()
    test_rsub_float_other_alpha_ok()
    test_rsub_eager_rejects_float_alpha_on_int()
    test_rsub_scalar_reflected()
    test_rsub_alpha()
    test_rsub_eager_spec()
