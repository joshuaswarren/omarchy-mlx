#!/usr/bin/env python3
"""Q-59: aten.unbind / aten.narrow / the safe as_strided subset parity.

unbind lowers per-index select (each output is a contiguous slice of a
contiguous input); narrow lowers as the contiguous dim-0 slice; as_strided
lowers the non-overlapping subset as a materialized gather and defers
overlapping strides to the CPU path by design (the happy path never
falls back).

fp32 compares exactly (index arithmetic carries no rounding); bf16
compares upcast at the bf16 mantissa tolerance.

Runs under plain python3 (no pytest needed) or pytest.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _compiled(target, ins, args):
    g = torch.fx.Graph()
    phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
    n = g.call_function(target, tuple(phs) + tuple(args), {})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    return run, stats


def test_unbind_eager_and_compiled():
    x = torch.arange(24).reshape(2, 3, 4)
    ref = torch.unbind(x, dim=1)
    # eager spec
    parts = mlx_vulkan._FUNC_SPECS["unbind"](mlx_vulkan._mx(x), 1)
    assert isinstance(parts, tuple) and len(parts) == 3
    for i, part in enumerate(parts):
        assert np.array_equal(np.array(part), ref[i].numpy()), (i, part, ref[i])
    # compiled: dynamo-style graphs carry unbind; the executor lowers it
    # per-index without CPU fallback on a contiguous input
    run, stats = _compiled(torch.ops.aten.unbind, (x,), (1,))
    out = run(x)
    assert stats["cpu"] == [], stats
    for i, piece in enumerate(out):
        assert np.array_equal(piece.numpy(), ref[i].numpy()), (i, piece)
    print("test_unbind PASS")


def test_narrow_dim0_and_defer():
    x = torch.arange(24).reshape(2, 3, 4)
    ref = torch.narrow(x, 0, 1, 1)
    run, stats = _compiled(torch.ops.aten.narrow.default, (x,), (0, 1, 1))
    out = run(x)
    assert stats["cpu"] == [], stats
    assert np.array_equal(out.numpy(), ref.numpy())
    # a middle-dim narrow carries strided semantics: the plan pass flags
    # it, the table's _slice lowers it as a correct materialized copy (the
    # repo's settled behavior), and the values match torch
    run2, stats2 = _compiled(torch.ops.aten.narrow.default, (x,), (1, 0, 2))
    out2 = run2(x)
    assert np.array_equal(out2.numpy(), torch.narrow(x, 1, 0, 2).numpy())
    print("test_narrow PASS")


def test_as_strided_safe_subset():
    # contiguous re-stride with an offset; each base holds every address
    # the view names (torch refuses views past the storage end)
    cases = [
        (torch.arange(24), (2, 3, 4), (12, 4, 1), 0),
        (torch.arange(32), (2, 3, 4), (12, 4, 1), 6),
        (torch.arange(26), (4, 6), (6, 1), 2),
        # gapped but non-overlapping: stride 8 skips elements 3..7 of row 0
        (torch.arange(24), (3, 3), (8, 2), 1),
        # a size-1 dimension with a wild stride is still non-overlapping
        (torch.arange(24), (1, 5), (999, 3), 4),
    ]
    for base, size, stride, offset in cases:
        ref = torch.as_strided(base, size, stride, offset)
        run, stats = _compiled(torch.ops.aten.as_strided.default, (base,), (size, stride, offset))
        out = run(base)
        assert stats["cpu"] == [], (size, stride, offset, stats)
        assert np.array_equal(out.numpy(), ref.numpy()), (size, stride, offset, out, ref)
    # bf16 keeps its dtype through the eager spec
    xb = torch.arange(24).bfloat16()
    low = mlx_vulkan._FUNC_SPECS["as_strided"](mlx_vulkan._mx(xb), (2, 3, 4), (12, 4, 1), 0)
    assert low.dtype == mx.bfloat16, low.dtype
    ref = torch.as_strided(xb, (2, 3, 4), (12, 4, 1), 0)
    assert np.array_equal(np.array(low.astype(mx.float32)), ref.float().numpy())
    print("test_as_strided_safe PASS")


def test_as_strided_overlap_defers():
    x = torch.arange(24)
    # overlapping strides: two index tuples name one offset, so the plan
    # pass flags the node and the CPU fallback carries it (by design)
    run, stats = _compiled(torch.ops.aten.as_strided.default, (x,), ((2, 2), (1, 1), 0))
    out = run(x)
    assert stats["cpu"], "an overlapping as_strided must defer to CPU"
    assert np.array_equal(out.numpy(), torch.as_strided(x, (2, 2), (1, 1), 0).numpy())
    # zero-stride broadcast overlaps the same way
    run2, stats2 = _compiled(torch.ops.aten.as_strided.default, (x,), ((4, 4), (0, 1), 0))
    out2 = run2(x)
    assert stats2["cpu"], "a zero-stride broadcast must defer to CPU"
    # the eager spec refuses loudly if it is ever reached directly
    try:
        mlx_vulkan._FUNC_SPECS["as_strided"](mlx_vulkan._mx(x), (2, 2), (1, 1), 0)
        raise AssertionError("the overlapping spec must raise")
    except ValueError:
        pass
    print("test_as_strided_overlap_defer PASS")


if __name__ == "__main__":
    test_unbind_eager_and_compiled()
    test_narrow_dim0_and_defer()
    test_as_strided_safe_subset()
    test_as_strided_overlap_defers()
    print("Q-59 ALL PASS")
