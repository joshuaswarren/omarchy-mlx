#!/usr/bin/env python3
"""Parity tests for the aten.view/aten.reshape lowerings in mlx_vulkan.

view lowers as mx reshape aliasing; reshape lowers as view-or-constexpr.
Only views on possibly non-contiguous inputs are refused to the CPU path
(mx.reshape would silently copy, breaking view's storage-alias contract and
torch's stride/error semantics).

Parity contract per node: output shape equality, stride equality, and a
sha256 digest of the float32 payload versus CPU torch eager.

Runs under plain python3 (no pytest needed) or pytest.
"""
import os, sys, hashlib, operator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _dg(v):
    """sha256[:16] of the float32 payload, mx array or torch tensor."""
    if isinstance(v, mx.array):
        a = np.array(v)
    elif isinstance(v, torch.Tensor):
        a = v.detach().cpu().numpy()
    else:
        a = np.asarray(v)
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float32).tobytes()).hexdigest()[:16]


def _gm(*ops):
    """One placeholder plus a chain of (kind, target, extra-args) nodes."""
    g = torch.fx.Graph()
    ph = g.placeholder("x")
    prev = ph
    for kind, target, rest in ops:
        prev = getattr(g, kind)(target, (prev,) + tuple(rest))
    g.output(prev)
    return torch.fx.GraphModule(torch.nn.Module(), g)


def _run(gm, x):
    run, stats = mlx_vulkan.make_executor(gm)
    return run(mx.array(x.numpy())), stats


def _run_exec(gm):
    return mlx_vulkan.make_executor(gm)


def test_view_contiguous_lowers():
    x = torch.randn(4, 8)
    for shp in ((2, 16), (8, 4), (32,), (-1,), (1, 32)):
        for op in (torch.ops.aten.view.default, torch.ops.aten._unsafe_view.default):
            gm = _gm(("call_function", op, (list(shp),)))
            out, stats = _run(gm, x)
            exp = x.view(*shp)
            assert tuple(out.shape) == tuple(exp.shape), (shp, out.shape, exp.shape)
            assert out.stride() == exp.stride(), (shp, out.stride(), exp.stride())
            assert _dg(out) == _dg(exp), shp
            assert "view" in stats["mlx"] or "_unsafe_view" in stats["mlx"], stats
            assert "view" not in stats["cpu"], stats
    print("test_view_contiguous_lowers PASS")


def test_view_strided_refuses():
    # Each producer may yield a non-contiguous torch tensor, so the
    # trailing aten.view is refused to the CPU path (mx.reshape would
    # silently copy, breaking view's alias/error semantics). Three
    # outcomes on the refused path, pinned per case below:
    # "raise": eager's view errors and the mx->torch boundary preserves the
    #   layout, so the same RuntimeError surfaces
    # "legal": eager's view runs; the refused path must too, matching
    #   shape + digest on the torch fallback
    # "copy": eager's view errors but the boundary materializes a
    #   contiguous tensor, so the fallback returns the reshape-equivalent
    #   values; in every case "view" must never reach stats["mlx"]
    cases = [
        ("raise", ("call_function", torch.ops.aten.transpose.int, (0, 1)),
         torch.randn(4, 8), lambda x: x.t().reshape(-1)),
        ("raise", ("call_function", torch.ops.aten.t.default, ()),
         torch.randn(4, 8), lambda x: x.t().reshape(-1)),
        ("raise", ("call_function", torch.ops.aten.permute.default, ([1, 0],)),
         torch.randn(4, 8), lambda x: x.permute(1, 0).reshape(-1)),
        ("legal", ("call_function", torch.ops.aten.select.int, (1, 0)),
         torch.randn(4, 8), lambda x: x.select(1, 0).view(-1)),
        ("legal", ("call_function", operator.getitem, ((slice(None), 0),)),
         torch.randn(4, 8), lambda x: x[:, 0].view(-1)),
        ("copy", ("call_function", torch.ops.aten.slice.Tensor, (1, None, None, 2)),
         torch.randn(4, 8), lambda x: x[:, ::2].reshape(-1)),
        ("copy", ("call_function", torch.ops.aten.expand.default, ([4, 8],)),
         torch.randn(4, 1), lambda x: x.expand(4, 8).reshape(-1)),
    ]
    for cls, prod, x, expf in cases:
        gm = _gm(prod, ("call_function", torch.ops.aten.view.default, ([-1],)))
        run, stats = mlx_vulkan.make_executor(gm)
        if cls == "raise":
            try:
                run(mx.array(x.numpy()))
            except RuntimeError:
                pass
            else:
                raise AssertionError((prod[1], "view did not refuse"))
        else:
            out = run(mx.array(x.numpy()))
            exp = expf(x)
            assert tuple(out.shape) == tuple(exp.shape), (prod[1], out.shape, exp.shape)
            assert _dg(out) == _dg(exp), prod[1]
            assert "view" in stats["cpu"], (prod[1], stats)
        assert "view" not in stats["mlx"], (prod[1], stats)
    print("test_view_strided_refuses PASS")


def test_reshape_always_lowers():
    x = torch.randn(4, 8)
    for shp in ((2, 16), (8, 4), (-1,)):
        gm = _gm(("call_function", torch.ops.aten.reshape.default, (list(shp),)))
        out, stats = _run(gm, x)
        exp = x.reshape(*shp)
        assert tuple(out.shape) == tuple(exp.shape)
        assert out.stride() == exp.stride()
        assert _dg(out) == _dg(exp)
        assert "reshape" in stats["mlx"], stats
    # strided input: torch reshape copies -> contiguous out; mx.reshape
    # matches value and stride semantics, so it stays lowered.
    gm = _gm(("call_function", torch.ops.aten.transpose.int, (0, 1)),
             ("call_function", torch.ops.aten.reshape.default, ([-1],)))
    out, stats = _run(gm, x)
    exp = x.t().reshape(-1)
    assert tuple(out.shape) == tuple(exp.shape)
    assert out.stride() == exp.stride()
    assert _dg(out) == _dg(exp)
    assert "reshape" in stats["mlx"], stats
    assert "reshape" not in stats["cpu"], stats
    print("test_reshape_always_lowers PASS")


def test_view_dim0_carveouts_lower():
    x = torch.randn(4, 8)
    cases = [
        # select/narrow/slice on dim 0 keep contiguity: the view still lowers
        (("call_function", torch.ops.aten.select.int, (0, 1)), (1, 8), x.select(0, 1).view(1, 8)),
        (("call_function", torch.ops.aten.narrow.default, (0, 1, 2)), (-1,), x.narrow(0, 1, 2).view(-1)),
        (("call_function", torch.ops.aten.slice.Tensor, (0, 1, 3, 1)), (2, 8), x[1:3].view(2, 8)),
    ]
    for prod, shp, exp in cases:
        gm = _gm(prod, ("call_function", torch.ops.aten.view.default, (list(shp),)))
        out, stats = _run(gm, x)
        assert "view" in stats["mlx"], (prod[1], stats)
        assert "view" not in stats["cpu"], (prod[1], stats)
        assert _dg(out) == _dg(exp), prod[1]
    # detach passthrough keeps the strided flag: still refused, and the
    # same RuntimeError eager raises must surface
    gm = _gm(("call_function", torch.ops.aten.transpose.int, (0, 1)),
             ("call_function", torch.ops.aten.detach.default, ()),
             ("call_function", torch.ops.aten.view.default, ([-1],)))
    run, stats = _run_exec(gm)
    try:
        run(mx.array(x.numpy()))
    except RuntimeError:
        pass
    else:
        raise AssertionError("strided view through detach silently lowered")
    print("test_view_dim0_carveouts_lower PASS")


def test_call_method_view():
    x = torch.randn(4, 8)
    gm = _gm(("call_method", "view", ([2, 16],)))
    out, stats = _run(gm, x)
    assert _dg(out) == _dg(x.view(2, 16))
    assert "view" in stats["mlx"], stats
    gm = _gm(("call_method", "reshape", ([2, 16],)))
    out, stats = _run(gm, x)
    assert _dg(out) == _dg(x.reshape(2, 16))
    assert "reshape" in stats["mlx"], stats
    # call_method view on a strided base refuses the same way: the eager
    # RuntimeError surfaces instead of a silent mx copy
    gm = _gm(("call_function", torch.ops.aten.transpose.int, (0, 1)),
             ("call_method", "view", ([-1],)))
    run, stats = _run_exec(gm)
    try:
        run(mx.array(x.numpy()))
    except RuntimeError:
        pass
    else:
        raise AssertionError("strided call_method view silently lowered")
    print("test_call_method_view PASS")


if __name__ == "__main__":
    test_view_contiguous_lowers()
    test_view_strided_refuses()
    test_reshape_always_lowers()
    test_view_dim0_carveouts_lower()
    test_call_method_view()
    print("ALL PASS")
