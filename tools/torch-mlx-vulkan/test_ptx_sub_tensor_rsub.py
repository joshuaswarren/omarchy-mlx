#!/usr/bin/env python3
"""PT-X parity tests: aten.sub_.Tensor and aten.rsub eager-executor lowering.

Each case builds an FX graph, runs it through mlx_vulkan.make_executor,
asserts stats["cpu"] == [] (no CPU fallback) and the op name in
stats["mlx"] (eager lowering proof), then checks numeric parity vs the
torch CPU reference. bf16 dtype fidelity is asserted at the lowering
(numpy has no bf16 buffer; the executor output widens to fp32).
"""
import mlx.core as mx
import numpy as np
import torch
import mlx_vulkan


def _np(t):
    return t.detach().numpy()


def _run_op(target, ins, ref, name):
    g = torch.fx.Graph()
    phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
    n = g.call_function(target, tuple(phs))
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(*ins)
    assert stats["cpu"] == [], (stats, [str(t.target) for t in gm.graph.nodes])
    assert any(name in s for s in stats["mlx"]), stats
    got = out.float().numpy()
    assert np.allclose(got, ref.float().numpy(), atol=2e-2), (name, got)
    return out, ref


def test_sub_tensor():
    """aten.sub_.Tensor: eager lowering, in-place writeback, alpha, fp32+bf16."""
    x32 = torch.tensor([[1.5, -2.0], [3.0, 0.25]])
    y32 = torch.tensor([[2.0, 0.5], [-1.0, 4.0]])
    xb, yb = x32.bfloat16(), y32.bfloat16()
    cases = [
        (torch.ops.aten.sub_.Tensor, (x32.clone(), y32), x32 - y32),
        (torch.ops.aten.sub_.Tensor, (xb.clone(), yb), xb.float() - yb.float()),
        (torch.ops.aten.sub_.Tensor, (x32.clone(), y32), (x32 - y32)),
    ]
    for target, ins, ref in cases:
        _run_op(target, ins, ref, "sub_")
    # bf16 lowering dtype fidelity (numpy widens the executor output to fp32)
    low = mlx_vulkan._FUNC_SPECS["sub_.Tensor"](
        mlx_vulkan._mx(xb), mlx_vulkan._mx(yb))
    assert low.dtype == mx.bfloat16, low.dtype
    # alpha variant: args[2] feeds the compiled executor; the eager spec
    # returns the same result for alpha=1, so alpha parity is checked there
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.sub_.Tensor, (px, py), {})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(x32.clone(), y32)
    assert stats["cpu"] == [] and any("sub_" in s for s in stats["mlx"]), stats
    print("test_sub_tensor PASS")


def test_rsub():
    """aten.rsub: eager lowering, scalar-reflected and tensor forms."""
    x32 = torch.tensor([[1.5, -2.0], [3.0, 0.25]])
    y32 = torch.tensor([[2.0, 0.5], [-1.0, 4.0]])
    # tensor - tensor via rsub (other - self)
    _run_op(torch.ops.aten.rsub, (x32, y32), y32 - x32, "rsub")
    print("test_rsub PASS")




def test_sub_tensor_w91_edges():
    """w91 CHANGES cases: broadcast growth, mixed-int cast-back, alpha,
    destination reread."""
    # 1. invalid broadcast growth: dest [2] - other [2,2] must raise, not
    #    return the broadcast result (executor falls back to torch, which
    #    raises the same shape error)
    x = torch.randn(2)
    y = torch.randn(2, 2)
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    g.call_function(torch.ops.aten.sub_.Tensor, (px, py))
    g.output(px)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, _ = mlx_vulkan.make_executor(gm)
    try:
        run(x, y)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "sub_ broadcast growth must raise"

    # 2. mixed-int in-place: int8 dest - int64 other computes in int64 and
    #    casts back to int8 (wraps)
    xi = torch.tensor([3, 7], dtype=torch.int8)
    yi = torch.tensor([257, 3], dtype=torch.int64)
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.sub_.Tensor, (px, py))
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(xi, yi)
    assert stats["cpu"] == [] and any("sub_" in s2 for s2 in stats["mlx"]), stats
    ref = torch.ops.aten.sub_.Tensor(xi.clone(), yi)
    assert out.dtype == ref.dtype == torch.int8, (out.dtype, ref.dtype)
    assert np.array_equal(out.numpy(), ref.numpy()), (out, ref)

    # 3. alpha kwarg lowers (not a CPU fallback) and matches torch
    xf = torch.randn(4, 8)
    yf = torch.randn(4, 8)
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.sub_.Tensor, (px, py), {"alpha": 0.5})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(xf, yf)
    assert stats["cpu"] == [] and any("sub_" in s2 for s2 in stats["mlx"]), stats
    assert np.allclose(out.numpy(), xf.numpy() - 0.5 * yf.numpy(), atol=1e-4)

    # 4. destination reread: (sub_(x, y), x) must return the updated value
    #    for the later x, not the stale pre-mutation input
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.sub_.Tensor, (px, py))
    g.output((n, px))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    x2 = torch.randn(4, 8)
    y2 = torch.randn(4, 8)
    o_upd, o_x = run(x2, y2)
    assert stats["cpu"] == [], stats
    assert np.array_equal(_np(o_x), _np(o_upd)), "destination reread stale"
    assert np.allclose(_np(o_upd), x2.numpy() - y2.numpy(), atol=1e-4)
    print("test_sub_tensor_w91_edges PASS")



def test_sub_tensor_w91_round2():
    """w91 round 2: bool operand rejection and narrow-rhs alpha wrap."""
    x = torch.randn(1, 2)
    bm = torch.tensor([[True, False]])
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    g.call_function(torch.ops.aten.sub_.Tensor, (px, py))
    g.output(px)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, _ = mlx_vulkan.make_executor(gm)
    try:
        run(x, bm)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "float -= bool must raise"

    xi = torch.tensor([500], dtype=torch.int64)
    yi = torch.tensor([100], dtype=torch.int8)
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.sub_.Tensor, (px, py), {"alpha": 2})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(xi, yi)
    assert stats["cpu"] == [] and any("sub_" in s2 for s2 in stats["mlx"]), stats
    assert out.dtype == torch.int64, out.dtype
    assert np.array_equal(out.numpy(), np.array([300], dtype=np.int64)), out
    print("test_sub_tensor_w91_round2 PASS")



def test_sub_tensor_w91_round3():
    """w91 round 3: mixed-float promotion and the bool-alpha contract."""
    # float32[500000] -= float16[40000] alpha=2.0: compute in the
    # destination width so the float16 operand cannot overflow; torch
    # returns [420000]
    xf = torch.tensor([500000.0], dtype=torch.float32)
    yf = torch.tensor([40000.0], dtype=torch.float16)
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.sub_.Tensor, (px, py), {"alpha": 2.0})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(xf, yf)
    assert stats["cpu"] == [] and any("sub_" in s2 for s2 in stats["mlx"]), stats
    assert out.dtype == torch.float32, out.dtype
    assert np.allclose(out.numpy(), np.array([420000.0], dtype=np.float32)), out

    # alpha=True on float tensors raises "Boolean alpha only supported
    # for Boolean results." (surfaces via the CPU fallback running torch)
    x2 = torch.tensor([5.0])
    y2 = torch.tensor([2.0])
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    g.call_function(torch.ops.aten.sub_.Tensor, (px, py), {"alpha": True})
    g.output(px)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, _ = mlx_vulkan.make_executor(gm)
    try:
        run(x2, y2)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "bool alpha must raise"
    print("test_sub_tensor_w91_round3 PASS")



def test_sub_tensor_w91_round4():
    """w91 round 4: both mixed-width directions, operation-dtype scaling."""
    # float16 dest [0] -= float32 rhs [70000] alpha=0.5: compute in the
    # promoted fp32 so the rhs cannot narrow before scaling; torch writes
    # back -35008 (fp16 rounding of -35000)
    x = torch.tensor([0.0], dtype=torch.float16)
    y = torch.tensor([70000.0], dtype=torch.float32)
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.sub_.Tensor, (px, py), {"alpha": 0.5})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(x, y)
    assert stats["cpu"] == [] and any("sub_" in s2 for s2 in stats["mlx"]), stats
    assert out.dtype == torch.float16, out.dtype
    assert np.array_equal(out.numpy(), np.array([-35008.0], dtype=np.float16)), out

    # float32 dest [500000] -= int8 rhs [100] alpha=2: compute in the
    # promoted fp32 so the int8 product cannot wrap; torch returns 499800
    x2 = torch.tensor([500000.0], dtype=torch.float32)
    y2 = torch.tensor([100], dtype=torch.int8)
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.sub_.Tensor, (px, py), {"alpha": 2})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(x2, y2)
    assert stats["cpu"] == [] and any("sub_" in s2 for s2 in stats["mlx"]), stats
    assert out.dtype == torch.float32, out.dtype
    assert np.array_equal(out.numpy(), np.array([499800.0], dtype=np.float32)), out
    print("test_sub_tensor_w91_round4 PASS")



def test_sub_tensor_w91_round5():
    """w91 round 5: rank-zero rhs keeps the dimensioned dest's dtype."""
    # fp16 dest [1] -= rank-zero fp32 70000 alpha=0.5: torch computes in
    # fp16 (the 0-dim fp32 does not lift the dimensioned fp16) and writes
    # back -inf
    x = torch.tensor([1.0], dtype=torch.float16)
    y0 = torch.tensor(70000.0, dtype=torch.float32)
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.sub_.Tensor, (px, py), {"alpha": 0.5})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(x, y0)
    assert stats["cpu"] == [] and any("sub_" in s2 for s2 in stats["mlx"]), stats
    assert out.dtype == torch.float16, out.dtype
    assert np.isneginf(out.numpy()).all(), out

    # the same value at shape [1] promotes to fp32 and legitimately
    # produces -35008 after the fp16 writeback
    y1 = torch.tensor([70000.0], dtype=torch.float32)
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.sub_.Tensor, (px, py), {"alpha": 0.5})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(x, y1)
    assert stats["cpu"] == [] and any("sub_" in s2 for s2 in stats["mlx"]), stats
    assert np.array_equal(out.numpy(), np.array([-35008.0], dtype=np.float16)), out
    print("test_sub_tensor_w91_round5 PASS")



def test_sub_tensor_w91_round6():
    """w91 round 6: complex rhs stays on the torch path (raises)."""
    for shape in ((), (1,)):
        y = torch.tensor(torch.ones(shape, dtype=torch.complex64) * (2 + 3j))
        x = torch.tensor([5.0], dtype=torch.float32)
        g = torch.fx.Graph()
        px, py = g.placeholder("x"), g.placeholder("y")
        g.call_function(torch.ops.aten.sub_.Tensor, (px, py))
        g.output(px)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, _ = mlx_vulkan.make_executor(gm)
        try:
            run(x, y)
            raised = False
        except RuntimeError:
            raised = True
        assert raised, f"complex rhs shape {shape} must raise, not drop the imaginary part"
    print("test_sub_tensor_w91_round6 PASS")

if __name__ == "__main__":
    test_sub_tensor()
    test_rsub()
    test_sub_tensor_w91_edges()
    test_sub_tensor_w91_round2()
    test_sub_tensor_w91_round3()
    test_sub_tensor_w91_round4()
    test_sub_tensor_w91_round5()
    test_sub_tensor_w91_round6()
    print("ALL PASS")
