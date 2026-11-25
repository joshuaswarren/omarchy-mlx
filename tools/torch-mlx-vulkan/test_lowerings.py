#!/usr/bin/env python3
"""Parity tests for the topk/multinomial lowerings in mlx_vulkan.

Runs under plain python3 (no pytest needed) or pytest.
Requires torch + mlx on the path; run from this directory or set
PYTHONPATH to it.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def _np(a):
    return np.array(a)


def test_topk():
    rng = np.random.RandomState(0)
    x = rng.randn(4, 7).astype(np.float32)
    for k, dim in ((1, 0), (3, 0), (1, 1), (3, 1), (7, 1), (2, -1), (3, None)):
        vals, idx = mlx_vulkan._topk(mx.array(x), k, dim)
        tv, ti = torch.topk(torch.tensor(x), k, -1 if dim is None else dim)
        assert np.allclose(_np(vals), tv.numpy()), (k, dim)
        assert np.array_equal(_np(idx), ti.numpy()), (k, dim)
    # largest=False
    vals, idx = mlx_vulkan._topk(mx.array(x), 3, 1, largest=False)
    tv, ti = torch.topk(torch.tensor(x), 3, 1, largest=False)
    assert np.allclose(_np(vals), tv.numpy())
    assert np.array_equal(_np(idx), ti.numpy())
    # 3-D, middle dim
    x3 = rng.randn(2, 5, 3).astype(np.float32)
    vals, idx = mlx_vulkan._topk(mx.array(x3), 2, 1)
    tv, ti = torch.topk(torch.tensor(x3), 2, 1)
    assert np.allclose(_np(vals), tv.numpy())
    assert np.array_equal(_np(idx), ti.numpy())
    print("test_topk PASS")


def test_multinomial():
    rng = np.random.RandomState(0)
    # peaked dist: sample must be the argmax row-wise (deterministic check)
    probs = np.full((8, 16), 1e-4, np.float32)
    peak = np.array([3, 9, 0, 15, 7, 1, 12, 5])
    probs[np.arange(8), peak] = 0.99
    out = mlx_vulkan._multinomial(mx.array(probs), 1)
    assert np.array_equal(_np(out).reshape(-1), peak), _np(out)
    assert _np(out).dtype == np.int64
    # shape/dtype: n>1 without replacement gives n distinct in-range draws
    probs2 = rng.rand(4, 32).astype(np.float32)
    out2 = np.array(mlx_vulkan._multinomial(mx.array(probs2), 5))
    assert out2.shape == (4, 5) and out2.dtype == np.int64
    assert out2.min() >= 0 and out2.max() < 32
    for row in out2:
        assert len(set(row.tolist())) == 5  # without replacement: distinct
    # replacement=True: shape only (draws may repeat)
    out3 = np.array(mlx_vulkan._multinomial(mx.array(probs2), 5, replacement=True))
    assert out3.shape == (4, 5) and out3.dtype == np.int64
    print("test_multinomial PASS")


def test_convolution_backward():
    rng = np.random.RandomState(0)
    for (n, ci, co, k, s, p, d) in ((2, 3, 8, 3, 1, 1, 1), (1, 4, 4, 2, 2, 0, 1), (2, 3, 5, 3, 1, 2, 2)):
        xn = rng.randn(n, ci, 9, 9).astype(np.float32)
        wn = rng.randn(co, ci, k, k).astype(np.float32) * 0.1
        dyn = rng.randn(n, co, (9 + 2 * p - d * (k - 1) - 1) // s + 1, (9 + 2 * p - d * (k - 1) - 1) // s + 1).astype(np.float32)
        mask = [True, True, True]
        gx, gw, gb = mlx_vulkan._convolution_backward(
            mx.array(dyn), mx.array(xn), mx.array(wn), [co],
            (s, s), (p, p), (d, d), False, (0, 0), 1, mask)
        tx = torch.tensor(xn, requires_grad=True)
        tw = torch.tensor(wn, requires_grad=True)
        ty = torch.conv2d(tx, tw, stride=s, padding=p, dilation=d)
        ty.backward(torch.tensor(dyn))
        assert np.abs(_np(gx) - tx.grad.numpy()).max() < 2e-4, (s, p, d)
        assert np.abs(_np(gw) - tw.grad.numpy()).max() < 2e-4, (s, p, d)
        tgb = torch.tensor(dyn).sum(dim=(0, 2, 3))
        assert np.abs(_np(gb) - tgb.numpy()).max() < 2e-4, (s, p, d)
    # mask: only grad_input requested
    xn = rng.randn(1, 2, 5, 5).astype(np.float32)
    wn = rng.randn(3, 2, 3, 3).astype(np.float32)
    dyn = rng.randn(1, 3, 3, 3).astype(np.float32)
    gx, gw, gb = mlx_vulkan._convolution_backward(
        mx.array(dyn), mx.array(xn), mx.array(wn), [3],
        (1, 1), (0, 0), (1, 1), False, (0, 0), 1, [True, False, False])
    assert gx is not None and gw is None and gb is None
    print("test_convolution_backward PASS")


def test_threshold_backward():
    rng = np.random.RandomState(1)
    x = rng.randn(4, 6).astype(np.float32)
    dy = rng.randn(4, 6).astype(np.float32)
    out = mlx_vulkan.ATEN_TABLE["threshold_backward"](mx.array(dy), mx.array(x), 0.0)
    tx, tdy = torch.tensor(x), torch.tensor(dy)
    exp = tdy * (tx > 0)
    assert np.array_equal(_np(out), exp.numpy())
    print("test_threshold_backward PASS")


def test_t():
    x = np.random.RandomState(2).randn(3, 5).astype(np.float32)
    out = mlx_vulkan.ATEN_TABLE["t"](mx.array(x))
    assert np.array_equal(_np(out), torch.tensor(x).t().numpy())
    print("test_t PASS")


def test_embedding_dense_backward():
    rng = np.random.RandomState(7)
    V, D = 9, 4
    for (shape, pad, scale) in (
            ((7,), -1, False), ((2, 5), -1, False), ((2, 5), 3, False),
            ((2, 5), -1, True), ((2, 5), 3, True), ((3, 4, 6), 0, False)):
        go = rng.randn(*shape, D).astype(np.float32)
        idx = rng.randint(0, V, size=shape).astype(np.int64)
        got = mlx_vulkan._embedding_dense_backward(
            mx.array(go), mx.array(idx), V, pad, scale)
        tw = torch.zeros(V, D, requires_grad=True)
        tidx = torch.tensor(idx)
        ty = torch.nn.functional.embedding(
            tidx, tw, padding_idx=pad if pad >= 0 else None,
            scale_grad_by_freq=scale)
        ty.backward(torch.tensor(go))
        assert np.abs(_np(got) - tw.grad.numpy()).max() < 1e-5, (shape, pad, scale)
    # duplicate indices accumulate; unused rows stay zero
    go = rng.randn(5, D).astype(np.float32)
    idx = np.array([4, 4, 4, 0, 8])
    got = _np(mlx_vulkan._embedding_dense_backward(
        mx.array(go), mx.array(idx), V, -1, False))
    exp = np.zeros((V, D), np.float32)
    np.add.at(exp, idx, go)
    assert np.abs(got - exp).max() < 1e-5
    # padding_idx with a NaN/Inf grad row must still produce an exact 0 row
    # (a mask multiply would turn 0*NaN into NaN)
    go = np.full((2, D), np.nan, dtype=np.float32)
    got = _np(mlx_vulkan._embedding_dense_backward(
        mx.array(go), mx.array(np.array([3, 3])), V, 3, False))
    assert np.array_equal(got, np.zeros((V, D), np.float32)), got[3]
    # scale_grad_by_freq counts must accumulate in int32: at 65536 dups an
    # fp16 count overflows to inf and out*1/inf collapses the row to 0
    n_dup = 65536
    go = np.zeros((n_dup, D), np.float16)
    go[0] = 1.0
    got = _np(mlx_vulkan._embedding_dense_backward(
        mx.array(go), mx.array(np.full(n_dup, 5)), V, -1, True))
    exp = 1.0 / n_dup
    assert np.abs(got[5] - exp).max() < 1e-5 and got[5].max() > 0, got[5]
    print("test_embedding_dense_backward PASS")


def test_bn_legit_functional():
    rng = np.random.RandomState(3)
    x = rng.randn(4, 6, 3, 3).astype(np.float32)
    w = rng.rand(6).astype(np.float32) + 0.5
    b = rng.randn(6).astype(np.float32) * 0.1
    rm = rng.randn(6).astype(np.float32)
    rv = rng.rand(6).astype(np.float32) + 0.5
    mom, eps = 0.1, 1e-5
    out, sm, si, nrm, nrv = mlx_vulkan._bn_legit_functional(
        mx.array(x), mx.array(w), mx.array(b), mx.array(rm), mx.array(rv), True, mom, eps)
    tx = torch.tensor(x)
    tw, tb = torch.tensor(w), torch.tensor(b)
    trm, trv = torch.tensor(rm), torch.tensor(rv)
    tout, tsm, tsi = torch.native_batch_norm(tx, tw, tb, trm.clone(), trv.clone(), True, mom, eps)
    # functional variant also returns updated running stats (unbiased var)
    n = x.shape[0] * x.shape[2] * x.shape[3]
    trm2 = trm * (1 - mom) + mom * tx.mean(dim=(0, 2, 3))
    trv2 = trv * (1 - mom) + mom * tx.var(dim=(0, 2, 3), unbiased=False) * (n / (n - 1))
    assert np.abs(_np(out) - tout.numpy()).max() < 2e-5
    assert np.abs(_np(sm) - tsm.numpy()).max() < 2e-5
    assert np.abs(_np(si) - tsi.numpy()).max() < 2e-5
    assert np.abs(_np(nrm) - trm2.numpy()).max() < 2e-5
    assert np.abs(_np(nrv) - trv2.numpy()).max() < 2e-5
    print("test_bn_legit_functional PASS")
def test_arange():
    cases = [
        ((5,), {}, torch.arange(5)),
        ((0, 5), {}, torch.arange(0, 5)),
        ((0, 5, 2), {}, torch.arange(0, 5, 2)),
        ((), {"end": 5}, torch.arange(5)),
        ((), {"start": 0, "end": 5}, torch.arange(0, 5)),
        ((), {"start": 0, "end": 8, "step": 2}, torch.arange(0, 8, 2)),
        ((), {"start": 0, "end": 16, "dtype": torch.float32},
         torch.arange(0, 16, dtype=torch.float32)),
        ((), {"start": 0.5, "end": 3.0, "step": 0.5},
         torch.arange(0.5, 3.0, 0.5)),
        ((0, 5), {"step": 2}, torch.arange(0, 5, 2)),
        ((1.5, 4.0), {}, torch.arange(1.5, 4.0)),
        # dynamo normalizes arange to all-kwargs; step kwarg has no valid
        # torch.arange spelling, so references use the positional form
        ((5,), {"step": 2}, torch.arange(0, 5, 2)),
        ((), {"end": 8, "step": 2}, torch.arange(0, 8, 2)),
        ((), {"start": 3, "end": 9, "step": 2},
         torch.arange(3, 9, 2)),
    ]
    for args, kw, want in cases:
        got = _np(mlx_vulkan._arange(*args, **kw))
        assert got.shape == want.numpy().shape, (args, kw, got.shape)
        assert got.dtype == want.numpy().dtype, (args, kw, got.dtype)
        assert np.allclose(got, want.numpy()), (args, kw)
    print("test_arange PASS")


def test_mul():
    # PT-A: aten.mul.default is the PAR-41 census top fallback (17,208 uses).
    # Prove the eager executor lowers mul/mul.Scalar/mul_ (stats["mlx"]) and
    # matches torch CPU at fp32 and bf16, including mixed-dtype promotion.
    x32 = torch.tensor([[1.5, -2.0], [3.0, 0.25]])
    y32 = torch.tensor([[2.0, 0.5], [-1.0, 4.0]])
    xb, yb = x32.bfloat16(), y32.bfloat16()
    i64 = torch.tensor([[2, 3], [-1, 4]], dtype=torch.int64)
    cases = [
        (torch.ops.aten.mul.default, (x32, y32), torch.mul(x32, y32)),
        (torch.ops.aten.mul.default, (xb, yb), torch.mul(xb, yb)),
        (torch.ops.aten.mul.Scalar, (x32,), torch.mul(x32, 2.5)),
        (torch.ops.aten.mul_.Tensor, (x32.clone(), y32), torch.mul(x32, y32)),
        (torch.ops.aten.mul.default, (x32, i64), torch.mul(x32, i64)),
        (torch.mul, (x32, y32), torch.mul(x32, y32)),
    ]
    for target, ins, ref in cases:
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        args = tuple(phs) if target is not torch.ops.aten.mul.Scalar else (phs[0], 2.5)
        n = g.call_function(target, args, {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, [str(t.target) for t in gm.graph.nodes])
        assert any("mul" in s for s in stats["mlx"]), stats
        got = out.float().numpy()
        assert np.allclose(got, ref.float().numpy(), atol=2e-2), (str(target), got)
        # the executor's _torch output widens bf16 -> fp32 (numpy has no
        # bf16 buffer); dtype fidelity lives at the lowering, so the bf16
        # case also checks the spec result dtype directly
        assert out.dtype == ref.dtype or (
            ref.dtype == torch.bfloat16 and out.dtype == torch.float32), \
            (str(target), out.dtype, ref.dtype)
        if ref.dtype == torch.bfloat16:
            low = mlx_vulkan._FUNC_SPECS["mul"](mlx_vulkan._mx(xb), mlx_vulkan._mx(yb))
            assert low.dtype == mx.bfloat16, low.dtype
    print("test_mul PASS")


def test_scalar_tensor_factory_lowering():
    # BERT emits scalar_tensor factory nodes (scalar arg, no tensor inputs):
    # the eager executor only lowers call_function nodes with an mx input or
    # a _PRODUCERS entry, so scalar_tensor used to hit the torch fallback.
    # symbolic_trace lifts constants to get_attr, so build the fx graph the
    # way dynamo emits it: a call_function on the aten OpOverload.
    g = torch.fx.Graph()
    x = g.placeholder("x")
    st = g.call_function(torch.ops.aten.scalar_tensor.default, (3,),
                         {"dtype": torch.float32})
    add = g.call_function(torch.ops.aten.add.Tensor, (x, st), {})
    g.output(add)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(torch.tensor([1.0, 2.0]))
    assert stats["cpu"] == [], (stats, [str(n.target) for n in gm.graph.nodes])
    assert np.allclose(_np(out), np.array([4.0, 5.0], dtype=np.float32))
    print("test_scalar_tensor_factory_lowering PASS")


def _closure_dict(run, key_is_node):
    for cell in (run.__closure__ or ()):
        v = cell.cell_contents
        if isinstance(v, dict) and all(isinstance(k, torch.fx.Node) for k in v) == key_is_node:
            return v
    return None


def test_pcache_skips_static_weights():
    # G_* placeholders are static weights: conv_static pins the (possibly
    # quantized) array, and pcache must not retain the raw source alongside.
    g = torch.fx.Graph()
    w = g.placeholder("G_w")
    p = g.placeholder("p_bias")
    x = g.placeholder("x")
    mm = g.call_function(torch.ops.aten.mm.default, (x, w), {})
    out = g.call_function(torch.ops.aten.add.Tensor, (mm, p), {})
    g.output(out)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run = mlx_vulkan.make_compiled_executor(gm, dt=mx.float32)
    tw = torch.randn(8, 8); tp = torch.randn(8); tx = torch.randn(4, 8)
    o1 = run(tw, tp, tx)
    o2 = run(tw, tp, tx)  # second call: pcache reuse path
    pcache = _closure_dict(run, True)
    assert pcache is not None
    names = {str(n.target) for n in pcache}
    assert "p_bias" in names and "G_w" not in names, names
    exp = (tx @ tw + tp).numpy()
    assert np.allclose(_np(o2[0]), exp, atol=1e-4)
    print("test_pcache_skips_static_weights PASS")


def test_compiled_executor_nonarray_outputs():
    # aot partitions can emit a non-array element among tensor outputs
    # (glue values); mx.eval must not be called on them.
    g = torch.fx.Graph()
    x = g.placeholder("x")
    y = g.call_function(torch.ops.aten.add.Tensor, (x, x), {})
    g.output((y, 7))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run = mlx_vulkan.make_compiled_executor(gm, dt=mx.float32)
    out = run(torch.tensor([1.0, 2.0]))
    assert np.allclose(_np(out[0]), [2.0, 4.0])
    assert out[1] == 7
    print("test_compiled_executor_nonarray_outputs PASS")


def test_gelu_backward():
    rng = np.random.RandomState(2)
    # fp32: exact formula vs the aten reference, both approximate modes.
    x = rng.randn(6, 8).astype(np.float32) * 3.0  # wide spread to hit both tails
    dy = rng.randn(6, 8).astype(np.float32)
    for approximate in ("none", "tanh"):
        tx = torch.tensor(x, requires_grad=True)
        torch.nn.functional.gelu(tx, approximate=approximate).backward(torch.tensor(dy))
        out = mlx_vulkan.ATEN_TABLE["gelu_backward"](
            mx.array(dy), mx.array(x), approximate=approximate)
        assert np.abs(_np(out) - tx.grad.numpy()).max() < 1e-5, approximate
    # bf16: aten runs the formula in fp32 opmath and rounds to bf16; the
    # lowering must do the same or the bf16 tail values drift.
    xb = torch.tensor(x, dtype=torch.bfloat16)
    gb = torch.tensor(dy, dtype=torch.bfloat16)
    for approximate in ("none", "tanh"):
        ref = torch.ops.aten.gelu_backward.default(gb, xb, approximate=approximate)
        mine = mlx_vulkan.ATEN_TABLE["gelu_backward"](
            mx.array(gb.float().numpy()).astype(mx.bfloat16),
            mx.array(xb.float().numpy()).astype(mx.bfloat16),
            approximate=approximate)
        assert mine.dtype == mx.bfloat16, mine.dtype
        got = torch.from_numpy(np.array(mine.astype(mx.float32)))
        d = float((got - ref.float()).abs().max())
        assert d < 0.02, f"bf16 {approximate}: {d:.3e}"
    # executor path: the aten.gelu_backward.default node must land in
    # stats["mlx"], not the CPU fallback — the census "lowered" status is
    # exactly this fact.
    gfx = torch.fx.Graph()
    ph_g, ph_x = gfx.placeholder("g"), gfx.placeholder("x")
    gfx.output(gfx.call_function(torch.ops.aten.gelu_backward.default, (ph_g, ph_x)))
    gm = torch.fx.GraphModule(torch.nn.Module(), gfx)
    run, stats = mlx_vulkan.make_executor(gm)
    ref = torch.ops.aten.gelu_backward.default(torch.tensor(dy), torch.tensor(x))
    assert np.abs(_np(run(torch.tensor(dy), torch.tensor(x))) - ref.numpy()).max() < 1e-5
    assert "gelu_backward" in stats["mlx"], stats
    assert "gelu_backward" not in stats["cpu"], stats
    print("test_gelu_backward PASS")


def test_linear():
    rng = np.random.RandomState(3)
    x = rng.randn(2, 4, 8).astype(np.float32)   # batched input
    w = rng.randn(6, 8).astype(np.float32)      # (out_features, in_features)
    b = rng.randn(6).astype(np.float32)
    for bias in (b, None):
        exp = torch.nn.functional.linear(torch.tensor(x), torch.tensor(w),
                                         None if bias is None else torch.tensor(bias))
        out = mlx_vulkan._linear(mx.array(x), mx.array(w), None if bias is None else mx.array(bias))
        assert np.abs(_np(out) - exp.numpy()).max() < 1e-5
    # through the eager executor: the aten.linear node must lower, not
    # fall back (compiled executor has its own hardcoded linear branch)
    tx, tw, tb = torch.randn(3, 8), torch.randn(6, 8), torch.randn(6)
    for args, bias_kw in ((("x", "w", "b"), False), (("x", "w"), False), (("x", "w"), True)):
        g = torch.fx.Graph()
        needed = set(args) | ({"b"} if bias_kw else set())
        ph = {k: g.placeholder(k) for k in ("x", "w", "b") if k in needed}
        kw = {"bias": ph["b"]} if bias_kw else {}
        n = g.call_function(torch.ops.aten.linear.default, tuple(ph[k] for k in args), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        ins = [tx, tw] + ([tb] if "b" in args or bias_kw else [])
        out = run(*ins)
        assert "linear" in stats["mlx"], stats
        assert "linear" not in stats["cpu"], stats
        exp = torch.nn.functional.linear(tx, tw, tb if len(ins) == 3 else None)
        assert np.allclose(_np(out), exp.numpy(), atol=1e-4)
    print("test_linear PASS")


def test_cat_clone():
    # PT-AC: aten.cat.default (PAR-41 rank 13, 16,632 receipts) and
    # aten.clone.default (rank 97, op_db-only) were already registered —
    # the gap was coverage plus two edge defects fixed here: out-of-range
    # cat dims silently wrapped via `dim % nd` (torch IndexErrors), and
    # identity clone on a torch-tensor env value let a downstream in-place
    # op mutate the source's storage.
    rng = np.random.RandomState(5)

    def fx_run(nodes_fn, ins):
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        out = nodes_fn(g, phs)
        g.output(out)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        return run(*ins), stats

    # multi-input cat, dim=1, fp32 + bf16
    for dt in (torch.float32, torch.bfloat16):
        xs = [torch.tensor(rng.randn(2, n).astype(np.float32), dtype=dt)
              for n in (3, 4, 1)]
        out, stats = fx_run(
            lambda g, phs: g.call_function(torch.ops.aten.cat.default,
                                           ([phs[0], phs[1], phs[2]], 1), {}),
            xs)
        ref = torch.cat(xs, 1)
        assert stats["cpu"] == [], stats
        assert "cat" in stats["mlx"], stats
        assert np.allclose(_np(out), ref.float().numpy(), atol=2e-2)
        # dtype fidelity at the lowering: bf16 stays bf16 (the executor's
        # _torch output widens bf16 -> fp32, numpy has no bf16 buffer)
        got = mlx_vulkan.ATEN_TABLE["cat"]([mlx_vulkan._mx(x) for x in xs], 1)
        assert got.dtype == (mx.bfloat16 if dt == torch.bfloat16 else mx.float32), got.dtype

    # non-contiguous inputs on the cat axis
    xa = torch.tensor(rng.randn(4, 3).astype(np.float32)).t()
    xb = torch.tensor(rng.randn(4, 3).astype(np.float32)).t()
    assert not xa.is_contiguous() and not xb.is_contiguous()
    out, stats = fx_run(
        lambda g, phs: g.call_function(torch.ops.aten.cat.default,
                                       ([phs[0], phs[1]], 0), {}),
        [xa, xb])
    assert stats["cpu"] == [], stats
    assert np.allclose(_np(out), torch.cat([xa, xb], 0).numpy())

    # negative dim + empty-tensor filtering
    ea = torch.tensor(rng.randn(2, 3).astype(np.float32))
    eb = torch.empty(2, 0)
    out, stats = fx_run(
        lambda g, phs: g.call_function(torch.ops.aten.cat.default,
                                       ([phs[0], phs[1]], -1), {}),
        [ea, eb])
    assert stats["cpu"] == [], stats
    assert np.allclose(_np(out), torch.cat([ea, eb], -1).numpy())

    # out-of-range dim: torch IndexErrors, the executor must not silently
    # wrap to another axis
    try:
        torch.cat([ea, ea], 5)
        raise AssertionError("torch.cat accepted dim=5 on 2-D")
    except IndexError:
        pass
    try:
        fx_run(lambda g, phs: g.call_function(torch.ops.aten.cat.default,
                                              ([phs[0], phs[1]], 5), {}),
               [ea, ea])
        raise AssertionError("cat dim=5 on 2-D lowered silently")
    except IndexError:
        pass

    # clone lowers and returns equal values; clone-then-mul_ leaves the
    # source unmutated, and mul_ on the source leaves the clone unchanged
    z = torch.tensor([2.0, 3.0, 4.0])
    out, stats = fx_run(
        lambda g, phs: g.call_function(torch.ops.aten.clone.default,
                                       (phs[0],), {}),
        [z])
    assert stats["cpu"] == [], stats
    assert "clone" in stats["mlx"], stats
    assert np.allclose(_np(out), z.numpy())

    def clone_mul_graph(mutate_clone):
        def f(g, phs):
            cl = g.call_function(torch.ops.aten.clone.default, (phs[0],), {})
            tgt, operand = (cl, phs[0]) if mutate_clone else (phs[0], cl)
            g.call_function(torch.ops.aten.mul_.Tensor, (tgt, operand), {})
            return g.call_function(torch.ops.aten.cat.default,
                                   ([cl, phs[0]], 0), {})
        return f

    out, stats = fx_run(clone_mul_graph(True), [z])
    ref = torch.cat([z * z, z], 0)          # b = z.clone(); b *= z
    assert stats["cpu"] == [], stats
    assert np.allclose(_np(out), ref.numpy()), _np(out)

    out, stats = fx_run(clone_mul_graph(False), [z])
    ref = torch.cat([z, z * z], 0)          # b = z.clone(); z *= z
    assert stats["cpu"] == [], stats
    assert np.allclose(_np(out), ref.numpy()), _np(out)

    # compiled executor: same alias semantics
    g = torch.fx.Graph()
    a = g.placeholder("a")
    cl = g.call_function(torch.ops.aten.clone.default, (a,), {})
    g.call_function(torch.ops.aten.mul_.Tensor, (a, cl), {})
    g.output(g.call_function(torch.ops.aten.cat.default, ([cl, a], 0), {}))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run = mlx_vulkan.make_compiled_executor(gm, dt=mx.float32)
    out = run(z)
    o = out[0] if isinstance(out, (tuple, list)) else out
    assert np.allclose(_np(o), torch.cat([z, z * z], 0).numpy()), _np(o)

    # clone of a torch-tensor env value (post-fallback crossing) must be a
    # real duplicate: an in-place op on the clone cannot reach the source
    src = torch.tensor([1.0, 2.0])
    dup = mlx_vulkan._FUNC_SPECS["clone"](src)
    assert dup is not src
    dup.mul_(5)
    assert src.tolist() == [1.0, 2.0], src.tolist()
    print("test_cat_clone PASS")


def test_dispatch_registered():
    assert torch.topk in mlx_vulkan.FUNC_TABLE
    assert torch.multinomial in mlx_vulkan.FUNC_TABLE
    for name in ("topk", "multinomial", "convolution", "convolution_backward",
                 "_native_batch_norm_legit_functional", "threshold_backward", "t",
                 "gelu_backward", "tanh_backward", "embedding_dense_backward", "linear",
                 "cat", "clone"):
        assert name in mlx_vulkan.ATEN_TABLE, name
    print("test_dispatch_registered PASS")


def test_numel():
    rng = np.random.RandomState(9)
    for shape in ((), (0,), (5,), (2, 4, 8), (1, 1)):
        x = np.asarray(rng.randn(*shape), dtype=np.float32)
        exp = torch.numel(torch.tensor(x))
        got = mlx_vulkan._FUNC_SPECS["numel"](mx.array(x))
        assert got == exp, (shape, got, exp)
        assert isinstance(got, int), (shape, type(got))
    # through the eager executor: the aten.numel node must lower (glue),
    # not fall back to the CPU op
    t = torch.randn(3, 5, 2)
    g = torch.fx.Graph()
    ph = g.placeholder("x")
    n = g.call_function(torch.ops.aten.numel.default, (ph,))
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(mx.array(t.numpy()))
    assert out == t.numel(), (out, t.numel())
    assert "numel" in stats["mlx"], stats
    assert "numel" not in stats["cpu"], stats
    print("test_numel PASS")


def test_slice():
    rng = np.random.RandomState(11)
    x = rng.randn(6, 16, 9).astype(np.float32)
    tx = torch.tensor(x)
    a = mx.array(x)
    cases = [
        (1, 2, 10, 1),      # basic
        (1, None, None, 1),  # full dim
        (0, 3, None, 1),     # open stop
        (1, None, 5, 1),     # open start
        (2, -4, -1, 1),      # negative bounds
        (1, 30, 40, 1),      # clamped out-of-range
        (2, 0, 9, 2),        # positive step
    ]
    for dim, start, end, step in cases:
        exp = torch.ops.aten.slice.Tensor(tx, dim, start, end, step).numpy()
        got = mlx_vulkan._slice(a, dim, start, end, step)
        assert np.array_equal(_np(got), exp), (dim, start, end, step)
    # through the eager executor: the aten.slice node must lower
    t = torch.randn(4, 12)
    for dim, start, end, step in ((0, 1, 3, 1), (1, None, 7, 2)):
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        n = g.call_function(torch.ops.aten.slice.Tensor,
                            (ph, dim, start, end, step))
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(mx.array(t.numpy()))
        exp = torch.ops.aten.slice.Tensor(t, dim, start, end, step).numpy()
        assert np.array_equal(_np(out), exp), (dim, start, end, step)
        assert "slice" in stats["mlx"], stats
        assert "slice" not in stats["cpu"], stats
    print("test_slice PASS")


def test_transpose_expand():
    # PT-Z: aten.transpose.int (PAR-41 census rank 4, 59592 receipts) and
    # aten.expand.default (rank 33, 233 receipts) — broadcast glue around
    # matmul/SDPA. Shape + digest parity vs eager on 2D/4D, covering
    # negative dims, the -1 keep-dim sentinel, and leading broadcast dims.
    def _exec(target, ins):
        g = torch.fx.Graph()
        targs = [a for a in ins if torch.is_tensor(a)]
        phs = [g.placeholder(f"p{i}") for i in range(len(targs))]
        pit = iter(phs)
        args = tuple(next(pit) if torch.is_tensor(a) else a for a in ins)
        n = g.call_function(target, args)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*[mx.array(t.numpy()) for t in targs])
        return torch.as_tensor(_np(out)), stats

    x2, x4 = torch.randn(3, 5), torch.randn(2, 3, 4, 5)
    for ins in ((x2, 0, 1), (x2, -1, 0), (x4, 1, 3), (x4, -2, -1)):
        got, stats = _exec(torch.ops.aten.transpose.int, ins)
        exp = torch.ops.aten.transpose.int(*ins)
        assert got.shape == exp.shape and torch.equal(got, exp), (ins[1:], got.shape, exp.shape)
        assert "transpose" in stats["mlx"], stats
        assert "transpose" not in stats["cpu"], stats

    v1 = torch.randn(5)
    for ins in ((v1, (3, -1)),                # leading broadcast dim + sentinel
                (x2, (-1, -1)),               # keep both dims
                (x2, (2, -1, -1)),            # leading dim + two sentinels
                (torch.randn(1, 4), (3, -1)),
                (x4, (-1, -1, -1, -1)),
                (torch.randn(1, 4), (2, 3, -1, 4))):
        got, stats = _exec(torch.ops.aten.expand.default, ins)
        exp = torch.ops.aten.expand.default(*ins)
        assert got.shape == exp.shape and torch.equal(got, exp), (ins[1], got.shape, exp.shape)
        assert "expand" in stats["mlx"], stats
        assert "expand" not in stats["cpu"], stats

    # -1 in a leading non-existent dim is a lowering error (eager errors too)
    try:
        mlx_vulkan._expand(mx.array(v1.numpy()), (-1, 2))
        raise AssertionError("leading -1 must not lower")
    except ValueError:
        pass
    print("test_transpose_expand PASS")
def test_sum():
    # PT-U1: aten.sum.dim_IntList is the PAR-41 census top remaining fallback
    # (6,025 uses). The spec existed, but three forms fell back or mismatched:
    # dim 0/-1 on a 0-d tensor (mx.sum rejects axis >= ndim), an empty dim
    # list (aten reduces all dims), and the dtype= kwarg (was swallowed).
    # Prove the eager executor lowers every form (stats["mlx"], no stats["cpu"])
    # and matches torch CPU at fp32 and bf16.
    x32 = torch.tensor([[1.5, -2.0, 0.25], [3.0, -1.0, 2.0]])
    xb = x32.bfloat16()
    x0 = torch.tensor(5.0)
    cases = [
        (x32, {"dim": [1]}, torch.sum(x32, dim=[1])),
        (x32, {"dim": [1], "keepdim": True}, torch.sum(x32, dim=[1], keepdim=True)),
        (x32, {"dim": (0, 1)}, torch.sum(x32, dim=(0, 1))),
        (x32, {"dim": [-1]}, torch.sum(x32, dim=[-1])),
        (x32, {"dim": []}, torch.sum(x32)),
        (x32, {"dim": [], "keepdim": True}, torch.sum(x32, dim=[0, 1], keepdim=True)),
        (xb, {"dim": [1], "dtype": torch.float32}, torch.sum(xb, dim=[1], dtype=torch.float32)),
        (x0, {"dim": [0]}, torch.ops.aten.sum.dim_IntList(x0, [0])),
        (x0, {"dim": [-1], "keepdim": True}, torch.ops.aten.sum.dim_IntList(x0, [-1], True)),
    ]
    for ins, kw, ref in cases:
        g = torch.fx.Graph()
        ph = g.placeholder("p0")
        n = g.call_function(torch.ops.aten.sum.dim_IntList, (ph,), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(ins)
        assert stats["cpu"] == [], (stats, kw)
        assert any("sum" in s for s in stats["mlx"]), (stats, kw)
        got = out.float().numpy()
        assert np.allclose(got, ref.float().numpy(), atol=2e-2), (kw, got, ref.float().numpy())
        assert out.dtype == ref.dtype, (kw, out.dtype, ref.dtype)
    for kw, ref in (({"dim": [1]}, torch.sum(xb, dim=[1])),
                    ({"dim": [1], "keepdim": True}, torch.sum(xb, dim=[1], keepdim=True))):
        g = torch.fx.Graph()
        ph = g.placeholder("p0")
        n = g.call_function(torch.ops.aten.sum.dim_IntList, (ph,), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(xb)
        assert stats["cpu"] == [], (stats, kw)
        got = out.float().numpy()
        assert np.allclose(got, ref.float().numpy(), atol=3e-2), (kw, got, ref.float().numpy())
    print("test_sum PASS")


def test_reductions():
    # PT-AB: mean.dim had the same 0-d/empty-dims/dtype gaps fixed for sum in
    # #867; var/var_mean/std_mean had no spec at all; and the std lambda
    # ignored `correction` so it silently computed ddof=0 (biased) where torch
    # defaults to ddof=1 (unbiased). Covers dim lists, keepdim, empty-dims,
    # correction values, 0-d tensors, dtype cast, fp32 + bf16 + fp16.
    x32 = torch.tensor([[1.5, -2.0, 0.25], [3.0, -1.0, 2.0]])
    xb = x32.bfloat16()
    xh = x32.half()
    x0 = torch.tensor(5.0)

    def run_one(target, ins, kw):
        ref = target(*ins, **kw)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target), kw)
        # #864: _torch widens bf16 outputs to f32 (numpy has no bf16 buffer)
        assert out.dtype == ref.dtype or (ref.dtype == torch.bfloat16 and out.dtype == torch.float32), \
            (str(target), out.dtype, ref.dtype)
        assert np.allclose(torch.as_tensor(out).float().numpy(), ref.float().numpy(), atol=2e-2), \
            (str(target), kw, torch.as_tensor(out).float().numpy(), ref.float().numpy())

    for target, ins, kw in [
        (torch.ops.aten.mean.dim, (x32,), {"dim": [1]}),
        (torch.ops.aten.mean.dim, (x32,), {"dim": [1], "keepdim": True}),
        (torch.ops.aten.mean.dim, (x32,), {"dim": []}),
        (torch.ops.aten.mean.dim, (x32,), {"dim": [1], "dtype": torch.float32}),
        (torch.ops.aten.mean.dim, (x0,), {"dim": [0]}),
        (torch.ops.aten.mean.dim, (x0,), {"dim": [-1], "keepdim": True}),
        (torch.ops.aten.mean.dim, (xb,), {"dim": [1]}),
        (torch.ops.aten.mean.dim, (xh,), {"dim": [1]}),
        (torch.ops.aten.mean.dim, (x32,), {"dim": [0, 1]}),
        (torch.ops.aten.mean.dim, (x32,), {"dim": [0, 1], "keepdim": True}),
        (torch.ops.aten.var.dim, (x32,), {"dim": [1]}),
        (torch.ops.aten.var.dim, (x32,), {"dim": [1], "unbiased": False}),
        (torch.ops.aten.var.dim, (x32,), {"dim": [0], "keepdim": True}),
        (torch.ops.aten.var.dim, (x32,), {"dim": []}),
        (torch.ops.aten.var.dim, (x32,), {"dim": [0, 1]}),
        (torch.ops.aten.var.dim, (x32,), {"dim": [0, 1], "keepdim": True}),
        (torch.ops.aten.var.dim, (xh,), {"dim": [1]}),
        (torch.ops.aten.std.dim, (x32,), {"dim": [1]}),
        (torch.ops.aten.std.dim, (x32,), {"dim": [0], "keepdim": True}),
        (torch.ops.aten.std.dim, (x32,), {"dim": [0, 1]}),
        (torch.ops.aten.std.dim, (xb,), {"dim": [1]}),
        (torch.ops.aten.std.dim, (xh,), {"dim": [1]}),
    ]:
        run_one(target, ins, kw)

    # correction lives on the .correction overloads (unbiased=1 default)
    for target in (torch.ops.aten.var.correction, torch.ops.aten.std.correction):
        for corr in (0, 1, 2):
            run_one(target, (x32,), {"dim": [1], "correction": corr})

    # .default overloads bind a bool `unbiased` as arg 2, not dim — a missed
    # rebind once produced per-row vectors instead of scalars (w74 review)
    for target in (torch.ops.aten.var.default, torch.ops.aten.std.default):
        for ub in (True, False):
            ref = target(x32, ub)
            g = torch.fx.Graph()
            ph = g.placeholder("p0")
            n = g.call_function(target, (ph, ub), {})
            g.output(n)
            gm = torch.fx.GraphModule(torch.nn.Module(), g)
            run, stats = mlx_vulkan.make_executor(gm)
            out = run(x32)
            assert stats["cpu"] == [], (stats, str(target), ub)
            assert np.allclose(torch.as_tensor(out).float().numpy(), ref.float().numpy(), atol=2e-2), \
                (str(target), ub, torch.as_tensor(out).float().numpy(), ref.float().numpy())

    # dof<=0 semantics (w74 review): positive ssq / nonpositive dof -> +inf,
    # zero ssq -> nan; negative correction widens dof. Scalar/empty inputs
    # intentionally fall back to CPU (subtle inf/nan cases) — parity only.
    x3 = torch.tensor([1.0, 2.0, 3.0])
    for target, ins, kw, lowers in [
        (torch.ops.aten.var.correction, (x3,), {"dim": [0], "correction": 3.5}, True),
        (torch.ops.aten.var.correction, (x3,), {"dim": [0], "correction": 3}, True),
        (torch.ops.aten.std.correction, (x3,), {"dim": [0], "correction": 3.5}, True),
        (torch.ops.aten.var.correction, (x3,), {"dim": [0], "correction": -1}, True),
        (torch.ops.aten.var.correction, (x0,), {"correction": -1}, False),
        (torch.ops.aten.var.correction, (x0,), {"correction": 1}, False),
        (torch.ops.aten.std.correction, (x0,), {"correction": 2}, False),
        (torch.ops.aten.var.correction, (x32,), {"dim": [1], "correction": 0.5}, True),
    ]:
        ref = target(*ins, **kw)
        g = torch.fx.Graph()
        ph = g.placeholder("p0")
        n = g.call_function(target, (ph,), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert (stats["cpu"] == []) == lowers, (stats, str(target), kw)
        assert np.allclose(torch.as_tensor(out).float().numpy(), ref.float().numpy(), atol=2e-2, equal_nan=True), \
            (str(target), kw, torch.as_tensor(out).float().numpy(), ref.float().numpy())

    # tuple-output ops
    for target, kw in ((torch.ops.aten.var_mean.dim, {"dim": [1]}),
                       (torch.ops.aten.std_mean.dim, {"dim": [1]}),
                       (torch.ops.aten.var_mean.default, {}),
                       (torch.ops.aten.std_mean.default, {})):
        ref = target(x32, **kw)
        g = torch.fx.Graph()
        ph = g.placeholder("p0")
        n = g.call_function(target, (ph,), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(x32)
        assert stats["cpu"] == [], (stats, str(target))
        for o, r in zip(out, ref):
            assert np.allclose(torch.as_tensor(o).float().numpy(), r.float().numpy(), atol=2e-2), \
                (str(target), o, r)
    print("test_reductions PASS")


def test_bitwise_and_():
    # PT-X: aten.bitwise_and_.Tensor (PAR-41 census rank 45) — the last
    # unclaimed in-place _ op, built on PT-W's mul_ propagation (#861).
    # Covers the Tensor + bool/logical paths, write-back so downstream
    # reads of the mutated arg see the update, and stats['mlx'] proof.
    xi = torch.tensor([[0b1100, 0b1010], [0b1111, 0b0000]], dtype=torch.int32)
    yi = torch.tensor([[0b1010, 0b1100], [0b1111, 0b1111]], dtype=torch.int32)
    xb = torch.tensor([[True, False], [True, True]])
    yb = torch.tensor([[True, True], [False, True]])
    for ins in ((xi.clone(), yi), (xb.clone(), yb), (xi.clone(), torch.tensor(0b1100, dtype=torch.int32))):
        g = torch.fx.Graph()
        px, py = g.placeholder("x"), g.placeholder("y")
        args = (px, py) if torch.is_tensor(ins[1]) else (px, ins[1])
        n = g.call_function(torch.ops.aten.bitwise_and_.Tensor, args)
        # downstream read of the ORIGINAL placeholder must see the mutation:
        # x | (x&y) = x if stale, (x&y) | (x&y) = x&y if write-back landed
        n2 = g.call_function(torch.ops.aten.bitwise_or.Tensor, (px, n))
        g.output(n2)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        targs = [a for a in ins if torch.is_tensor(a)]
        out = run(*[mx.array(t.numpy()) for t in targs])
        ref = ins[0].clone()
        torch.ops.aten.bitwise_and_.Tensor(ref, ins[1])
        exp = torch.ops.aten.bitwise_or.Tensor(ref, ref)
        got = torch.as_tensor(_np(out))
        assert got.shape == exp.shape and torch.equal(got, exp), (got, exp)
        assert "bitwise_and_" in stats["mlx"], stats
        assert "bitwise_and_" not in stats["cpu"], stats

    # In-place contract: the broadcast result keeps the destination's shape
    # and dtype kind (int64->int8 narrows back; int64->bool and shape growth
    # refuse the same way torch does, verified 2026-11-24).
    di8 = torch.tensor([[127, 6]], dtype=torch.int8)
    ri64 = torch.tensor([[255, 3]], dtype=torch.int64)
    g = torch.fx.Graph()
    px, py = g.placeholder("x"), g.placeholder("y")
    n = g.call_function(torch.ops.aten.bitwise_and_.Tensor, (px, py))
    n2 = g.call_function(torch.ops.aten.bitwise_or.Tensor, (px, n))
    g.output(n2)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(mx.array(di8.numpy()), mx.array(ri64.numpy()))
    ref = di8.clone()
    torch.ops.aten.bitwise_and_.Tensor(ref, ri64)
    exp = torch.ops.aten.bitwise_or.Tensor(ref, ref)
    got = torch.as_tensor(_np(out))
    assert got.dtype == torch.int8 and torch.equal(got, exp), (got, exp)
    assert "bitwise_and_" in stats["mlx"] and "bitwise_and_" not in stats["cpu"], stats

    for xt, yt in (
        (torch.tensor([[True, False]]), torch.tensor([[1, 0]], dtype=torch.int64)),
        (torch.tensor([[1, 2]], dtype=torch.int32), torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)),
    ):
        try:
            torch.ops.aten.bitwise_and_.Tensor(xt.clone(), yt)
        except RuntimeError:
            pass
        else:
            raise AssertionError("torch itself accepts this case; not a refusal test")
        g = torch.fx.Graph()
        px, py = g.placeholder("x"), g.placeholder("y")
        n = g.call_function(torch.ops.aten.bitwise_and_.Tensor, (px, py))
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        try:
            run(mx.array(xt.numpy()), mx.array(yt.numpy()))
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                f"in-place refusal expected: {xt.dtype}{tuple(xt.shape)} & {yt.dtype}{tuple(yt.shape)}")
    print("test_bitwise_and_ PASS")


def test_arc_trig():
    # PT-AH: the circular arc-trig set, the highest-ranked unclaimed aten
    # family in w7R's PAR-41 fallback census: acos rank 53, asin 77,
    # atan 79, atan2 78 (hyperbolic sibling trio owned by wB1). Prove the
    # eager executor lowers each aten op (stats["mlx"]) and matches torch
    # CPU at fp32 and bf16; atan2 additionally covers broadcasting.
    u = torch.tensor([[0.5, -0.75], [0.125, 0.999]])
    w = torch.tensor([[0.2, -0.3], [-0.6, 0.45]])
    ub, wb = u.bfloat16(), w.bfloat16()
    cases = [
        (torch.ops.aten.acos.default, (u,), torch.acos(u), "acos"),
        (torch.ops.aten.acos.default, (ub,), torch.acos(ub), "acos"),
        (torch.ops.aten.asin.default, (u,), torch.asin(u), "asin"),
        (torch.ops.aten.asin.default, (ub,), torch.asin(ub), "asin"),
        (torch.ops.aten.atan.default, (u,), torch.atan(u), "atan"),
        (torch.ops.aten.atan.default, (ub,), torch.atan(ub), "atan"),
        (torch.ops.aten.atan2.default, (u, w), torch.atan2(u, w), "atan2"),
        (torch.ops.aten.atan2.default, (ub, wb), torch.atan2(ub, wb), "atan2"),
        (torch.ops.aten.atan2.default, (u, w[0]), torch.atan2(u, w[0]), "atan2"),
    ]
    for target, ins, ref, name in cases:
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, name)
        assert any(name in s for s in stats["mlx"]), (stats, name)
        assert np.allclose(out.float().numpy(), ref.float().numpy(),
                           atol=2e-2), (name, out.float().numpy())
        assert out.dtype == ref.dtype or (
            ref.dtype == torch.bfloat16 and out.dtype == torch.float32), \
            (name, out.dtype, ref.dtype)
    # domain guard: acos/asin inputs outside [-1, 1] must produce NaN like
    # torch, not a wrong number or an exception
    oob = torch.tensor([1.5, -1.5])
    for target in (torch.ops.aten.acos.default, torch.ops.aten.asin.default):
        g = torch.fx.Graph()
        p = g.placeholder("p0")
        n = g.call_function(target, (p,), {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(oob)
        assert stats["cpu"] == [], stats
        assert bool(torch.isnan(out).all()) == bool(
            torch.isnan(target(oob)).all()), target
    # IEEE signed-zero quadrants: atan2(+0, -0) = +pi, atan2(-0, -0) = -pi.
    # Pin values and sign bits for all four (+/-0, +/-0) pairs, plus the
    # non-zero-y rows of the x = -0 column the fix must not regress.
    for yv, xv in ((0.0, 0.0), (0.0, -0.0), (-0.0, 0.0), (-0.0, -0.0),
                   (1.0, -0.0), (-1.0, -0.0),
                   (float("inf"), -0.0), (float("-inf"), -0.0),
                   (float("nan"), -0.0)):
        yt = torch.tensor([yv])
        xt = torch.tensor([xv])
        g = torch.fx.Graph()
        py, px = g.placeholder("y"), g.placeholder("x")
        n = g.call_function(torch.ops.aten.atan2.default, (py, px), {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(yt, xt)
        ref = torch.atan2(yt, xt)
        assert stats["cpu"] == [], stats
        if math.isnan(ref.item()):
            assert math.isnan(out.item()), (yv, xv, out.item())
            continue
        assert out.item() == ref.item(), (yv, xv, out.item(), ref.item())
        assert math.copysign(1.0, out.item()) == math.copysign(1.0, ref.item()), \
            (yv, xv, out.item(), ref.item())
    # the correction must not widen the result: FP16 inputs keep FP16 and
    # torch's rounded fp16 pi, BF16 inputs keep BF16 through the helper.
    for dt, yv, xv in ((torch.float16, 0.0, -0.0), (torch.float16, -0.0, -0.0)):
        yt = torch.tensor([yv], dtype=dt)
        xt = torch.tensor([xv], dtype=dt)
        g = torch.fx.Graph()
        py, px = g.placeholder("y"), g.placeholder("x")
        n = g.call_function(torch.ops.aten.atan2.default, (py, px), {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(yt, xt)
        ref = torch.atan2(yt, xt)
        assert stats["cpu"] == [], stats
        assert out.dtype == ref.dtype == dt, (out.dtype, ref.dtype)
        assert out.item() == ref.item(), (dt, yv, xv, out.item(), ref.item())
    b16 = mlx_vulkan._atan2(mx.array([-0.0], dtype=mx.bfloat16),
                            mx.array([-0.0], dtype=mx.bfloat16))
    assert b16.dtype == mx.bfloat16, b16.dtype
    assert float(b16[0]) == torch.atan2(
        torch.tensor(-0.0, dtype=torch.bfloat16),
        torch.tensor(-0.0, dtype=torch.bfloat16)).item(), float(b16[0])
    print("test_arc_trig PASS")


if __name__ == "__main__":
    test_topk()
    test_multinomial()
    test_convolution_backward()
    test_threshold_backward()
    test_t()
    test_embedding_dense_backward()
    test_bn_legit_functional()
    test_scalar_tensor_factory_lowering()
    test_arange()
    test_mul()
    test_pcache_skips_static_weights()
    test_compiled_executor_nonarray_outputs()
    test_gelu_backward()
    test_linear()
    test_dispatch_registered()
    test_numel()
    test_slice()
    test_transpose_expand()
    test_cat_clone()
    test_sum()
    test_reductions()
    test_bitwise_and_()
    test_arc_trig()
    print("ALL PASS")
