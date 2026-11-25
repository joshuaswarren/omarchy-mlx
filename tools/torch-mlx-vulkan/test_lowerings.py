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


def _provenance():
    # AGENTS.md: a parity result travels with the loaded binary's identity.
    # Prefer the repo gate (libmlx hash vs the wheel's RECORD); a bare file
    # copy outside the tree falls back to a direct libmlx hash.
    scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, os.pardir, "scripts")
    if os.path.exists(os.path.join(scripts, "mlx_provenance.py")):
        sys.path.insert(0, scripts)
        import mlx_provenance
        print("provenance:", mlx_provenance.provenance_line(
            mlx_provenance.installed_provenance()))
        return
    import hashlib
    lib = os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so")
    h = hashlib.sha256(open(lib, "rb").read()).hexdigest()[:16]
    print(f"provenance: mx={mx.__version__} libmlx sha256:{h} "
          "(source build, no RECORD)")


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
        # allclose broadcasts — pin the shape too so a scalar-vs-(1,1)
        # keepdim slip can't pass silently
        assert torch.as_tensor(out).shape == ref.shape, \
            (str(target), kw, torch.as_tensor(out).shape, ref.shape)
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
        (torch.ops.aten.std.dim, (x32,), {"dim": [0, 1], "keepdim": True}),
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


def test_remainder():
    _provenance()
    # PT-AN: aten.remainder.default (PAR-41 rank 132, receipt-zero) had no
    # spec. torch.remainder takes the DIVISOR's sign; mx.remainder matches on
    # every sign/dtype combo (verified -7%3=2, 7%-3=-2 in both).
    x32 = torch.tensor([7.0, -7.0, 7.5, -7.5])
    y32 = torch.tensor([3.0, 3.0, -2.0, -2.0])
    i64 = torch.tensor([7, -7, 7, -7], dtype=torch.int64)
    j64 = torch.tensor([3, 3, -4, -4], dtype=torch.int64)

    def fx_fn(target, args, ins):
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        it = iter(phs)
        a = tuple(next(it) if torch.is_tensor(v) else v for v in args)
        n = g.call_function(target, a)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        return run(*ins), stats

    def check(name, target, args, ins, ref):
        out, stats = fx_fn(target, args, ins)
        assert stats["cpu"] == [], (name, stats)
        assert name in stats["mlx"], (name, stats)
        out = torch.as_tensor(_np(out))
        ref = torch.as_tensor(ref)
        assert out.dtype == ref.dtype, (name, out.dtype, ref.dtype)
        assert np.allclose(_np(out.float()), ref.float().numpy()), \
            (name, _np(out), ref)

    # sign matrix: (dividend sign, divisor sign) for fp32 + int64
    check("remainder", torch.ops.aten.remainder.Tensor, (x32, y32),
          [x32, y32], torch.remainder(x32, y32))
    check("remainder", torch.ops.aten.remainder.Tensor, (i64, j64),
          [i64, j64], torch.remainder(i64, j64))
    # scalar divisor, both signs, int + float
    check("remainder", torch.ops.aten.remainder.Scalar, (x32, -2.0), [x32],
          torch.remainder(x32, -2.0))
    check("remainder", torch.ops.aten.remainder.Scalar, (i64, 3), [i64],
          torch.remainder(i64, 3))
    # scalar-only aten.remainder.default(7,3): the census hand-sample and the
    # _ALWAYS_LOWER gate; int64 boxing via _ar (assert_close checks dtype)
    check("remainder", torch.ops.aten.remainder.default, (7, 3), [],
          torch.remainder(torch.tensor(7), torch.tensor(3)))
    print("test_remainder PASS")


def test_hyperbolic():
    # PT-U11: the hyperbolic family — the top unclaimed aten family in
    # w7R's PAR-41 fallback census once the circular arc-trig ops went to
    # #889. Prove the eager executor lowers each op (stats["mlx"]) and
    # matches torch CPU at fp32 and bf16.
    u = torch.tensor([[0.5, -0.75], [0.125, 0.999]])
    v = torch.tensor([[1.5, 2.0], [1.001, 4.25]])   # acosh domain: x >= 1
    a = torch.tensor([[0.5, -0.75], [0.125, 0.9]])  # atanh domain: |x| < 1
    ub, vb, ab = u.bfloat16(), v.bfloat16(), a.bfloat16()
    cases = [
        (torch.ops.aten.sinh.default, (u,), torch.sinh(u), "sinh"),
        (torch.ops.aten.sinh.default, (ub,), torch.sinh(ub), "sinh"),
        (torch.ops.aten.cosh.default, (u,), torch.cosh(u), "cosh"),
        (torch.ops.aten.cosh.default, (ub,), torch.cosh(ub), "cosh"),
        (torch.ops.aten.asinh.default, (u,), torch.asinh(u), "asinh"),
        (torch.ops.aten.asinh.default, (ub,), torch.asinh(ub), "asinh"),
        (torch.ops.aten.acosh.default, (v,), torch.acosh(v), "acosh"),
        (torch.ops.aten.acosh.default, (vb,), torch.acosh(vb), "acosh"),
        (torch.ops.aten.atanh.default, (a,), torch.atanh(a), "atanh"),
        (torch.ops.aten.atanh.default, (ab,), torch.atanh(ab), "atanh"),
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
    # domain guards: acosh(x < 1) and atanh(|x| >= 1) are NaN in torch;
    # the lowering must produce NaN, not a wrong number or an exception.
    for target, oob in ((torch.ops.aten.acosh.default,
                         torch.tensor([0.5, -1.0])),
                        (torch.ops.aten.atanh.default,
                         torch.tensor([1.0, -1.5]))):
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
    # IEEE edges the Vulkan kernels drop: sinh(-0) keeps the sign bit,
    # asinh(+/-inf) returns the signed infinities, cosh(+/-inf) = +inf,
    # acosh(+inf) = +inf. Pin value AND sign bit against torch.
    for target, xv in (
            (torch.ops.aten.sinh.default, -0.0),
            (torch.ops.aten.sinh.default, float("inf")),
            (torch.ops.aten.sinh.default, float("-inf")),
            (torch.ops.aten.asinh.default, float("inf")),
            (torch.ops.aten.asinh.default, float("-inf")),
            (torch.ops.aten.asinh.default, -0.0),
            (torch.ops.aten.cosh.default, float("inf")),
            (torch.ops.aten.cosh.default, float("-inf")),
            (torch.ops.aten.acosh.default, float("inf"))):
        xt = torch.tensor([xv])
        g = torch.fx.Graph()
        p = g.placeholder("p0")
        n = g.call_function(target, (p,), {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(xt)
        ref = target(xt)
        assert stats["cpu"] == [], (target, stats)
        assert out.item() == ref.item(), (target, xv, out.item(), ref.item())
        assert math.copysign(1.0, out.item()) == \
            math.copysign(1.0, ref.item()), (target, xv, out.item())
    print("test_hyperbolic PASS")


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


def test_addcmul_addcdiv():
    # PT-U10: aten.addcmul/addcdiv (input + value*t1*t2, input + value*t1/t2)
    # — the last unclaimed receipt-zero composite row. Prove the eager
    # executor lowers both (stats["mlx"]) and matches torch CPU at fp32
    # and bf16, with broadcasting and a non-default value.
    rng = np.random.RandomState(7)
    a = torch.tensor(rng.randn(2, 3).astype(np.float32))
    t1 = torch.tensor(rng.randn(2, 3).astype(np.float32))
    t2 = torch.tensor(rng.randn(2, 3).astype(np.float32) + 1.5)
    rb = torch.tensor(rng.randn(1, 3).astype(np.float32))

    def fx_run(target, ins, kw):
        ref = target(*ins, **kw)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target))
        name = str(target).split("aten.")[1].split(".")[0]
        assert any(name in s for s in stats["mlx"]), (stats, name)
        assert torch.as_tensor(out).shape == ref.shape, \
            (str(target), torch.as_tensor(out).shape, ref.shape)
        # the executor's bf16 output path widens to fp32 (no bf16 numpy
        # dtype); pin dtype parity everywhere it can be expressed
        if ref.dtype != torch.bfloat16:
            assert str(out.dtype).split(".")[-1] == \
                str(ref.dtype).split(".")[-1], \
                (str(target), kw, str(out.dtype), str(ref.dtype))
        assert np.allclose(torch.as_tensor(out).float().numpy(),
                           ref.float().numpy(), atol=2e-2), \
            (str(target), kw, torch.as_tensor(out).float().numpy(),
             ref.float().numpy())

    def fx_expect_raise(target, ins, kw):
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        try:
            run(*ins)
        except RuntimeError:
            return
        raise AssertionError("no raise", str(target), kw)

    def fx_defer(target, ins, kw):
        # 0-dim mixed-dtype promotion is weak in torch; the lowering defers
        # to the CPU fallback, whose output must still match torch
        ref = target(*ins, **kw)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        name = str(target).split("aten.")[1].split(".")[0]
        assert any(name in s for s in stats["cpu"]), (stats, name)
        assert torch.as_tensor(out).dtype == ref.dtype, \
            (str(target), torch.as_tensor(out).dtype, ref.dtype)
        assert np.allclose(torch.as_tensor(out).float().numpy(),
                           ref.float().numpy(), atol=2e-2), \
            (str(target), kw, torch.as_tensor(out).float().numpy(),
             ref.float().numpy())

    ia = torch.tensor([1, -2, 3], dtype=torch.int64)
    it1 = torch.tensor([4, 5, -6], dtype=torch.int64)
    it2 = torch.tensor([2, -1, 3], dtype=torch.int64)
    b3 = torch.tensor([True, True, True])
    h0 = torch.zeros(1, dtype=torch.float16)
    h6 = torch.full((1,), 60000.0, dtype=torch.float16)
    hq = torch.full((1,), 0.25, dtype=torch.float16)

    cases = [
        (torch.ops.aten.addcmul.default, (a, t1, t2), {}),
        (torch.ops.aten.addcmul.default, (a, t1, t2), {"value": 2.5}),
        (torch.ops.aten.addcmul.default, (a, t1, rb), {}),
        (torch.ops.aten.addcmul.default,
         (a.bfloat16(), t1.bfloat16(), t2.bfloat16()), {}),
        (torch.ops.aten.addcdiv.default, (a, t1, t2), {}),
        (torch.ops.aten.addcdiv.default, (a, t1, t2), {"value": -0.5}),
        (torch.ops.aten.addcdiv.default, (a, t1, rb), {}),
        (torch.ops.aten.addcdiv.default,
         (a.bfloat16(), t1.bfloat16(), t2.bfloat16()), {}),
        # int value is legal (promotes like a wrapped scalar)
        (torch.ops.aten.addcmul.default, (a, t1, t2), {"value": 1}),
        # integral tensors: result stays integral; a fractional value
        # truncates to the compute dtype (wrapped-scalar rule)
        (torch.ops.aten.addcmul.default, (ia, it1, it2), {"value": 2}),
        (torch.ops.aten.addcmul.default, (ia, it1, it2), {"value": 2.5}),
        (torch.ops.aten.addcmul.default, (ia, it1, it2), {"value": -2.5}),
        # mixed integral/floating promotes to float — legal
        (torch.ops.aten.addcmul.default, (ia, t1[0], it2), {"value": 0.5}),
        (torch.ops.aten.addcdiv.default, (ia, t1[0], it2), {}),
        # fp32 opmath: value*t1 overflows fp16 but the torch result is
        # finite (2*(60000*0.25)=30000, 2*(60000/60000)=2)
        (torch.ops.aten.addcmul.default, (h0, h6, hq), {"value": 2}),
        (torch.ops.aten.addcdiv.default, (h0, h6, h6), {"value": 2}),
        # bool+int mix promotes to int — legal
        (torch.ops.aten.addcmul.default,
         (b3, torch.tensor([2, 2, 2]), torch.tensor([3, 3, 3])), {}),
        # value applies to t1 first: 0*(t1*t2) with an inf fp32 product
        # would be NaN; (0*t1)*t2 is 0 like torch
        (torch.ops.aten.addcmul.default,
         (a, torch.full((2, 3), 1e20), torch.full((2, 3), 1e20)),
         {"value": 0}),
        (torch.ops.aten.addcdiv.default,
         (a, torch.full((2, 3), 1e20), torch.full((2, 3), 1e-20)),
         {"value": 0}),
        # operands cast to the common dtype before multiply: 2*int8(100)
        # wraps to -112 in int8 but is 400 at int64
        (torch.ops.aten.addcmul.default,
         (torch.zeros(1, dtype=torch.int64),
          torch.full((1,), 100, dtype=torch.int8),
          torch.full((1,), 2, dtype=torch.int64)), {"value": 2}),
        (torch.ops.aten.addcmul.default,
         (torch.zeros(1, dtype=torch.int64),
          torch.full((1,), 100, dtype=torch.int8),
          torch.full((1,), 2, dtype=torch.int8)), {"value": 1}),
    ]
    for target, ins, kw in cases:
        fx_run(target, ins, kw)

    # Torch raises on integral addcdiv (integer division unsupported) —
    # the guard keys on the division operands, not self
    fx_expect_raise(torch.ops.aten.addcdiv.default, (ia, it1, it2), {})
    fx_expect_raise(torch.ops.aten.addcdiv.default, (ia, it1, it2),
                    {"value": 1})
    fx_expect_raise(torch.ops.aten.addcdiv.default, (a[0], it1, it2), {})
    fx_expect_raise(torch.ops.aten.addcdiv.default, (a[0], b3, it2), {})
    # all-bool addcmul has no kernel
    fx_expect_raise(torch.ops.aten.addcmul.default, (b3, b3, b3), {})

    # 0-dim mixed-dtype operands: torch promotes weakly, so the lowering
    # defers to the CPU fallback rather than emulate it
    fx_defer(torch.ops.aten.addcmul.default,
             (h0, torch.tensor(2.0, dtype=torch.float32),
              hq.expand(1)), {"value": 1})
    fx_defer(torch.ops.aten.addcdiv.default,
             (h0, torch.tensor(2.0, dtype=torch.float32),
              hq.expand(1)), {"value": 1})
    fx_defer(torch.ops.aten.addcmul.default,
             (torch.zeros(3, dtype=torch.int8),
              torch.tensor(100, dtype=torch.int64),
              torch.full((3,), 2, dtype=torch.int8)), {"value": 1})
    print("test_addcmul_addcdiv PASS")

def test_random_family():
    # PT-U12: the aten sampling family (uniform/cauchy/exponential/
    # geometric/log_normal/normal.*) was evaluated for lowering and
    # deliberately deferred: mx.random cannot reproduce torch's
    # generator-state consumption, exclusive-bound semantics, or weak
    # promotion. Every op must land in stats["cpu"], freshness and
    # seed-replay must hold, and torch's param checks must surface.
    def _fx(target, arglist, kw=None):
        ins = [a for a in arglist if isinstance(a, torch.Tensor)]
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        it = iter(phs)
        n = g.call_function(
            target, tuple(
                next(it) if isinstance(a, torch.Tensor) else a
                for a in arglist), kw or {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        name = str(target).split("aten.")[1].split(".")[0]
        return torch.as_tensor(out), stats, name

    def fx_defer(target, arglist, kw=None):
        out, stats, name = _fx(target, arglist, kw)
        assert name in stats["cpu"], (stats, name)
        return out

    big = torch.zeros(20000)
    # each member of the family defers to torch and draws correctly
    o = fx_defer(torch.ops.aten.uniform.default, [big, 2.0, 5.0])
    assert o.shape == big.shape and o.min() >= 2.0 and o.max() < 5.0
    o = fx_defer(torch.ops.aten.exponential.default, [big],
                 {"lambd": 0.5})
    assert o.min() >= 0 and 1.6 < o.mean() < 2.4
    o = fx_defer(torch.ops.aten.geometric.default, [big], {"p": 0.25})
    assert o.min() >= 1 and 3.6 < o.mean() < 4.4
    o = fx_defer(torch.ops.aten.cauchy.default, [big],
                 {"median": 2.0, "sigma": 0.5})
    assert 1.9 < o.median() < 2.1
    o = fx_defer(torch.ops.aten.log_normal.default, [big],
                 {"mean": 1.0, "std": 0.5})
    lo = torch.log(o)
    assert 0.9 < lo.mean() < 1.1 and 0.4 < lo.std() < 0.6
    mt = torch.full((20000,), 2.0)
    st = torch.full((20000,), 0.5)
    for arglist in ([mt, st], [mt, 0.5], [2.0, st], [2.0, 0.5, (4, 3)]):
        target = (torch.ops.aten.normal.Tensor_Tensor
                  if len(arglist) == 2
                  and all(isinstance(a, torch.Tensor) for a in arglist)
                  else torch.ops.aten.normal.Tensor_float
                  if isinstance(arglist[0], torch.Tensor)
                  else torch.ops.aten.normal.float_Tensor
                  if isinstance(arglist[1], torch.Tensor)
                  else torch.ops.aten.normal.float_float)
        fx_defer(target, arglist)

    # deferred draws stay fresh across calls through the registered
    # backend, and a torch.manual_seed reset replays the same sample
    def reg_uniform():
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        n = g.call_function(torch.ops.aten.uniform.default, (ph,))
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        return mlx_vulkan.mlx_vulkan_backend(gm, [])
    runner = reg_uniform()
    x = torch.zeros(4096)

    def r1():
        o = runner(x)
        return torch.as_tensor(
            o[0] if isinstance(o, (tuple, list)) else o)

    o1, o2 = r1(), r1()
    assert not torch.equal(o1, o2)
    torch.manual_seed(1729)
    s1 = r1()
    torch.manual_seed(1729)
    s2 = r1()
    assert torch.equal(s1, s2)
    # and the deferred op consumed torch's generator: the next torch
    # draw after our call matches eager torch's next draw
    torch.manual_seed(1729)
    defer_then_torch = r1(), torch.zeros(4096).uniform_()
    torch.manual_seed(1729)
    eager_then_torch = torch.zeros(4096).uniform_(), torch.zeros(4096).uniform_()
    assert torch.equal(defer_then_torch[1], eager_then_torch[1])

    # torch's parameter checks surface through the fallback
    def fx_raise(target, arglist, kw=None):
        ins = [a for a in arglist if isinstance(a, torch.Tensor)]
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        it = iter(phs)
        n = g.call_function(
            target, tuple(
                next(it) if isinstance(a, torch.Tensor) else a
                for a in arglist), kw or {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, _stats = mlx_vulkan.make_executor(gm)
        try:
            run(*ins)
        except RuntimeError:
            return
        raise AssertionError(f"{target} accepted invalid params")

    fx_raise(torch.ops.aten.uniform.default, [big, 5.0, 2.0])
    fx_raise(torch.ops.aten.exponential.default, [big], {"lambd": 0.0})
    fx_raise(torch.ops.aten.exponential.default, [big], {"lambd": -0.5})
    fx_raise(torch.ops.aten.geometric.default, [big], {"p": 0.0})
    fx_raise(torch.ops.aten.geometric.default, [big], {"p": 1.0})
    fx_raise(torch.ops.aten.geometric.default, [big], {"p": 1.5})
    fx_raise(torch.ops.aten.cauchy.default, [big], {"sigma": 0.0})
    fx_raise(torch.ops.aten.log_normal.default, [big], {"std": 0.0})
    fx_raise(torch.ops.aten.log_normal.default, [big], {"std": -1.0})
    fx_raise(torch.ops.aten.normal.Tensor_float, [big, -0.5])
    st_bad = torch.full((20000,), 0.5)
    st_bad[0] = -1.0
    fx_raise(torch.ops.aten.normal.Tensor_Tensor, [big, st_bad])
    fx_raise(torch.ops.aten.uniform.default,
             [torch.zeros(4, dtype=torch.int32)])
    print("test_random_family PASS")


def test_sub_div():
    _provenance()
    # PT-U8: aten.sub.default/div.default/true_divide.Scalar (+ mul.default,
    # PAR-41 rank 11, 17,208 receipts) had specs yet missed the table on the
    # census's sampled forms — three separate binding defects fixed here:
    # scalar-only args never tripped the has_mx gate, an alpha kwarg (or
    # positional alpha, which the old *k spec silently dropped) broke the
    # lambdas, and the "true_divide" opname had no ATEN_TABLE entry.
    x32 = torch.tensor([[3.0, -1.5], [0.5, 4.0]])
    y32 = torch.tensor([[1.0, 2.5], [-2.0, 0.25]])
    i64 = torch.tensor([[2, -3], [4, 1]], dtype=torch.int64)

    def fx_fn(target, args, ins, kwargs=None):
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        it = iter(phs)
        a = tuple(next(it) if torch.is_tensor(v) else v for v in args)
        n = g.call_function(target, a, kwargs or {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        return run(*ins), stats

    def check(name, target, args, ins, ref, kwargs=None):
        out, stats = fx_fn(target, args, ins, kwargs)
        assert stats["cpu"] == [], (name, stats)
        assert name in stats["mlx"], (name, stats)
        out = torch.as_tensor(out).float()
        ref = torch.as_tensor(ref).float()
        assert np.allclose(_np(out), ref.numpy(), atol=2e-2), \
            (name, _np(out), ref)

    # tensor forms incl. alpha both ways (positional alpha was silently
    # dropped by the old (a, b, *k) spec; kwarg alpha TypeError'd it)
    check("sub", torch.ops.aten.sub.Tensor, (x32, y32), [x32, y32],
          torch.sub(x32, y32))
    check("sub", torch.ops.aten.sub.Tensor, (x32, y32, 2), [x32, y32],
          torch.sub(x32, y32, alpha=2))
    check("sub", torch.ops.aten.sub.Tensor, (x32, y32), [x32, y32],
          torch.sub(x32, y32, alpha=2), kwargs={"alpha": 2})
    check("sub", torch.ops.aten.sub.Scalar, (x32, 1.5), [x32],
          torch.sub(x32, 1.5))
    # alpha*other computes in the PROMOTED dtype: f16 40000 * 2 must yield
    # f32 80000, not an f16 intermediate that overflows to inf (codex P2).
    z32 = torch.zeros(2, dtype=torch.float32)
    big16 = torch.full((2,), 40000.0, dtype=torch.float16)
    check("add", torch.ops.aten.add.Tensor, (z32, big16, 2), [z32, big16],
          torch.add(z32, big16, alpha=2))
    check("add", torch.ops.aten.add.Tensor, (x32, y32), [x32, y32],
          torch.add(x32, y32, alpha=2), kwargs={"alpha": 2})
    check("add", torch.ops.aten.add.Scalar, (x32, 0.5), [x32],
          torch.add(x32, 0.5))
    # int tensor + int alpha lowers (integral alpha is legal)
    check("sub", torch.ops.aten.sub.Tensor, (i64, i64, 2), [i64, i64],
          torch.sub(i64, i64, alpha=2))

    # div family: div.Tensor, div.Scalar, true_divide.{Scalar,Tensor}
    check("div", torch.ops.aten.div.Tensor, (x32, y32), [x32, y32],
          torch.div(x32, y32))
    check("div", torch.ops.aten.div.Scalar, (x32, 2.0), [x32],
          torch.div(x32, 2.0))
    check("true_divide", torch.ops.aten.true_divide.Scalar, (x32, 2.0),
          [x32], torch.div(x32, 2.0))
    check("true_divide", torch.ops.aten.true_divide.Tensor, (x32, y32),
          [x32, y32], torch.div(x32, y32))

    # scalar-only args: no mx input, previously gated to CPU
    check("mul", torch.ops.aten.mul.default, (2, 3), [],
          torch.ops.aten.mul.default(2, 3))
    check("sub", torch.ops.aten.sub.default, (2, 3), [],
          torch.ops.aten.sub.default(2, 3))
    check("div", torch.ops.aten.div.default, (2, 3), [],
          torch.ops.aten.div.default(2, 3))

    # method path: t.sub(x, alpha=2) routes through _run_method -> spec
    g = torch.fx.Graph()
    p1, p2 = g.placeholder("p1"), g.placeholder("p2")
    n = g.call_method("sub", (p1, p2), {"alpha": 2})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(x32, y32)
    assert stats["cpu"] == [], stats
    assert np.allclose(_np(out), torch.sub(x32, y32, alpha=2).numpy())

    # parity on torch's rejections: float alpha on integral operands and
    # bool subtraction must raise, matching torch's errors
    for mk in (lambda: torch.sub(i64, i64, alpha=2.5),
               lambda: fx_fn(torch.ops.aten.sub.Tensor, (i64, i64, 2.5),
                             [i64, i64])):
        try:
            mk()
            raise AssertionError("float alpha on integral inputs accepted")
        except RuntimeError:
            pass
    bx = torch.tensor([True, False])
    by = torch.tensor([True, True])
    for mk in (lambda: torch.sub(bx, by),
               lambda: fx_fn(torch.ops.aten.sub.Tensor, (bx, by), [bx, by])):
        try:
            mk()
            raise AssertionError("bool subtraction accepted")
        except RuntimeError:
            pass
    print("test_sub_div PASS")


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
    test_addcmul_addcdiv()
    test_sub_div()
    test_remainder()
    test_hyperbolic()
    test_random_family()
    print("ALL PASS")


