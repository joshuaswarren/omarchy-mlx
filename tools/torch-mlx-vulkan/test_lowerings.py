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


def test_mode():
    rng = np.random.RandomState(1)
    # fp32 with duplicates: value must match torch.mode, index must point at
    # a position holding the mode value (torch's exact duplicate pick is not
    # stable-defined; the value and the position's value are the contract)
    x = rng.randn(4, 9).astype(np.float32)
    x[0, 0] = x[0, 1] = x[0, 2] = 0.5
    x[1, :] = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0]  # tie 3/3 -> smallest 2.0
    for dim in (1, 0, -1):
        vals, idx = mlx_vulkan._mode(mx.array(x), dim)
        tv, ti = torch.mode(torch.tensor(x), dim)
        assert np.allclose(_np(vals), tv.numpy()), (dim, _np(vals), tv.numpy())
        got = _np(idx)
        # every returned index must point at a position holding the mode value
        deff = dim % x.ndim
        gax = got.reshape([1 if i == deff else got.size for i in range(x.ndim)])
        held = np.take_along_axis(x, gax, axis=deff)
        assert np.array_equal(held.reshape(-1), np.broadcast_to(
            tv.reshape(-1).numpy(), held.size)), (dim, held.reshape(-1))
    # int64
    xi = rng.randint(0, 4, size=(3, 11)).astype(np.int64)
    vi, ii = mlx_vulkan._mode(mx.array(xi), 1)
    tvi, tii = torch.mode(torch.tensor(xi), 1)
    assert np.array_equal(_np(vi), tvi.numpy())
    assert np.array_equal(_np(ii), tii.numpy()), (_np(ii), tii.numpy())
    # 1-D, keepdim both ways
    x1 = np.array([7, 7, 7, 1, 2], dtype=np.int64)
    v1, i1 = mlx_vulkan._mode(mx.array(x1), keepdim=True)
    tv1, ti1 = torch.mode(torch.tensor(x1), keepdim=True)
    assert v1.shape == tv1.shape and np.array_equal(_np(v1), tv1.numpy())
    v2, i2 = mlx_vulkan._mode(mx.array(x1), 0, keepdim=False)
    assert v2.shape == torch.mode(torch.tensor(x1), 0).values.shape
    assert int(_np(v2)) == 7
    # 3-D middle dim
    x3 = rng.randint(0, 3, size=(2, 5, 3)).astype(np.int64)
    v3, i3 = mlx_vulkan._mode(mx.array(x3), 1)
    tv3, ti3 = torch.mode(torch.tensor(x3), 1)
    assert np.array_equal(_np(v3), tv3.numpy())
    assert np.array_equal(_np(i3), ti3.numpy())
    print("test_mode PASS")


def test_kthvalue():
    rng = np.random.RandomState(2)
    # fp32 distinct values: value and index must match exactly
    x = rng.randn(4, 9).astype(np.float32)
    for k, dim in ((1, 1), (3, 1), (5, 1), (2, 0), (1, 0), (4, -1)):
        vals, idx = mlx_vulkan._kthvalue(mx.array(x), k, dim)
        tv, ti = torch.kthvalue(torch.tensor(x), k, dim)
        assert np.allclose(_np(vals), tv.numpy()), (k, dim)
        # torch CPU's kthvalue index is a quickselect artifact (it can point
        # at a different element than the returned value, even for distinct
        # inputs), so the index is verified against the value contract: the
        # element at the returned index must equal the returned k-th value
        deff = dim % x.ndim
        shape = [1] * x.ndim
        free = [i for i in range(x.ndim) if i != deff][0]
        shape[free] = _np(idx).size
        gax = _np(idx).reshape(shape)
        held = np.take_along_axis(x, gax, axis=deff)
        ref = np.broadcast_to(_np(vals).reshape(shape).astype(np.float32), held.shape)
        assert np.array_equal(held, ref), (k, dim, held, ref)
    # int64 with duplicates: the k-th smallest VALUE must match torch
    # exactly. torch CPU's index is a quickselect artifact and does not even
    # point at the returned value on duplicates (x row 0, k=1: value -2.13 at
    # index 2, torch's index 7 holds -1.24), so the lowering's index is
    # verified against the value contract instead: the element at the
    # returned index must equal the returned k-th value.
    xi = np.array([[3, 1, 3, 2, 3, 1, 2], [5, 5, 5, 1, 1, 2, 2]], dtype=np.int64)
    for k in (1, 2, 3, 4, 5, 6):
        vals, idx = mlx_vulkan._kthvalue(mx.array(xi), k, 1)
        tv, ti = torch.kthvalue(torch.tensor(xi), k, 1)
        assert np.array_equal(_np(vals), tv.numpy()), (k, _np(vals), tv.numpy())
        gshape = [1] * xi.ndim
        gshape[1 - 1] = _np(idx).size
        held = np.take_along_axis(xi, _np(idx).reshape(gshape), axis=1)
        assert np.array_equal(held.reshape(-1), np.asarray(_np(vals)).reshape(-1)), (k, held.reshape(-1), _np(vals))
    # 1-D + keepdim
    x1 = np.array([9, 4, 7, 1, 8, 2], dtype=np.int64)
    v1, i1 = mlx_vulkan._kthvalue(mx.array(x1), 3, 0, keepdim=True)
    tv1, ti1 = torch.kthvalue(torch.tensor(x1), 3, 0, keepdim=True)
    assert v1.shape == tv1.shape
    assert np.array_equal(_np(v1), tv1.numpy())
    assert int(_np(x1)[_np(i1)[0]]) == int(_np(v1)[0])
    # 3-D middle dim
    x3 = rng.randn(2, 6, 3).astype(np.float32)
    v3, i3 = mlx_vulkan._kthvalue(mx.array(x3), 2, 1)
    tv3, ti3 = torch.kthvalue(torch.tensor(x3), 2, 1)
    assert np.allclose(_np(v3), tv3.numpy())
    shape3 = list(x3.shape)
    shape3[1] = 1
    held = np.take_along_axis(x3, _np(i3).reshape(shape3), axis=1)
    assert np.array_equal(held.reshape(-1), np.asarray(_np(v3)).reshape(-1))
    print("test_kthvalue PASS")


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


def test_remainder():
    # PT-AN: aten.remainder.default is a receipt-zero PAR-41 census row
    # (elementwise modulo for quantization / zigzag index math). Prove the
    # eager executor lowers remainder/remainder.Scalar (stats["mlx"]) and
    # matches torch CPU at fp32, bf16, and int64 on negative operands:
    # torch.remainder takes the divisor's sign, the same numpy-style
    # contract mx.remainder implements.
    x32 = torch.tensor([[7.0, -7.0], [7.5, -7.5]])
    y32 = torch.tensor([[3.0, 3.0], [-2.0, -2.0]])
    xb, yb = x32.bfloat16(), y32.bfloat16()
    i64 = torch.tensor([[7, -7], [7, -7]], dtype=torch.int64)
    j64 = torch.tensor([[3, 3], [-4, -4]], dtype=torch.int64)
    cases = [
        (torch.ops.aten.remainder.default, (x32, y32), None, torch.remainder(x32, y32)),
        (torch.ops.aten.remainder.default, (xb, yb), None, torch.remainder(xb, yb)),
        (torch.ops.aten.remainder.Scalar, (x32,), -2.0, torch.remainder(x32, -2.0)),
        (torch.ops.aten.remainder.Scalar, (i64,), 3, torch.remainder(i64, 3)),
        (torch.ops.aten.remainder.default, (i64, j64), None, torch.remainder(i64, j64)),
        (torch.remainder, (x32, y32), None, torch.remainder(x32, y32)),
        # scalar-only call (aten.remainder.default(7, 3)): torch returns the
        # int64 scalar 1; boxing must not leave an int32 mx array behind.
        (torch.ops.aten.remainder.default, (7, 3), None,
         torch.ops.aten.remainder.default(7, 3)),
    ]
    for target, ins, scalar, ref in cases:
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        args = tuple(phs) if scalar is None else (phs[0], scalar)
        n = g.call_function(target, args, {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        ref = torch.as_tensor(ref)
        assert stats["cpu"] == [], (stats, [str(t.target) for t in gm.graph.nodes])
        assert any("remainder" in s for s in stats["mlx"]), stats
        got = out.float().numpy()
        assert np.allclose(got, ref.float().numpy(), atol=1e-5), (str(target), got)
        assert out.dtype == ref.dtype or (
            ref.dtype == torch.bfloat16 and out.dtype == torch.float32), \
            (str(target), out.dtype, ref.dtype)
        if ref.dtype == torch.bfloat16:
            low = mlx_vulkan._FUNC_SPECS["remainder"](mlx_vulkan._mx(xb), mlx_vulkan._mx(yb))
            assert low.dtype == mx.bfloat16, low.dtype
    print("test_remainder PASS")



def test_xlogy_xlog1py():
    # PT-AL-h: aten.xlogy/xlog1py (KL-divergence / custom-loss paths):
    # self * log(other) / self * log1p(other) with aten's NaN-first rule —
    # NaN in either operand gives NaN even when self == 0; otherwise
    # self == 0 gives 0 for every other, including other <= 0 where the
    # product form is 0 * -inf. fp32 + bf16 parity vs torch CPU on zero,
    # negative and NaN operands.
    x = torch.tensor([[0.0, 2.0], [-3.0, 0.5], [0.0, -1.0], [0.0, float("nan")]])
    y = torch.tensor([[4.0, 0.0], [2.0, -1.0], [-2.0, 0.25], [float("nan"), 2.0]])
    xb, yb = x.bfloat16(), y.bfloat16()
    cases = [
        (torch.ops.aten.xlogy.Tensor, (x, y), torch.xlogy(x, y), "xlogy", None),
        (torch.ops.aten.xlogy.Tensor, (xb, yb), torch.xlogy(xb, yb), "xlogy", (xb, yb)),
        (torch.xlogy, (x, y), torch.xlogy(x, y), "xlogy", None),
        (torch.ops.aten.xlogy.Scalar_Self, (y,), torch.ops.aten.xlogy.Scalar_Self(0.0, y), "xlogy", (0.0, y)),
        (torch.ops.aten.special_xlog1py.default, (x, y), torch.special.xlog1py(x, y), "xlog1py", None),
        (torch.ops.aten.special_xlog1py.default, (xb, yb), torch.special.xlog1py(xb, yb), "xlog1py", (xb, yb)),
        (torch.special.xlog1py, (x, y), torch.special.xlog1py(x, y), "xlog1py", None),
        (torch.ops.aten.special_xlog1py.self_scalar, (y,), torch.ops.aten.special_xlog1py.self_scalar(0.0, y), "xlog1py", (0.0, y)),
    ]
    for target, ins, ref, name, low_in in cases:
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        # the *_scalar overloads carry the scalar in the op call, not a placeholder
        args = (0.0, phs[0]) if ("Scalar_Self" in str(target) or "self_scalar" in str(target)) else tuple(phs)
        n = g.call_function(target, args, {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        ref = torch.as_tensor(ref)
        assert stats["cpu"] == [], (stats, [str(t.target) for t in gm.graph.nodes])
        assert any(name in s for s in stats["mlx"]), stats
        got = out.float().numpy()
        tol = 2e-2 if ref.dtype == torch.bfloat16 else 1e-5
        assert np.allclose(got, ref.float().numpy(), atol=tol,
                           equal_nan=True), (str(target), got)
        assert out.dtype == ref.dtype or (
            ref.dtype == torch.bfloat16 and out.dtype == torch.float32), \
            (str(target), out.dtype, ref.dtype)
        if ref.dtype == torch.bfloat16:
            low = mlx_vulkan._FUNC_SPECS[name](
                *[mlx_vulkan._mx(v) for v in low_in])
            assert low.dtype == mx.bfloat16, low.dtype
    print("test_xlogy_xlog1py PASS")

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


def test_tanh_backward():
    # PT-U4: aten.tanh_backward.default (PAR-41 census rank 29, 275 receipts).
    # The aten signature is tanh_backward(grad_output, OUTPUT) — the forward
    # result y=tanh(x), not x — so grad * (1 - y^2).
    rng = np.random.RandomState(4)
    x = rng.randn(6, 8).astype(np.float32) * 4.0  # saturating inputs: y -> +-1
    dy = rng.randn(6, 8).astype(np.float32)
    # fp32: exact formula vs the aten reference.
    y = np.tanh(x)
    ref = torch.ops.aten.tanh_backward.default(torch.tensor(dy), torch.tensor(y))
    out = mlx_vulkan.ATEN_TABLE["tanh_backward"](mx.array(dy), mx.array(y))
    assert np.abs(_np(out) - ref.numpy()).max() < 1e-5
    # saturated edge: y=+-1 exactly must give grad*0, not a fp artifact.
    ys = np.array([[-1.0, 1.0], [1.0, -1.0]], dtype=np.float32)
    ds = np.array([[2.0, -3.0], [0.5, 4.0]], dtype=np.float32)
    ref_s = torch.ops.aten.tanh_backward.default(torch.tensor(ds), torch.tensor(ys))
    got_s = mlx_vulkan.ATEN_TABLE["tanh_backward"](mx.array(ds), mx.array(ys))
    assert np.array_equal(_np(got_s), ref_s.numpy()), (_np(got_s), ref_s.numpy())
    # bf16: aten runs the formula in fp32 opmath and rounds to bf16.
    yb = torch.tensor(y, dtype=torch.bfloat16)
    gb = torch.tensor(dy, dtype=torch.bfloat16)
    ref_b = torch.ops.aten.tanh_backward.default(gb, yb)
    mine = mlx_vulkan.ATEN_TABLE["tanh_backward"](
        mx.array(gb.float().numpy()).astype(mx.bfloat16),
        mx.array(yb.float().numpy()).astype(mx.bfloat16))
    assert mine.dtype == mx.bfloat16, mine.dtype
    got_b = torch.from_numpy(np.array(mine.astype(mx.float32)))
    d = float((got_b - ref_b.float()).abs().max())
    assert d < 0.02, f"bf16: {d:.3e}"
    # executor path: the aten.tanh_backward.default node must land in
    # stats["mlx"], not the CPU fallback.
    gfx = torch.fx.Graph()
    ph_g, ph_y = gfx.placeholder("g"), gfx.placeholder("y")
    gfx.output(gfx.call_function(torch.ops.aten.tanh_backward.default, (ph_g, ph_y)))
    gm = torch.fx.GraphModule(torch.nn.Module(), gfx)
    run, stats = mlx_vulkan.make_executor(gm)
    out_e = run(torch.tensor(dy), torch.tensor(y))
    assert np.abs(_np(out_e) - ref.numpy()).max() < 1e-5
    assert "tanh_backward" in stats["mlx"], stats
    assert "tanh_backward" not in stats["cpu"], stats
    print("test_tanh_backward PASS")


def test_comparisons():
    # PT-U7: the comparison family — aten .default overloads are the
    # (Scalar, Scalar) -> bool forms, which hit the has_mx gate with no mx
    # operand (same miss class as PT-U8's scalar arith). The .Tensor and
    # .Scalar overloads lower via the tensor operand already.
    rng = np.random.RandomState(5)
    a = rng.randn(4, 6).astype(np.float32)
    b = rng.randn(4, 6).astype(np.float32)
    OPS = [
        ("eq", torch.ops.aten.eq.Tensor, mx.equal),
        ("ne", torch.ops.aten.ne.Tensor, mx.not_equal),
        ("lt", torch.ops.aten.lt.Tensor, mx.less),
        ("le", torch.ops.aten.le.Tensor, mx.less_equal),
        ("gt", torch.ops.aten.gt.Tensor, mx.greater),
        ("ge", torch.ops.aten.ge.Tensor, mx.greater_equal),
    ]
    # elementwise Tensor forms: bool dtype + value parity vs aten
    for name, aten_op, _ in OPS:
        ref = aten_op(torch.tensor(a), torch.tensor(b))
        got = mlx_vulkan.ATEN_TABLE[name](mx.array(a), mx.array(b))
        assert got.dtype == mx.bool_, (name, got.dtype)
        assert np.array_equal(_np(got), ref.numpy()), name
    # Scalar overloads: (Scalar, Scalar) -> bool through the executor must
    # lower despite no tensor operand (the _ALWAYS_LOWER fix).
    for name, aten_default, _ in [
            ("eq", torch.ops.aten.eq.default, None),
            ("ne", torch.ops.aten.ne.default, None),
            ("lt", torch.ops.aten.lt.default, None),
            ("le", torch.ops.aten.le.default, None),
            ("gt", torch.ops.aten.gt.default, None),
            ("ge", torch.ops.aten.ge.default, None)]:
        gfx = torch.fx.Graph()
        gfx.output(gfx.call_function(aten_default, (2, 2)))
        gm = torch.fx.GraphModule(torch.nn.Module(), gfx)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run()
        exp = aten_default(2, 2)
        assert bool(torch.as_tensor(out)) == exp, (name, out, exp)
        assert name in stats["mlx"], (name, stats)
        assert name not in stats["cpu"], (name, stats)
    # .Scalar (Tensor, Scalar) forms: mixed dtype must still lower via the
    # tensor operand, bool result parity.
    for name, aten_scalar in [
            ("eq", torch.ops.aten.eq.Scalar), ("ne", torch.ops.aten.ne.Scalar),
            ("lt", torch.ops.aten.lt.Scalar), ("le", torch.ops.aten.le.Scalar),
            ("gt", torch.ops.aten.gt.Scalar), ("ge", torch.ops.aten.ge.Scalar)]:
        ref = aten_scalar(torch.tensor(a), 0.5)
        got = mlx_vulkan.ATEN_TABLE[name](mx.array(a), mx.array(0.5))
        assert got.dtype == mx.bool_, name
        assert np.array_equal(_np(got), ref.numpy()), name
    # executor path on a Tensor pair: lands in stats["mlx"]
    gfx = torch.fx.Graph()
    pa, pb = gfx.placeholder("a"), gfx.placeholder("b")
    gfx.output(gfx.call_function(torch.ops.aten.eq.Tensor, (pa, pb)))
    gm = torch.fx.GraphModule(torch.nn.Module(), gfx)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(torch.tensor(a), torch.tensor(b))
    assert np.array_equal(_np(out), torch.eq(torch.tensor(a), torch.tensor(b)).numpy())
    assert "eq" in stats["mlx"], stats
    assert "eq" not in stats["cpu"], stats
    print("test_comparisons PASS")


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
    # PAR-41 census rerun @d08a7e37f: aten.linear's 1-D-input OpInfo sample
    # decomposes to t + unsqueeze + mm + squeeze_.dim - the missing squeeze_
    # spec sent that node to CPU, counting `linear` as the census's top
    # fallback. The whole 1-D graph must lower end to end.
    t1, w1 = torch.randn(8), torch.randn(6, 8)
    g = torch.fx.Graph()
    pw, px = g.placeholder("w"), g.placeholder("x")
    n_t = g.call_function(torch.ops.aten.t.default, (pw,), {})
    n_u = g.call_function(torch.ops.aten.unsqueeze.default, (px, 0), {})
    n_mm = g.call_function(torch.ops.aten.mm.default, (n_u, n_t), {})
    n_sq = g.call_function(torch.ops.aten.squeeze_.dim, (n_mm, 0), {})
    g.output(n_sq)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(w1, t1)
    assert stats["cpu"] == [], stats
    exp = torch.nn.functional.linear(t1, w1)
    assert np.allclose(_np(out), exp.numpy(), atol=1e-4)
    print("test_linear PASS")


def test_squeeze_inplace():
    # squeeze_ covers aten.squeeze_.default / .dim / .dims via _opname's
    # first-component mapping; the executor's inplace rebinding gives the
    # input node the squeezed value, matching torch's in-place semantics.
    x = torch.randn(1, 4, 1, 3)
    for target, args in ((torch.ops.aten.squeeze_.default, ()),
                         (torch.ops.aten.squeeze_.dim, (0,)),
                         (torch.ops.aten.squeeze_.dim, (2,))):
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        n = g.call_function(target, (ph,) + args, {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(x)
        assert stats["cpu"] == [], (target, stats)
        exp = x.clone()
        target(exp, *args) if args else exp.squeeze_()
        assert np.allclose(_np(out), exp.numpy(), atol=0)
    # squeeze.dims takes an int[] dim list positionally
    g = torch.fx.Graph()
    ph = g.placeholder("x")
    n = g.call_function(torch.ops.aten.squeeze.dims, (ph, [0, 2]), {})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    out = run(x)
    assert stats["cpu"] == [], stats
    assert np.allclose(_np(out), x.squeeze().numpy(), atol=0)
    print("test_squeeze_inplace PASS")


def test_clamp_bounds():
    # PAR-41 census rerun (w7R, true main): clamp_min.clamp_max had no specs
    # (plain FALLBACK). Scalar bounds pass python numbers, .Tensor bounds pass
    # broadcastable tensors; the in-place variants share the lowerings.
    x = torch.randn(4, 5)
    lo = torch.full((5,), -0.5)
    hi = torch.full((5,), 0.5)
    cases = (
        (torch.ops.aten.clamp_min.default, (x,), {"min": 0.2},
         torch.clamp_min(x, 0.2)),
        (torch.ops.aten.clamp_min.Tensor, (x, lo), {},
         torch.clamp_min(x, lo)),
        (torch.ops.aten.clamp_max.default, (x,), {"max": -0.2},
         torch.clamp_max(x, -0.2)),
        (torch.ops.aten.clamp_max.Tensor, (x, hi), {},
         torch.clamp_max(x, hi)),
        (torch.ops.aten.clamp_min_.default, (x,), {"min": 0.0},
         x.clone().clamp_min_(0.0)),
        (torch.ops.aten.clamp_max_.default, (x,), {"max": 0.0},
         x.clone().clamp_max_(0.0)),
        (torch.ops.aten.clamp.Tensor, (x, lo, hi), {},
         torch.clamp(x, lo, hi)),
    )
    for target, ins, kw, ref in cases:
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (target, stats)
        assert np.allclose(_np(out), ref.numpy(), atol=1e-6), target
    print("test_clamp_bounds PASS")


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
    # PT-U6: aten.slice.Tensor defaults every arg after self — the OpInfo
    # census hand-samples `slice(x, 0)` (dim only) and all-kwarg calls,
    # which TypeError'd against the old required-arg signature
    opt_cases = [
        ((0,), {}),                                   # dim only
        ((), {"dim": 1, "start": 1, "end": -2}),      # all kwargs
        ((), {"dim": 1, "start": 1, "end": -2, "step": 3}),
        ((), {"dim": 0, "start": -10, "end": -2, "step": 2}),
        ((), {}),                                     # every default
    ]
    for pos, kw in opt_cases:
        exp = torch.ops.aten.slice.Tensor(tx, *pos, **kw).numpy()
        got = mlx_vulkan._slice(a, *pos, **kw)
        assert np.array_equal(_np(got), exp), (pos, kw)
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
    # PT-U6: dim-only and kwargs node forms through the executor (the
    # OpInfo census's exact dispatch-miss shapes)
    for pos, kw in (((0,), {}), ((), {"dim": 1, "start": 1, "end": -2})):
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        n = g.call_function(torch.ops.aten.slice.Tensor, (ph, *pos), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(mx.array(t.numpy()))
        exp = torch.ops.aten.slice.Tensor(t, *pos, **kw).numpy()
        assert np.array_equal(_np(out), exp), (pos, kw)
        assert "slice" in stats["mlx"], stats
        assert "slice" not in stats["cpu"], stats
    print("test_slice PASS")


def test_fft():
    # PT-AU-a: aten.fft_fft / aten.fft_ifft -> mx.fft — audio-model ops
    # (Parakeet, Whisper). Parity vs torch CPU over fp32 (promotes to
    # complex64) and complex64 inputs, exercising n (pad/truncate), dim
    # and norm; then the executor must lower both aten nodes.
    rng = np.random.RandomState(17)
    xr = rng.randn(4, 6, 10).astype(np.float32)
    xc = (rng.randn(4, 6, 10) + 1j * rng.randn(4, 6, 10)).astype(np.complex64)
    for name, torch_op in (("fft_fft", torch.ops.aten.fft_fft),
                           ("fft_ifft", torch.ops.aten.fft_ifft)):
        low = mlx_vulkan._FUNC_SPECS[name]
        for x_np in (xr, xc):
            tx = torch.tensor(x_np)
            a = mx.array(x_np)
            for pos, kw in [
                ((), {}),                              # all defaults
                ((10,), {}),                           # positional n
                ((), {"n": 16}),                       # pad with zeros
                ((), {"n": 4}),                        # truncate
                ((), {"dim": 1}),                      # middle axis
                ((), {"dim": -1, "norm": "ortho"}),    # normalization
                ((), {"dim": 0, "norm": "forward"}),   # axis 0 + forward
            ]:
                exp = torch_op(tx, *pos, **kw).resolve_conj().numpy()
                got = low(a, *pos, **kw)
                assert np.allclose(_np(got), exp, rtol=2e-4, atol=2e-5), \
                    (name, x_np.dtype, pos, kw)
        t = torch.randn(4, 12)
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        n = g.call_function(torch_op, (ph,), {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(mx.array(t.numpy()))
        exp = torch_op(t).resolve_conj().numpy()
        assert np.allclose(_np(out), exp, rtol=2e-4, atol=2e-5), name
        assert name in stats["mlx"], stats
        assert name not in stats["cpu"], stats
    print("test_fft PASS")


def test_rfft():
    # PT-AU-b: aten.fft_rfft / aten.fft_irfft -> mx.fft — the
    # real<->complex transforms (Parakeet mel-spectrogram). Parity vs
    # torch CPU: rfft on fp32 input (promotes to complex64), irfft on
    # complex64 input (fp32 out), exercising n, dim and norm.
    rng = np.random.RandomState(29)
    xr = rng.randn(4, 6, 10).astype(np.float32)
    xc = (rng.randn(4, 6, 10) + 1j * rng.randn(4, 6, 10)).astype(np.complex64)
    for name, torch_op, x_np in (
            ("fft_rfft", torch.ops.aten.fft_rfft, xr),
            ("fft_irfft", torch.ops.aten.fft_irfft, xc)):
        low = mlx_vulkan._FUNC_SPECS[name]
        tx = torch.tensor(x_np)
        a = mx.array(x_np)
        for pos, kw in [
            ((), {}),                              # all defaults
            ((10,), {}),                           # positional n
            ((), {"n": 16}),                       # pad with zeros
            ((), {"n": 4}),                        # truncate
            ((), {"dim": 1}),                      # middle axis
            ((), {"dim": -1, "norm": "ortho"}),    # normalization
            ((), {"dim": 0, "norm": "forward"}),   # axis 0 + forward
        ]:
            exp = torch_op(tx, *pos, **kw).resolve_conj().numpy()
            got = low(a, *pos, **kw)
            assert np.allclose(_np(got), exp, rtol=2e-4, atol=2e-5), \
                (name, x_np.dtype, pos, kw)
        if name == "fft_rfft":
            t = torch.randn(4, 12)
        else:
            t = (torch.randn(4, 7) + 1j * torch.randn(4, 7))
def test_fft2():
    # PT-AU-c: aten.fft_fft2 / aten.fft_ifft2 -> mx.fft — the 2-D
    # transforms (vision, spectrogram models). Parity vs torch CPU over
    # fp32 and complex64 inputs, exercising s (per-axis pad/truncate),
    # dim (axis pair) and norm.
    rng = np.random.RandomState(41)
    xr = rng.randn(4, 6, 10).astype(np.float32)
    xc = (rng.randn(4, 6, 10) + 1j * rng.randn(4, 6, 10)).astype(np.complex64)
    for name, torch_op in (("fft_fft2", torch.ops.aten.fft_fft2),
                           ("fft_ifft2", torch.ops.aten.fft_ifft2)):
        low = mlx_vulkan._FUNC_SPECS[name]
        for x_np in (xr, xc):
            tx = torch.tensor(x_np)
            a = mx.array(x_np)
            for pos, kw in [
                ((), {}),                                # all defaults
                (([8, 12],), {}),                        # positional s
                ((), {"s": [3, 5]}),                     # truncate both
                ((), {"dim": [0, 1]}),                   # leading axes
                ((), {"dim": [-1, -2], "norm": "ortho"}),  # swapped axes
                ((), {"s": [4, 6], "dim": [0, 2],
                      "norm": "forward"}),               # all three
            ]:
                exp = torch_op(tx, *pos, **kw).resolve_conj().numpy()
                got = low(a, *pos, **kw)
                assert np.allclose(_np(got), exp, rtol=2e-4, atol=2e-5), \
                    (name, x_np.dtype, pos, kw)
        t = torch.randn(4, 12, 8)
        g = torch.fx.Graph()
        ph = g.placeholder("x")
        n = g.call_function(torch_op, (ph,), {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(mx.array(t.numpy()))
        exp = torch_op(t).resolve_conj().numpy()
        assert np.allclose(_np(out), exp, rtol=2e-4, atol=2e-5), name
        assert name in stats["mlx"], stats
        assert name not in stats["cpu"], stats
    print("test_rfft PASS")
    print("test_fft2 PASS")


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


def test_piecewise_activations():
    # PT-AL-b: the piecewise activation family (MobileNetV3 et al.) —
    # hardswish, hardsigmoid, hardshrink, softshrink, tanhshrink, softsign.
    # Prove the eager executor lowers each op (stats["mlx"], no CPU leg)
    # and matches torch at fp32 and bf16; the bf16 reference is computed
    # in fp32 and compared with a bf16-scale tolerance.
    x = torch.tensor([[-4.0, -2.9, -0.49, 0.0, 0.51, 3.1, 7.0],
                      [-3.2, -0.5, 0.5, 1.0, -1.0, 2.9, -7.0]])
    cases = [
        (torch.ops.aten.hardswish.default, (x,), torch.nn.functional.hardswish(x), "hardswish"),
        (torch.ops.aten.hardsigmoid.default, (x,), torch.nn.functional.hardsigmoid(x), "hardsigmoid"),
        (torch.ops.aten.hardshrink.default, (x, 0.5), torch.nn.functional.hardshrink(x, 0.5), "hardshrink"),
        (torch.ops.aten.softshrink.default, (x, 0.5), torch.nn.functional.softshrink(x, 0.5), "softshrink"),
    ]
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        for target, ins, ref, name in cases:
            ins = tuple(t.to(dtype) if torch.is_tensor(t) else t for t in ins)
            ref = ref.to(torch.float32) if dtype == torch.bfloat16 else ref
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
                               atol=tol), (name, dtype, out.float().numpy())
    # boundary pins: hardshrink/softshrink kill |x| == lambd exactly;
    # hardsigmoid saturates at x <= -3 and x >= 3.
    edge = torch.tensor([-0.5, 0.5, -3.0, 3.0])
    for target, ins, ref, name in (
            (torch.ops.aten.hardshrink.default, (edge, 0.5),
             torch.tensor([0.0, 0.0, -3.0, 3.0]), "hardshrink-edge"),
            (torch.ops.aten.hardsigmoid.default, (edge,),
             torch.tensor([2.5 / 6, 3.5 / 6, 0.0, 1.0]), "hardsigmoid-edge")):
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, name)
        assert np.allclose(out.float().numpy(), ref.numpy(), atol=1e-5), \
            (name, out.float().numpy(), ref.numpy())
    # torch 2.15 registers no aten.tanhshrink / aten.softsign dispatcher op —
    # F.* decomposes to primitives at trace time. Keep the table entries for
    # torch versions that do dispatch them, and prove parity at the table.
    F = torch.nn.functional
    for name, ref in (("tanhshrink", F.tanhshrink(x)),
                      ("softsign", F.softsign(x)),
                      ("hardswish_", F.hardswish(x)),
                      ("hardsigmoid_", F.hardsigmoid(x))):
        lowering = mlx_vulkan.ATEN_TABLE[name]
        for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
            m = mx.array(x.numpy())
            if dtype == torch.bfloat16:
                m = m.astype(mx.bfloat16)
            got = torch.from_numpy(np.array(lowering(m).astype(mx.float32)))
            assert torch.allclose(got, ref.float(), atol=tol, rtol=tol), \
                (name, dtype, (got - ref.float()).abs().max())
    print("test_piecewise_activations PASS")


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

def test_blas_accum():
    # PT-AJ: the BLAS accumulate family — aten.addmv, aten.addbmm,
    # aten.addr, aten.vdot (receipt-zero rows in the PAR-41 census).
    # Prove the eager executor lowers all four (stats["mlx"]) and matches
    # torch CPU at fp32 and bf16, including non-default beta/alpha, the
    # aten beta=0 input-drop rule (NaN payload must NOT leak), batch-sum
    # semantics for addbmm, and complex-conjugate vdot.
    rng = np.random.RandomState(11)
    inp_mv = torch.tensor(rng.randn(3).astype(np.float32))
    inp_bmm = torch.tensor(rng.randn(3, 4).astype(np.float32))
    inp_addr = torch.tensor(rng.randn(4, 6).astype(np.float32))
    mat = torch.tensor(rng.randn(3, 5).astype(np.float32))
    vec = torch.tensor(rng.randn(5).astype(np.float32))
    v1 = torch.tensor(rng.randn(4).astype(np.float32))
    v2 = torch.tensor(rng.randn(6).astype(np.float32))
    b1 = torch.tensor(rng.randn(2, 3, 5).astype(np.float32))
    b2 = torch.tensor(rng.randn(2, 5, 4).astype(np.float32))
    nanmv = inp_mv.clone()
    nanmv[1] = float("nan")

    def fx_run(target, ins, kw, atol=2e-2):
        ref = target(*ins, **kw)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target))
        raw = getattr(target, "__name__", None) or str(target)
        name = raw.split("aten.")[-1].split(".")[0]
        assert any(name in s for s in stats["mlx"]), (stats, name)
        got = torch.as_tensor(out).float().numpy()
        refn = ref.float().numpy() if torch.is_tensor(ref) else np.asarray(ref)
        assert got.shape == refn.shape, (str(target), got.shape, refn.shape)
        assert np.allclose(got, refn, atol=atol), (str(target), kw, got, refn)

    fx_run(torch.ops.aten.addmv.default, (inp_mv, mat, vec), {})
    fx_run(torch.ops.aten.addmv.default, (inp_mv, mat, vec), {"beta": 0.5, "alpha": 2.0})
    fx_run(torch.ops.aten.addmv.default, (nanmv, mat, vec), {"beta": 0})
    fx_run(torch.ops.aten.addbmm.default, (inp_bmm, b1, b2), {})
    fx_run(torch.ops.aten.addbmm.default, (inp_bmm, b1, b2), {"beta": 0.5, "alpha": 2.0})
    fx_run(torch.ops.aten.addr.default, (inp_addr, v1, v2), {})
    fx_run(torch.ops.aten.addr.default, (inp_addr, v1, v2), {"beta": 0.5, "alpha": 3.0})
    fx_run(torch.ops.aten.addr.default, (inp_addr, v1, v2), {"beta": 0})
    # PT-AK: a 0-dim input broadcasts against the outer product.
    scalar_inp = torch.tensor(2.0)
    fx_run(torch.ops.aten.addr.default, (scalar_inp, v1, v2), {})
    fx_run(torch.ops.aten.addr.default, (scalar_inp, v1, v2), {"beta": 0.5, "alpha": 2.0})
    fx_run(torch.ops.aten.vdot.default, (vec, vec), {})
    # vdot conjugates the first argument for complex inputs
    ca = torch.tensor(rng.randn(4) + 1j * rng.randn(4)).cfloat()
    cb = torch.tensor(rng.randn(4) + 1j * rng.randn(4)).cfloat()
    fx_run(torch.ops.aten.vdot.default, (ca, cb), {}, atol=2e-3)
    # bf16 dtype fidelity for this family (aten requires one dtype here):
    xb = mat.bfloat16()
    vb = vec.bfloat16()
    ib = inp_mv.bfloat16()
    fx_run(torch.ops.aten.addmv.default, (ib, xb, vb), {})
    # torch.* functional entrypoints take the same lowering
    fx_run(torch.addmv, (inp_mv, mat, vec), {})
    fx_run(torch.addr, (inp_addr, v1, v2), {})
    print("test_blas_accum PASS")

def test_group_norm():
    # PT-AL: aten.native_group_norm (receipt-zero PAR-41 row; the
    # diffusion/vision family after PT-AJ's BLAS accumulate set).
    # aten signature: (input(N,C,HxW), weight?(C), bias?(C), N, C, HxW,
    # group, eps) -> (out, mean(N,group), rstd(N,group)). Prove the eager
    # executor lowers it (stats["mlx"]) and matches torch CPU at fp32 and
    # bf16, with and without the affine weight/bias, and across group
    # counts. mx.group_norm is absent in the pinned tree, so the lowering
    # computes mean/rstd manually and reshapes.
    rng = np.random.RandomState(5)
    N, C, HxW, group = 2, 8, 16, 4
    x = torch.tensor(rng.randn(N, C, HxW).astype(np.float32))
    w = torch.tensor(rng.randn(C).astype(np.float32))
    b = torch.tensor(rng.randn(C).astype(np.float32))
    # the diffusion/vision case: 4-D NCHW input (HxW = H*W); aten
    # returns out shaped like the INPUT, mean/rstd still (N, group)
    x4 = torch.tensor(rng.randn(2, 8, 2, 8).astype(np.float32))

    def fx_run(ins, atol=2e-2):
        ref = torch.ops.aten.native_group_norm.default(*ins)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        out_node = g.call_function(torch.ops.aten.native_group_norm.default,
                                   tuple(phs), {})
        g.output(out_node)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, ins[1:])
        assert any("group_norm" in s for s in stats["mlx"]), stats
        assert len(out) == 3 and len(ref) == 3, (len(out), len(ref))
        for got_t, ref_t, name in zip(out, ref,
                                      ("out", "mean", "rstd"), strict=True):
            got = torch.as_tensor(got_t).float().numpy()
            refn = ref_t.float().numpy()
            assert got.shape == refn.shape, (name, got.shape, refn.shape)
            assert np.allclose(got, refn, atol=atol), \
                (name, got, refn)

    fx_run((x, None, None, N, C, HxW, group, 1e-5))
    fx_run((x, w, b, N, C, HxW, group, 1e-5))
    fx_run((x, w, None, N, C, HxW, group, 1e-5))
    fx_run((x, None, b, N, C, HxW, 2, 1e-5))
    # 4-D NCHW: out must come back rank-4 with the affine applied over
    # the channel axis of the real trailing dims
    fx_run((x4, w, b, 2, C, 16, group, 1e-5))
    xb = x.bfloat16()
    fx_run((xb, None, None, N, C, HxW, group, 1e-5))
    print("test_group_norm PASS")

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



def test_sum_aot_dim_input():
    # PT-B: AOT graphs functionalize the dim list into a tensor graph input.
    # The lowering must accept a tensor of dims, not only python lists.
    import numpy as np
    x32 = torch.tensor([[1.5, -2.0, 0.25], [3.0, -1.0, 2.0]])
    g = torch.fx.Graph()
    ph = g.placeholder("p0")
    dp = g.placeholder("p1")
    n = g.call_function(torch.ops.aten.sum.dim_IntList, (ph,), {"dim": dp, "keepdim": True})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    dim = torch.tensor([1])
    out = run(x32, dim)
    ref = torch.sum(x32, dim=[1], keepdim=True)
    assert stats["cpu"] == [], (stats, "aten.sum.dim_IntList fell back")
    assert np.allclose(out.float().numpy(), ref.numpy(), atol=2e-2), (out, ref)
    print("test_sum_aot_dim_input PASS")


def test_bmm_matmul():
    # PT-AE: aten.bmm.default and aten.matmul.default are the batched-GEMM
    # census rows (PAR-41: matmul/bmm carry the MLP/BERT weight). Prove the
    # eager executor lowers both overloads into stats["mlx"] and matches
    # eager at fp32 and bf16, including the 2D x 3D broadcast.
    import numpy as np
    a3 = torch.randn(2, 2, 3)
    b3 = torch.randn(2, 3, 4)
    a2 = torch.randn(3, 4)
    ab2 = torch.randn(2, 3)
    a3b, b3b = a3.bfloat16(), b3.bfloat16()
    cases = [
        (torch.ops.aten.bmm.default, (a3, b3), torch.bmm(a3, b3)),
        (torch.ops.aten.bmm.default, (a3b, b3b), torch.bmm(a3b, b3b)),
        (torch.ops.aten.matmul.default, (a3, b3), torch.matmul(a3, b3)),
        (torch.ops.aten.matmul.default, (a3b, b3b), torch.matmul(a3b, b3b)),
        (torch.ops.aten.matmul.default, (ab2, b3), torch.matmul(ab2, b3)),  # 2D x 3D broadcast
        (torch.ops.aten.matmul.default, (a3, a2), torch.matmul(a3, a2)),   # 3D x 2D broadcast
    ]
    for target, ins, ref in cases:
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target))
        assert any(k in s for s in stats["mlx"] for k in ("matmul", "bmm")), (stats, str(target))
        got = out.float().numpy()
        assert np.allclose(got, ref.float().numpy(), atol=3e-2), (str(target), got, ref.float().numpy())
        if out.dtype != ref.dtype:
            # KNOWN GAP (d-PT-AE follow-up): bf16 GEMM answers fp32; values match
            # exactly and the mx lowering is bf16 - the executor output path
            # upcasts. Tracked; do not hide it silently.
            print(f"WARN {target}: out {out.dtype} != torch {ref.dtype} (values match)")
        else:
            print(f"dtype ok {target}")
    print("test_bmm_matmul PASS")
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


def test_meshgrid():
    rng = np.random.RandomState(3)
    x = torch.tensor(rng.randn(4).astype(np.float32))
    y = torch.tensor(rng.randn(3).astype(np.float32))
    z = torch.tensor(rng.randn(2).astype(np.float32))
    i = torch.tensor([5, 1, 7, 2], dtype=torch.int64)
    h = torch.tensor(rng.randn(3).astype(np.float16))

    def fx_mesh(target, ins, kw):
        ref = target(*ins, **kw)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{j}") for j in range(len(ins))]
        n = g.call_function(target, tuple(phs), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target))
        assert any("meshgrid" in s for s in stats["mlx"]), (stats, str(target))
        outs = list(out) if isinstance(out, (list, tuple)) else [out]
        assert len(outs) == len(ref), (len(outs), len(ref), str(target))
        for o, r in zip(outs, ref):
            o = torch.as_tensor(o)
            assert tuple(o.shape) == tuple(r.shape), \
                (str(target), kw, o.shape, r.shape)
            assert str(o.dtype).split(".")[-1] == str(r.dtype).split(".")[-1], \
                (str(target), kw, str(o.dtype), str(r.dtype))
            assert np.allclose(o.float().numpy(), r.float().numpy(), atol=2e-2), \
                (str(target), kw)

    i2 = torch.tensor([3, 9], dtype=torch.int64)
    h2 = torch.tensor(rng.randn(2).astype(np.float16))
    for ins, kw in (
        ((x,), {}),
        ((x, y), {}),
        ((x, y), {"indexing": "ij"}),
        ((x, y), {"indexing": "xy"}),
        ((x, y, z), {}),
        ((x, y, z), {"indexing": "xy"}),
        ((i, i2), {}),  # same-dtype int pair
        ((h, h2), {}),  # same-dtype fp16 pair
    ):
        fx_mesh(torch.meshgrid, ins, kw)

    # same lowering via the aten overloads (single Tensor[] arg)
    def fx_aten(target, ins, kw):
        ref = target(list(ins), **kw)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{j}") for j in range(len(ins))]
        n = g.call_function(target, (list(phs),), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target))
        for o, r in zip(out, ref):
            o = torch.as_tensor(o)
            assert tuple(o.shape) == tuple(r.shape), (o.shape, r.shape)
            assert np.allclose(o.float().numpy(), r.float().numpy(), atol=2e-2)

    fx_aten(torch.ops.aten.meshgrid.default, (x, y), {})
    fx_aten(torch.ops.aten.meshgrid.indexing, (x, y), {"indexing": "xy"})
    fx_aten(torch.ops.aten.meshgrid.indexing, (x, y, z), {"indexing": "ij"})

    # aten requires same dtype; mx accepts mixed — the lowering defers
    # and the CPU fallback raises torch's own error
    g = torch.fx.Graph()
    p0, p1 = g.placeholder("p0"), g.placeholder("p1")
    n = g.call_function(torch.ops.aten.meshgrid.default, ([p0, p1],), {})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    try:
        run(i, y)
        raise AssertionError("no raise on mixed dtypes")
    except RuntimeError:
        pass
    print("test_meshgrid PASS")


def test_where_masked_fill():
    # PT-AA: aten.where.self (PAR-41 census rank 25, 900 uses, already
    # lowered) and aten.masked_fill.Tensor/Scalar/_ (registered but no
    # Tensor-variant coverage). Prove lowering + dtype promotion + parity
    # at fp32/fp16/bf16 with a shared bool-mask digest.
    cond = torch.tensor([True, False, True, False])
    x32, y32 = torch.tensor([1.0, -2.0, 3.5, -4.0]), torch.tensor([0.5, 2.0, -1.0, 4.0])
    cases = [
        (torch.ops.aten.where.self, (cond, x32, y32)),
        (torch.ops.aten.where.self, (cond, x32, torch.tensor([1, 2, 3, 4]))),
        (torch.ops.aten.where.self, (cond, x32.half(), y32)),
        (torch.ops.aten.where.self, (cond, x32.bfloat16(), y32)),
        (torch.ops.aten.where.self, (cond, torch.tensor([1, 2, 3, 4]),
                                    torch.tensor([5, 6, 7, 8], dtype=torch.int32))),
        (torch.ops.aten.masked_fill.Scalar, (x32, cond, 9.0)),
        (torch.ops.aten.masked_fill.Tensor, (x32, cond, torch.tensor(9.0))),
        (torch.ops.aten.masked_fill.Scalar, (x32.bfloat16(), cond, 2.5)),
        (torch.ops.aten.masked_fill.Scalar, (x32.half(), cond, 2.5)),
        (torch.ops.aten.masked_fill.Scalar, (torch.randn(2, 3),
                                             torch.tensor([[True], [False]]), 7.0)),
    ]
    for target, ins in cases:
        ref = target(*ins)
        g = torch.fx.Graph()
        targs = [a for a in ins if torch.is_tensor(a)]
        phs = [g.placeholder(f"p{i}") for i in range(len(targs))]
        pit = iter(phs)
        args = tuple(next(pit) if torch.is_tensor(a) else a for a in ins)
        n = g.call_function(target, args, {})
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*targs)
        assert stats["cpu"] == [], (stats, str(target))
        fam = "where" if "where" in str(target) else "masked_fill"
        assert any(fam in s for s in stats["mlx"]), (stats, str(target))
        got = torch.as_tensor(out)
        # the executor's _torch output widens bf16 -> fp32 (numpy has no
        # bf16 buffer); dtype fidelity lives at the lowering, so the bf16
        # case also checks the spec result dtype directly
        assert got.dtype == ref.dtype or (
            ref.dtype == torch.bfloat16 and got.dtype == torch.float32), \
            (str(target), got.dtype, ref.dtype)
        if ref.dtype == torch.bfloat16:
            low = mlx_vulkan._FUNC_SPECS[fam](*(
                mlx_vulkan._mx(a) if torch.is_tensor(a) else a for a in ins))
            assert low.dtype == mx.bfloat16, low.dtype
        assert torch.equal(got.float(), ref.float()), (str(target), got, ref)

    # d-review: the in-place case used to share one tensor between the torch
    # reference and the executor, so the reference mutated it first and a
    # functional (out-of-place) lowering passed unnoticed. Independent inputs,
    # and the graph returns the INPUT placeholder too: after masked_fill_ the
    # graph's own view of the input must carry the fill, matching torch
    # in-place semantics for any later node.
    ref_in = x32.clone()
    ref = torch.ops.aten.masked_fill_.Scalar(ref_in, cond, 9.0)
    exec_in = x32.clone()
    g = torch.fx.Graph()
    p0 = g.placeholder("p0")
    p1 = g.placeholder("p1")
    p2 = g.placeholder("p2")
    n = g.call_function(torch.ops.aten.masked_fill_.Scalar, (p0, p1, p2), {})
    g.output((p0, n))
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    seen, out = run(exec_in, cond, torch.tensor(9.0))
    assert stats["cpu"] == [] and any("masked_fill" in s for s in stats["mlx"]), stats
    assert torch.equal(torch.as_tensor(seen).float(), ref.float()), (
        "masked_fill_ did not mutate the graph's input placeholder: "
        f"seen={torch.as_tensor(seen)} ref={ref}"
    )
    assert torch.equal(torch.as_tensor(out).float(), ref.float())
    print("test_where_masked_fill PASS")


def test_mv_addmv():
    # PT-U9: aten.mv.default and aten.addmv.default are the matvec census
    # rows. #825's glue block added the _FUNC_SPECS entries; this pins that
    # both reach stats["mlx"] (not the torch fallback) and match eager,
    # including addmv's beta/alpha kwargs and the degenerate beta=0 arm.
    import numpy as np
    m32 = torch.randn(4, 3)
    v32 = torch.randn(3)
    b32 = torch.randn(4)
    cases = [
        (torch.ops.aten.mv.default, (m32, v32), {}, "mv"),
        (torch.ops.aten.mv.default, (m32.bfloat16(), v32.bfloat16()), {}, "mv"),
        (torch.ops.aten.mv.default, (m32.half(), v32.half()), {}, "mv"),
        (torch.ops.aten.addmv.default, (b32, m32, v32), {}, "addmv"),
        (torch.ops.aten.addmv.default, (b32, m32, v32), {"beta": 0.5, "alpha": 2.0}, "addmv"),
        (torch.ops.aten.addmv.default, (b32, m32, v32), {"beta": 0, "alpha": 1}, "addmv"),
        (torch.ops.aten.addmv.default, (b32.bfloat16(), m32.bfloat16(), v32.bfloat16()), {}, "addmv"),
    ]
    for target, ins, kwargs, fam in cases:
        ref = target(*ins, **kwargs)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), dict(kwargs))
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target), kwargs)
        # "mv" is a substring of "addmv": match the spec name exactly.
        assert fam in stats["mlx"], (stats, str(target), kwargs)
        got = torch.as_tensor(out)
        assert got.dtype == ref.dtype or (
            ref.dtype == torch.bfloat16 and got.dtype == torch.float32), (
            str(target), got.dtype, ref.dtype)
        if ref.dtype == torch.bfloat16:
            # Same dtype-fidelity contract as test_mul: _torch widens bf16
            # to fp32 at the executor boundary (numpy has no bf16 buffer);
            # the lowering itself must still produce bf16.
            low = mlx_vulkan._FUNC_SPECS[fam](*(
                mlx_vulkan._mx(a) if torch.is_tensor(a) else a for a in ins),
                **kwargs)
            assert low.dtype == mx.bfloat16, low.dtype
        assert np.allclose(got.float().numpy(), ref.float().numpy(), atol=3e-2), (
            str(target), kwargs, got, ref)
    print("test_mv_addmv PASS")


def test_softplus_mish():
    # PT-AL-c: softplus (T5, GDN gate) + mish (EfficientNet).
    # aten.softplus(x, beta=1, threshold=20) is log1p(exp(beta*x))/beta,
    # reverting to the linear x only when beta*x > threshold — strictly
    # greater, so the equality point stays on the log branch. The inputs
    # straddle that edge at the defaults (x crosses 20) and at
    # beta=2,threshold=10 (the edge moves to x=5). bf16 pins the aten
    # opmath contract: fp32 math rounded to bf16, not bf16 intermediates.
    x32 = torch.tensor([[-4.0, -1.0, 0.0], [0.001, 1.0, 5.0]])
    e32 = torch.tensor([[19.5, 20.0], [20.5, 21.0]])      # default edge: beta*x == 20
    e32b = torch.tensor([[4.5, 5.0], [5.5, 6.0]])         # beta=2 edge: beta*x == 10
    m32 = torch.tensor([[-5.0, -1.0], [0.0, 3.0]])
    xb, eb, mb = x32.bfloat16(), e32.bfloat16(), m32.bfloat16()
    cases = [
        (torch.ops.aten.softplus.default, (x32,), {},
         torch.ops.aten.softplus.default(x32, 1, 20), "softplus"),
        (torch.ops.aten.softplus.default, (e32,), {},
         torch.ops.aten.softplus.default(e32, 1, 20), "softplus"),
        (torch.ops.aten.softplus.default, (e32b,), {"beta": 2, "threshold": 10},
         torch.ops.aten.softplus.default(e32b, 2, 10), "softplus"),
        (torch.ops.aten.softplus.default, (xb,), {},
         torch.ops.aten.softplus.default(xb, 1, 20), "softplus"),
        (torch.ops.aten.softplus.default, (eb,), {},
         torch.ops.aten.softplus.default(eb, 1, 20), "softplus"),
        (torch.nn.functional.softplus, (x32,), {}, torch.nn.functional.softplus(x32), "softplus"),
        (torch.nn.functional.softplus, (x32,), {"beta": 0.5, "threshold": 40},
         torch.nn.functional.softplus(x32, beta=0.5, threshold=40), "softplus"),
        (torch.ops.aten.mish.default, (m32,), {},
         torch.ops.aten.mish.default(m32), "mish"),
        (torch.ops.aten.mish.default, (mb,), {},
         torch.ops.aten.mish.default(mb), "mish"),
        (torch.nn.functional.mish, (m32,), {}, torch.nn.functional.mish(m32), "mish"),
    ]
    for target, ins, kw, ref, name in cases:
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target), kw)
        assert any(name in s for s in stats["mlx"]), (stats, str(target))
        got = out.float().numpy()
        assert np.allclose(got, ref.float().numpy(), atol=2e-2), \
            (str(target), kw, got, ref.float().numpy())
        assert out.dtype == ref.dtype or (
            ref.dtype == torch.bfloat16 and out.dtype == torch.float32), \
            (str(target), kw, out.dtype, ref.dtype)
        if ref.dtype == torch.bfloat16:
            # Same dtype-fidelity contract as test_mul: _torch widens bf16
            # to fp32 at the executor boundary (numpy has no bf16 buffer);
            # the lowering itself must still produce bf16.
            low = mlx_vulkan._FUNC_SPECS[name](mlx_vulkan._mx(ins[0]), **kw)
            assert low.dtype == mx.bfloat16, low.dtype
    print("test_softplus_mish PASS")



def test_elu_selu_celu():
    # PT-AL-a: the elu family (ViT, Conformer, speech models).
    # aten.elu(input, alpha=1, scale=1, input_scale=1):
    #   scale*x for x > 0, else alpha*scale*(exp(input_scale*x) - 1).
    # aten.selu is elu at the published constants (scale
    # 1.0507009873554805, alpha 1.6732632423543772). aten.celu(input,
    # alpha=1) is max(0,x) + min(0, alpha*(exp(x/alpha) - 1)). The inputs
    # straddle zero and run deep negative so the exp branch saturates to
    # -alpha*scale.
    x = torch.tensor([[-8.0, -2.5, -0.3, 0.0], [0.25, 1.5, 6.0, -0.001]])
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        xd = x.to(dtype)
        cases = [
            (torch.ops.aten.elu.default, (xd,), {},
             torch.ops.aten.elu.default(xd, 1.0, 1.0, 1.0), "elu"),
            (torch.ops.aten.elu.default, (xd,), {"alpha": 2.0},
             torch.ops.aten.elu.default(xd, 2.0, 1.0, 1.0), "elu"),
            (torch.ops.aten.elu.default, (xd, 1.5, 1.2, 0.8), {},
             torch.ops.aten.elu.default(xd, 1.5, 1.2, 0.8), "elu"),
            (torch.ops.aten.selu.default, (xd,), {},
             torch.ops.aten.selu.default(xd), "selu"),
            (torch.ops.aten.celu.default, (xd,), {},
             torch.ops.aten.celu.default(xd, 1.0), "celu"),
            (torch.ops.aten.celu.default, (xd,), {"alpha": 2.5},
             torch.ops.aten.celu.default(xd, 2.5), "celu"),
            (torch.nn.functional.elu, (xd,), {"alpha": 0.5},
             torch.nn.functional.elu(xd, alpha=0.5), "elu"),
            (torch.nn.functional.selu, (xd,), {},
             torch.nn.functional.selu(xd), "selu"),
            (torch.nn.functional.celu, (xd,), {"alpha": 1.5},
             torch.nn.functional.celu(xd, alpha=1.5), "celu"),
        ]
        for target, ins, kw, ref, name in cases:
            g = torch.fx.Graph()
            phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
            n = g.call_function(target, tuple(phs), kw)
            g.output(n)
            gm = torch.fx.GraphModule(torch.nn.Module(), g)
            run, stats = mlx_vulkan.make_executor(gm)
            out = run(*ins)
            assert stats["cpu"] == [], (stats, str(target), kw)
            assert any(name in s for s in stats["mlx"]), (stats, str(target))
            got = out.float().numpy()
            assert np.allclose(got, ref.float().numpy(), atol=tol), \
                (str(target), kw, dtype, got, ref.float().numpy())
    # the in-place aten names share the table entries — parity at the table.
    F = torch.nn.functional
    for name, ref, params in (("elu_", F.elu(x, alpha=1.7), (1.7,)),
                              ("selu_", F.selu(x), ()),
                              ("celu_", F.celu(x, alpha=2.2), (2.2,))):
        lowering = mlx_vulkan.ATEN_TABLE[name]
        m = mx.array(x.numpy())
        got = torch.from_numpy(
            np.array(lowering(m, *params).astype(mx.float32)))
        assert torch.allclose(got, ref.float(), atol=1e-5), \
            (name, (got - ref.float()).abs().max())
    # exact-zero pin: all three give 0 at x = 0 under any parameters.
    z = mx.zeros((2,))
    for name in ("elu", "selu", "celu"):
        got = np.array(mlx_vulkan.ATEN_TABLE[name](z))
        assert np.all(np.array(got) == 0.0), (name, got)
    print("test_elu_selu_celu PASS")


def test_cosine_similarity():
    # PT-NORM-a: aten.cosine_similarity.default (embedding-eval family).
    # sum(x1*x2, dim) / (clamp(norm1,eps) * clamp(norm2,eps)) - eps clamps
    # each norm separately, pinned by the small-norm case below.
    import numpy as np
    a32 = torch.randn(4, 3)
    b32 = torch.randn(4, 3)
    a3d = torch.randn(2, 5, 4)
    b3d = torch.randn(2, 5, 4)
    tiny_a = torch.randn(4, 3) * 1e-3
    tiny_b = torch.randn(4, 3) * 1e-3
    cases = [
        (torch.ops.aten.cosine_similarity.default, (a32, b32), {}, "cosine_similarity"),
        (torch.ops.aten.cosine_similarity.default, (a32, b32), {"dim": 0}, "cosine_similarity"),
        (torch.ops.aten.cosine_similarity.default, (a3d, b3d), {"dim": 2}, "cosine_similarity"),
        (torch.ops.aten.cosine_similarity.default, (tiny_a, tiny_b), {"eps": 0.5}, "cosine_similarity"),
        (torch.ops.aten.cosine_similarity.default, (a32.bfloat16(), b32.bfloat16()), {}, "cosine_similarity"),
    ]
    for target, ins, kwargs, fam in cases:
        ref = target(*ins, **kwargs)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), dict(kwargs))
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target), kwargs)
        assert fam in stats["mlx"], (stats, str(target), kwargs)
        got = torch.as_tensor(out)
        assert got.dtype == ref.dtype or (
            ref.dtype == torch.bfloat16 and got.dtype == torch.float32), (
            str(target), got.dtype, ref.dtype)
        if ref.dtype == torch.bfloat16:
            # Same dtype-fidelity contract as test_mul: _torch widens bf16
            # to fp32 at the executor boundary (numpy has no bf16 buffer);
            # the lowering itself must still produce bf16.
            low = mlx_vulkan._FUNC_SPECS[fam](*(
                mlx_vulkan._mx(a) if torch.is_tensor(a) else a for a in ins),
                **kwargs)
            assert low.dtype == mx.bfloat16, low.dtype
        assert np.allclose(got.float().numpy(), ref.float().numpy(), atol=3e-2), (
            str(target), kwargs, got, ref)
    print("test_cosine_similarity PASS")


def test_glu():
    # PT-AL-d: aten.glu.default (GDN gate, Conformer).
    # aten.glu(input, dim=-1): the input halves along dim; a * sigmoid(b).
    x2 = torch.randn(3, 8)
    x3 = torch.randn(2, 6, 4)
    x1 = torch.randn(10)
    F = torch.nn.functional
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        cases = [
            (torch.ops.aten.glu.default, (x2.to(dtype),), {},
             F.glu(x2.to(dtype)), "glu"),
            (torch.ops.aten.glu.default, (x3.to(dtype),), {"dim": 0},
             F.glu(x3.to(dtype), dim=0), "glu"),
            (torch.ops.aten.glu.default, (x3.to(dtype),), {"dim": 1},
             F.glu(x3.to(dtype), dim=1), "glu"),
            (torch.ops.aten.glu.default, (x1.to(dtype),), {},
             F.glu(x1.to(dtype)), "glu"),
            (F.glu, (x2.to(dtype),), {},
             F.glu(x2.to(dtype)), "glu"),
            (F.glu, (x3.to(dtype),), {"dim": -2},
             F.glu(x3.to(dtype), dim=-2), "glu"),
        ]
        for target, ins, kw, ref, name in cases:
            g = torch.fx.Graph()
            phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
            n = g.call_function(target, tuple(phs), kw)
            g.output(n)
            gm = torch.fx.GraphModule(torch.nn.Module(), g)
            run, stats = mlx_vulkan.make_executor(gm)
            out = run(*ins)
            assert stats["cpu"] == [], (stats, str(target), kw)
            assert any(name in s for s in stats["mlx"]), (stats, str(target))
            got = out.float().numpy()
            assert np.allclose(got, ref.float().numpy(), atol=tol), \
                (str(target), kw, dtype, got, ref.float().numpy())
    print("test_glu PASS")


def test_reciprocal_frac():
    # PT-AL-e: aten.reciprocal + aten.frac (rsqrt was already lowered).
    # frac is x - trunc(x), sign-kept: frac(-2.3) == -0.3, NOT x - floor(x)
    # (+0.7) — the negative inputs below pin that contract.
    x = torch.tensor([[-2.3, -0.7, -0.5, 0.0], [0.25, 1.9, 8.0, -4.0]])
    F = torch.nn.functional
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        xd = x.to(dtype)
        cases = [
            (torch.ops.aten.reciprocal.default, (xd,), {},
             torch.ops.aten.reciprocal.default(xd), "reciprocal"),
            (torch.ops.aten.frac.default, (xd,), {},
             torch.ops.aten.frac.default(xd), "frac"),
            (torch.ops.aten.reciprocal_.default, (xd,), {},
             torch.ops.aten.reciprocal.default(xd), "reciprocal"),
            (torch.ops.aten.frac_.default, (xd,), {},
             torch.ops.aten.frac.default(xd), "frac"),
            (torch.frac, (xd,), {}, torch.frac(xd), "frac"),
            (torch.reciprocal, (xd,), {},
             torch.reciprocal(xd), "reciprocal"),
        ]
        for target, ins, kw, ref, name in cases:
            g = torch.fx.Graph()
            phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
            n = g.call_function(target, tuple(phs), kw)
            g.output(n)
            gm = torch.fx.GraphModule(torch.nn.Module(), g)
            run, stats = mlx_vulkan.make_executor(gm)
            out = run(*ins)
            assert stats["cpu"] == [], (stats, str(target), kw)
            assert any(name in s for s in stats["mlx"]), (stats, str(target))
            got = out.float().numpy()
            assert np.allclose(got, ref.float().numpy(), atol=tol), \
                (str(target), kw, dtype, got, ref.float().numpy())
    # sign pin: frac of a negative keeps the negative part.
    m = mx.array([-2.3, -0.7, 1.9])
    got = np.array(mlx_vulkan.ATEN_TABLE["frac"](m))
    assert np.allclose(got, [-0.3, -0.7, 0.9], atol=1e-6), got
    print("test_reciprocal_frac PASS")


def test_linalg_norm():
    # PT-NORM-b: aten.linalg_vector_norm + aten.linalg_norm (normalization,
    # grad clipping). Also pins aten.norm's inf/-inf contract: the prior
    # lowering silently computed a 2-norm for both.
    x = torch.randn(3, 4)
    x3 = torch.randn(2, 3, 4)
    cases = [
        (torch.ops.aten.linalg_vector_norm.default, (x,), {},
         torch.linalg.vector_norm(x), "linalg_vector_norm"),
        (torch.ops.aten.linalg_vector_norm.default, (x,), {"ord": 1.0},
         torch.linalg.vector_norm(x, ord=1.0), "linalg_vector_norm"),
        (torch.ops.aten.linalg_vector_norm.default, (x,),
         {"ord": float("inf"), "dim": [1]},
         torch.linalg.vector_norm(x, ord=float("inf"), dim=[1]),
         "linalg_vector_norm"),
        (torch.ops.aten.linalg_vector_norm.default, (x3,),
         {"ord": 0.5, "dim": [1, 2], "keepdim": True},
         torch.linalg.vector_norm(x3, ord=0.5, dim=[1, 2], keepdim=True),
         "linalg_vector_norm"),
        (torch.ops.aten.linalg_vector_norm.default, (x,),
         {"dtype": torch.float64},
         torch.linalg.vector_norm(x, dtype=torch.float64),
         "linalg_vector_norm"),
        (torch.ops.aten.linalg_norm.default, (x,), {},
         torch.linalg.norm(x), "linalg_norm"),
        (torch.ops.aten.linalg_norm.default, (x,),
         {"ord": float("-inf"), "dim": [0]},
         torch.linalg.norm(x, ord=float("-inf"), dim=[0]), "linalg_norm"),
        (torch.ops.aten.linalg_norm.default, (x3,), {"ord": 2.0, "dim": [2]},
         torch.linalg.norm(x3, ord=2.0, dim=[2]), "linalg_norm"),
    ]
    for target, ins, kw, ref, name in cases:
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), kw)
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target), kw)
        assert any(name in s for s in stats["mlx"]), (stats, str(target))
        got = out.float().numpy()
        assert np.allclose(got, ref.float().numpy(), atol=1e-5), \
            (str(target), kw, got, ref.float().numpy())
    # the aten.norm inf/-inf fix at table level.
    m = mx.array(x.numpy())
    for p, ref in (("inf", torch.norm(x, float("inf"))),
                   ("-inf", torch.norm(x, float("-inf"))),
                   (1.0, torch.norm(x, 1.0)),
                   (2.0, torch.norm(x, 2.0))):
        got = torch.from_numpy(np.array(
            mlx_vulkan.ATEN_TABLE["norm"](m, p)))
        assert torch.allclose(got, ref.float(), atol=1e-5), \
            (p, float(got), float(ref))
    print("test_linalg_norm PASS")


def test_logsumexp():
    # PT-AL-g: aten.logsumexp(self, int[1] dim, keepdim) — used in
    # contrastive learning. mx.logsumexp is the same stable reduction;
    # the dim list becomes the axis tuple.
    x2 = torch.randn(3, 4)
    x3 = torch.randn(2, 3, 4)
    cases = [
        (torch.ops.aten.logsumexp.default, (x2,), {"dim": [0]}, "logsumexp"),
        (torch.ops.aten.logsumexp.default, (x2,), {"dim": [0, 1]}, "logsumexp"),
        (torch.ops.aten.logsumexp.default, (x3,), {"dim": [2]}, "logsumexp"),
        (torch.ops.aten.logsumexp.default, (x3,),
         {"dim": [0, 2], "keepdim": True}, "logsumexp"),
        (torch.ops.aten.logsumexp.default, (x2,), {"dim": [-1]}, "logsumexp"),
        (torch.ops.aten.logsumexp.default, (x2.bfloat16(),), {"dim": [1]},
         "logsumexp"),
        # numerically hostile: large logits would overflow a naive
        # log(sum(exp)) — the stable path must match torch.
        (torch.ops.aten.logsumexp.default, (x2 + 1e4,), {"dim": [1]},
         "logsumexp"),
    ]
    for target, ins, kwargs, fam in cases:
        ref = target(*ins, **kwargs)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), dict(kwargs))
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target), kwargs)
        assert fam in stats["mlx"], (stats, str(target), kwargs)
        got = torch.as_tensor(out)
        tol = 2e-2 if ins[0].dtype == torch.bfloat16 else 1e-5
        assert np.allclose(got.float().numpy(), ref.float().numpy(), atol=tol), (
            str(target), kwargs, got, ref)
    print("test_logsumexp PASS")


def test_pdist_cdist():
    # PT-PAIR-a: aten.pdist + aten.cdist (pairwise distance, metric
    # learning). Composed from broadcast-difference norms; the pdist
    # condense order is built arithmetically (mx has no compress).
    a = torch.randn(5, 3)
    b = torch.randn(4, 3)
    ab = torch.randn(2, 5, 3)
    bb = torch.randn(2, 4, 3)
    cases = [
        (torch.ops.aten.cdist.default, (a, b), {}, "cdist"),
        (torch.ops.aten.cdist.default, (a, b), {"p": 1.0}, "cdist"),
        (torch.ops.aten.cdist.default, (a, b), {"p": 0.5}, "cdist"),
        (torch.ops.aten.cdist.default, (a, b), {"p": float("inf")}, "cdist"),
        (torch.ops.aten.cdist.default, (ab, bb), {}, "cdist"),
        (torch.ops.aten.pdist.default, (a,), {}, "pdist"),
        (torch.ops.aten.pdist.default, (a,), {"p": 1.0}, "pdist"),
        (torch.ops.aten.pdist.default, (a,), {"p": float("inf")}, "pdist"),
    ]
    for target, ins, kwargs, fam in cases:
        ref = target(*ins, **kwargs)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), dict(kwargs))
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target), kwargs)
        assert any(fam in s for s in stats["mlx"]), (stats, str(target), kwargs)
        got = torch.as_tensor(out)
        assert np.allclose(got.float().numpy(), ref.float().numpy(), atol=1e-5), (
            str(target), kwargs, got, ref)
    print("test_pdist_cdist PASS")


def test_log1p_expm1_logaddexp():
    # PT-AL-f: the dev table must match the wheel table — aten.expm1,
    # aten.logaddexp lower with fp32/bf16 parity including near-zero
    # inputs where the naive forms lose precision.
    torch.manual_seed(0)
    x = torch.tensor([0.0, 1e-9, -1e-9, 1e-7, -1e-7, 1e-4, -1e-4,
                      1e-2, -1e-2, -0.5, 0.5, 1.0, -0.9999, 2.0])
    y = torch.tensor([0.0, 1e-9, -1e-9, -20.0, 20.0, 1e-4, -1.0,
                      0.5, -0.5, 3.0, -3.0, 10.0, -10.0, 0.001])
    cases = [
        (torch.ops.aten.log1p.default, (x,), {}, "log1p"),
        (torch.ops.aten.expm1.default, (x,), {}, "expm1"),
        (torch.ops.aten.expm1_.default, (x,), {}, "expm1"),
        (torch.ops.aten.logaddexp.default, (x, y), {}, "logaddexp"),
        (torch.ops.aten.expm1.default, (x.bfloat16(),), {}, "expm1"),
        (torch.ops.aten.logaddexp.default,
         (x.bfloat16(), y.bfloat16()), {}, "logaddexp"),
    ]
    for target, ins, kwargs, fam in cases:
        ref = target(*[a.clone() for a in ins], **kwargs)
        g = torch.fx.Graph()
        phs = [g.placeholder(f"p{i}") for i in range(len(ins))]
        n = g.call_function(target, tuple(phs), dict(kwargs))
        g.output(n)
        gm = torch.fx.GraphModule(torch.nn.Module(), g)
        run, stats = mlx_vulkan.make_executor(gm)
        out = run(*ins)
        assert stats["cpu"] == [], (stats, str(target), kwargs)
        assert any(fam in s for s in stats["mlx"]), (stats, str(target), kwargs)
        got = torch.as_tensor(out)
        tol = 2e-2 if ins[0].dtype == torch.bfloat16 else 1e-5
        assert np.allclose(got.float().numpy(), ref.float().numpy(), atol=tol), (
            str(target), got, ref)
    print("test_log1p_expm1_logaddexp PASS")


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
    test_remainder()
    test_pcache_skips_static_weights()
    test_compiled_executor_nonarray_outputs()
    test_gelu_backward()
    test_tanh_backward()
    test_comparisons()
    test_linear()
    test_dispatch_registered()
    test_numel()
    test_slice()
    test_fft()
    test_rfft()
    test_fft2()
    test_transpose_expand()
    test_cat_clone()
    test_sum()
    test_reductions()
    test_bitwise_and_()
    test_arc_trig()
    test_addcmul_addcdiv()
    test_sub_div()
    test_hyperbolic()
    test_random_family()
    test_sum_aot_dim_input()
    test_bmm_matmul()
    test_blas_accum()
    test_meshgrid()
    test_group_norm()
    test_where_masked_fill()
    test_mv_addmv()
    test_xlogy_xlog1py()
    test_softplus_mish()
    test_elu_selu_celu()
    test_log1p_expm1_logaddexp()
    test_logsumexp()
    test_glu()
    test_reciprocal_frac()
    test_linalg_norm()
    test_cosine_similarity()
    test_pdist_cdist()
    print("ALL PASS")


