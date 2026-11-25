#!/usr/bin/env python3
"""PAR-41 batch-norm parity: native_batch_norm, batch_norm and
_native_batch_norm_legit_functional vs torch CPU, train and eval modes,
fp32 and bf16, plus the backward - all three entries exercised in ONE
process, in sequence (that is how a torch.compile session runs them).

Contracts (verified against torch CPU; see w7X's review of the earlier
draft):
- native_batch_norm returns the TUPLE (out, save_mean, save_invstd);
  slot 3 is the INVERSE STD, not the variance. Torch mutates the passed
  running stats in place; the MLX lowering is pure - updated stats are
  returned by the _legit_functional entry instead.
- batch_norm returns the normalized tensor only.
- _native_batch_norm_legit_functional returns the 5-tuple with
  new_rm / new_rv (unbiased running-var update).
- Channel parameters broadcast as (1, C, 1, 1) on NCHW.
- Tolerances are values (2e-5 fp32 / 4e-2 bf16).

Run: python3 test_batch_norm.py  (prints PASS; exits nonzero on the
first failing check with got/want dumps).
"""
import numpy as np
import torch
import mlx.core as mx

import mlx_vulkan


def to32(a):
    """mx array -> fp32 numpy (bf16 has no PEP 3118 buffer format)."""
    return np.asarray(a.astype(mx.float32), dtype=np.float32)


def close(tag, got, want, tol):
    if hasattr(got, "astype") and not isinstance(got, np.ndarray):
        got = to32(got).astype(np.float64)
    else:
        got = np.asarray(got, dtype=np.float64)
    if hasattr(want, "detach"):
        want = want.detach().cpu().float().numpy()
    want = np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape, f"{tag}: shape {got.shape} vs {want.shape}"
    d = np.abs(got - want).max()
    assert d < tol, (f"{tag}: max diff {d:g} (tol {tol})\n"
                     f" got : {np.round(got.ravel()[:8], 4).tolist()}\n"
                     f" want: {np.round(want.ravel()[:8], 4).tolist()}")


def make_case(shape, dt):
    """Deterministic fp32 case data; the caller casts as needed."""
    C = shape[1]
    axes = tuple(d for d in range(len(shape)) if d != 1)
    n = 1
    for d in range(len(shape)):
        if d != 1:
            n *= shape[d]
    rng = np.random.RandomState(11)
    x = rng.randn(*shape).astype(np.float32)
    w = (rng.rand(C) + 0.5).astype(np.float32)
    b = (rng.randn(C) * 0.1).astype(np.float32)
    rm = rng.randn(C).astype(np.float32)
    rv = (rng.rand(C) + 0.5).astype(np.float32)
    dy = rng.randn(*shape).astype(np.float32)
    return dict(x=x, w=w, b=b, rm=rm, rv=rv, dy=dy, C=C, axes=axes, n=n,
                mom=0.1, eps=1e-5)


def check_native(c, dt_name, tol):
    """native_batch_norm: (out, save_mean, save_invstd); pure w.r.t. the
    running stats (torch mutates them in place; MLX does not)."""
    bf16 = dt_name == "bf16"
    dt = mx.bfloat16 if bf16 else mx.float32
    tdt = torch.bfloat16 if bf16 else torch.float32
    x_t = torch.tensor(c["x"]).to(tdt)
    w_t = torch.tensor(c["w"]).to(tdt)
    b_t = torch.tensor(c["b"]).to(tdt)
    rm_t = torch.tensor(c["rm"]).to(tdt)
    rv_t = torch.tensor(c["rv"]).to(tdt)
    x_m, w_m, b_m = (mx.array(c[k]).astype(dt) for k in ("x", "w", "b"))
    rm_m, rv_m = mx.array(c["rm"]).astype(dt), mx.array(c["rv"]).astype(dt)
    mom, eps = c["mom"], c["eps"]
    name = "native_batch_norm"

    # eval: torch returns EMPTY save stats; MLX returns the running stats
    r = mlx_vulkan.ATEN_TABLE[name](x_m, w_m, b_m, rm_m, rv_m, False, mom, eps)
    wr = torch.native_batch_norm(x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), False, mom, eps)
    close(f"{name} eval out", r[0], wr[0], tol)
    close(f"{name} eval save_mean (running)", r[1], rm_t, tol)
    close(f"{name} eval save_invstd (running)", r[2], 1.0 / torch.sqrt(rv_t + eps), tol)
    close(f"{name} eval rm untouched", to32(rm_m), rm_t, tol)
    close(f"{name} eval rv untouched", to32(rv_m), rv_t, tol)

    # train: batch stats; slot 3 is the batch INVERSE STD
    r = mlx_vulkan.ATEN_TABLE[name](x_m, w_m, b_m, rm_m, rv_m, True, mom, eps)
    wr = torch.native_batch_norm(x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), True, mom, eps)
    close(f"{name} train out", r[0], wr[0], tol)
    close(f"{name} train save_mean", r[1], wr[1], tol)
    close(f"{name} train save_invstd", r[2], wr[2], tol)
    close(f"{name} train rm untouched (pure lowering)", to32(rm_m), rm_t, tol)
    close(f"{name} train rv untouched (pure lowering)", to32(rv_m), rv_t, tol)


def check_batch_norm(c, dt_name, tol):
    """batch_norm: tensor-only contract."""
    bf16 = dt_name == "bf16"
    dt = mx.bfloat16 if bf16 else mx.float32
    tdt = torch.bfloat16 if bf16 else torch.float32
    x_t = torch.tensor(c["x"]).to(tdt)
    w_t = torch.tensor(c["w"]).to(tdt)
    b_t = torch.tensor(c["b"]).to(tdt)
    rm_t = torch.tensor(c["rm"]).to(tdt)
    rv_t = torch.tensor(c["rv"]).to(tdt)
    x_m, w_m, b_m = (mx.array(c[k]).astype(dt) for k in ("x", "w", "b"))
    rm_m, rv_m = mx.array(c["rm"]).astype(dt), mx.array(c["rv"]).astype(dt)
    mom, eps = c["mom"], c["eps"]
    name = "batch_norm"
    for training in (False, True):
        r = mlx_vulkan.ATEN_TABLE[name](x_m, w_m, b_m, rm_m, rv_m, training, mom, eps)
        if isinstance(r, tuple):  # MLX may return the native 3-tuple
            r = r[0]
        tb = torch.batch_norm(x_t, w_t, b_t, rm_t.clone(), rv_t.clone(),
                              training, mom, eps, True)
        if isinstance(tb, tuple):
            tb = tb[0]
        close(f"{name} {'train' if training else 'eval'} out", r, tb, tol)


def check_legit(c, dt_name, tol):
    """_native_batch_norm_legit_functional: 5-tuple with unbiased rv update."""
    bf16 = dt_name == "bf16"
    dt = mx.bfloat16 if bf16 else mx.float32
    tdt = torch.bfloat16 if bf16 else torch.float32
    x_t = torch.tensor(c["x"]).to(tdt)
    w_t = torch.tensor(c["w"]).to(tdt)
    b_t = torch.tensor(c["b"]).to(tdt)
    rm_t = torch.tensor(c["rm"]).to(tdt)
    rv_t = torch.tensor(c["rv"]).to(tdt)
    x_m, w_m, b_m = (mx.array(c[k]).astype(dt) for k in ("x", "w", "b"))
    rm_m, rv_m = mx.array(c["rm"]).astype(dt), mx.array(c["rv"]).astype(dt)
    mom, eps = c["mom"], c["eps"]
    name = "_native_batch_norm_legit_functional"

    r = mlx_vulkan.ATEN_TABLE[name](x_m, w_m, b_m, rm_m, rv_m, False, mom, eps)
    assert isinstance(r, tuple) and len(r) == 5, f"{name} eval contract"
    wr = torch.ops.aten._native_batch_norm_legit_functional(
        x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), False, mom, eps)
    close(f"{name} eval out", r[0], wr[0], tol)
    close(f"{name} eval rm untouched", to32(rm_m), rm_t, tol)

    r = mlx_vulkan.ATEN_TABLE[name](x_m, w_m, b_m, rm_m, rv_m, True, mom, eps)
    assert isinstance(r, tuple) and len(r) == 5, f"{name} train contract"
    wr = torch.ops.aten._native_batch_norm_legit_functional(
        x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), True, mom, eps)
    close(f"{name} train out", r[0], wr[0], tol)
    close(f"{name} train save_mean", r[1], wr[1], tol)
    close(f"{name} new_rm", r[3], wr[3], tol)
    close(f"{name} new_rv", r[4], wr[4], tol)


def check_backward(c, dt_name, tol):
    """native_batch_norm_backward: training dx (batch-stat formula), dw, db;
    eval dx through the running stats."""
    bf16 = dt_name == "bf16"
    dt = mx.bfloat16 if bf16 else mx.float32
    tdt = torch.bfloat16 if bf16 else torch.float32
    x_t = torch.tensor(c["x"]).to(tdt)
    w_t = torch.tensor(c["w"]).to(tdt)
    dy_t = torch.tensor(c["dy"]).to(tdt)
    rv_t = torch.tensor(c["rv"]).to(tdt)
    x_m = mx.array(c["x"]).astype(dt)
    dy_m = mx.array(c["dy"]).astype(dt)
    w_m = mx.array(c["w"]).astype(dt)
    rm_m = mx.array(c["rm"]).astype(dt)
    rv_m = mx.array(c["rv"]).astype(dt)
    C, axes = c["C"], c["axes"]
    chw = (1, C, *([1] * (len(c["x"].shape) - 2)))
    eps = c["eps"]

    sm = x_t.mean(dim=axes)
    si = 1.0 / torch.sqrt(x_t.var(dim=axes, unbiased=False) + eps)
    # build the saved stats from fp32 and cast to the case dtype (bf16
    # torch tensors cannot convert to numpy directly)
    sm_m = mx.array(sm.reshape(-1).float().numpy()).astype(x_m.dtype)
    si_m = mx.array(si.reshape(-1).float().numpy()).astype(x_m.dtype)

    g = mlx_vulkan._bn_backward(dy_m, x_m, w_m, rm_m, rv_m, sm_m, si_m, True, eps,
                                (True, True, True))
    sm_c, si_c = sm.reshape(chw), si.reshape(chw)
    xh = (x_t - sm_c) * si_c
    close("bn_backward train dx", g[0],
          (dy_t - dy_t.mean(dim=axes, keepdim=True)
           - xh * (dy_t * xh).mean(dim=axes, keepdim=True)) * si_c * w_t.reshape(chw), tol)
    close("bn_backward train dw", g[1], (dy_t * xh).sum(dim=axes), tol)
    close("bn_backward train db", g[2], dy_t.sum(dim=axes), tol)

    g = mlx_vulkan._bn_backward(dy_m, x_m, w_m, rm_m, rv_m, sm_m, si_m, False, eps,
                                (True, True, True))
    close("bn_backward eval dx", g[0],
          dy_t / torch.sqrt(rv_t.reshape(chw) + eps) * w_t.reshape(chw), tol)


def check_no_training(c, dt_name, tol):
    """_native_batch_norm_legit_no_training: aten order (x, w, b, rm, rv,
    momentum, eps) with DISTINCT momentum and eps; torch returns
    (out, empty, empty), MLX returns the running stats in slots 2/3."""
    bf16 = dt_name == "bf16"
    dt = mx.bfloat16 if bf16 else mx.float32
    tdt = torch.bfloat16 if bf16 else torch.float32
    x_t = torch.tensor(c["x"]).to(tdt)
    w_t = torch.tensor(c["w"]).to(tdt)
    b_t = torch.tensor(c["b"]).to(tdt)
    rm_t = torch.tensor(c["rm"]).to(tdt)
    rv_t = torch.tensor(c["rv"]).to(tdt)
    x_m, w_m, b_m = (mx.array(c[k]).astype(dt) for k in ("x", "w", "b"))
    rm_m, rv_m = mx.array(c["rm"]).astype(dt), mx.array(c["rv"]).astype(dt)
    mom, eps = 0.25, 1e-3  # distinct, to catch argument-order binding
    name = "_native_batch_norm_legit_no_training"
    # FX graph through the executor: the real ATen entry, all 3 outputs,
    # compared against torch (not a manual oracle)
    # FP16 input with FP32 affine/stats: torch returns (fp16 out,
    # fp32 empty, fp32 empty); the lowering must keep that split.
    if dt_name == "fp32":
        x16 = mx.array(c["x"]).astype(mx.float16)
        r16 = mlx_vulkan.ATEN_TABLE[name](x16, w_m, b_m, rm_m, rv_m, mom, eps)
        assert r16[0].dtype == mx.float16, r16[0].dtype
        assert r16[1].dtype == mx.float32 and r16[2].dtype == mx.float32, \
            (r16[1].dtype, r16[2].dtype)
    g = torch.fx.Graph()
    phs = [g.placeholder(f"p{i}") for i in range(7)]
    g.call_function(torch.ops.aten._native_batch_norm_legit_no_training,
                    tuple(phs), {})
    g.output((phs[6],))  # placeholder; replaced below with the real node
    # rebuild properly: capture the node
    g = torch.fx.Graph()
    phs = [g.placeholder(f"p{i}") for i in range(7)]
    n = g.call_function(torch.ops.aten._native_batch_norm_legit_no_training,
                        tuple(phs), {})
    g.output(n)
    gm = torch.fx.GraphModule(torch.nn.Module(), g)
    run, stats = mlx_vulkan.make_executor(gm)
    r = run(x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), mom, eps)
    assert stats["cpu"] == [], stats
    wr = torch.ops.aten._native_batch_norm_legit_no_training(
        x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), mom, eps)
    assert isinstance(r, tuple) and len(r) == 3, f"{name} contract"
    close(f"{name} out", r[0], wr[0], tol)
    # torch's slots 2/3 are EMPTY typed tensors; the lowering must match
    # (shape/dtype only - a max-reduction over a zero-size array is undefined)
    assert tuple(r[1].shape) == tuple(wr[1].shape) == (0,), (r[1].shape, wr[1].shape)
    assert tuple(r[2].shape) == tuple(wr[2].shape) == (0,), (r[2].shape, wr[2].shape)
    # the executor widens bf16 outputs to fp32 (numpy has no bf16 buffer);
    # dtype fidelity at the entry itself is asserted in the fp16 case above
    assert r[0].dtype == wr[0].dtype or (
        dt_name == "bf16" and r[0].dtype == torch.float32), (r[0].dtype, wr[0].dtype)
    # the eval math itself must use the passed eps, not the momentum
    C, chw = c["C"], (1, c["C"], *([1] * (len(c["x"].shape) - 2)))
    want = (x_t - rm_t.reshape(chw)) / torch.sqrt(rv_t.reshape(chw) + eps) * w_t.reshape(chw) + b_t.reshape(chw)
    close(f"{name} out vs manual running-stat formula", r[0], want, tol)


def main():
    # one process, entries in sequence: native, batch_norm, legit per case
    for dt_name, tol in (("fp32", 2e-5), ("bf16", 4e-2)):
        for shape in ((4, 6, 3, 3), (2, 6)):
            c = make_case(shape, dt_name)
            check_native(c, dt_name, tol)
            check_batch_norm(c, dt_name, tol)
            check_legit(c, dt_name, tol)
            check_no_training(c, dt_name, tol)
            check_backward(c, dt_name, tol)
    print("ALL BN parity checks PASS (one process, in sequence)")


if __name__ == "__main__":
    main()
