"""Parity tests for unclaimed train-path ops: _log_softmax fwd+bwd,
t, _unsafe_view, mv/dot — the matmul-family backward glue.

Run: MV_LOG=1 python test_train_ops.py   (needs mlx_vulkan importable)
Each check compares a torch.compile(backend="mlx_vulkan") result against
the eager CPU reference. Fails on unlowered ops (CPU fallback still
produces correct values, so we also assert no FALLBACK lines fired).
"""
import io
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_BUF = io.StringIO()
_h = logging.StreamHandler(_BUF)
_h.setLevel(logging.INFO)
logging.getLogger("mlx_vulkan").addHandler(_h)
logging.getLogger("mlx_vulkan").setLevel(logging.INFO)

import mlx_vulkan  # noqa: E402

RESULTS = []


def check(name, got, want, tol=1e-5):
    got = got.float() if isinstance(got, torch.Tensor) else torch.as_tensor(got)
    d = float((got - want.float()).abs().max()) if got.numel() else 0.0
    ok = d <= tol and tuple(got.shape) == tuple(want.shape)
    RESULTS.append((name, ok, d))
    print(f"{'PASS' if ok else 'FAIL'} {name} maxdiff={d:.3e} shape={tuple(got.shape)}", flush=True)


def no_fallback(tag):
    lines = [l for l in _BUF.getvalue().splitlines() if "FALLBACK" in l]
    _BUF.truncate(0); _BUF.seek(0)
    ok = not lines
    RESULTS.append((f"{tag} (no FALLBACK)", ok, len(lines)))
    print(f"{'PASS' if ok else 'FAIL'} {tag} no_fallback ({len(lines)} fallbacks)", flush=True)


def t_lsm_fwd():
    x = torch.randn(4, 32, 64)
    for dim in (-1, 1):
        f = torch.compile(lambda a: F.log_softmax(a, dim=dim), backend="mlx_vulkan")
        check(f"_log_softmax dim={dim}", f(x), F.log_softmax(x, dim=dim))
    no_fallback("_log_softmax")


def t_lsm_bwd():
    torch.manual_seed(0)
    a = torch.randn(16, 64)
    f = torch.compile(lambda t: F.log_softmax(t, dim=-1), backend="mlx_vulkan_train")
    ga = a.clone().requires_grad_(True)
    f(ga).sum().backward()
    b = a.clone().requires_grad_(True)
    F.log_softmax(b, dim=-1).sum().backward()
    check("_log_softmax_backward_data sum", ga.grad, b.grad)
    # weighted grads exercise the exp(out)*sum(g) term, not just the pass-through
    w = torch.randn(16, 64)
    g = torch.compile(lambda t: F.log_softmax(t, dim=-1), backend="mlx_vulkan_train")
    ga2 = a.clone().requires_grad_(True)
    (g(ga2) * w).sum().backward()
    b2 = a.clone().requires_grad_(True)
    (F.log_softmax(b2, dim=-1) * w).sum().backward()
    check("_log_softmax_backward_data weighted", ga2.grad, b2.grad)
    no_fallback("_log_softmax_backward")


def t_t_backward():
    # t() itself plus the t+mm pair torch autograd emits for Linear backward
    x = torch.randn(8, 16)
    f = torch.compile(lambda a: a.t(), backend="mlx_vulkan")
    check("t 2d", f(x), x.t())
    x1 = torch.randn(4)
    f1 = torch.compile(lambda a: a.t(), backend="mlx_vulkan")
    check("t 1d", f1(x1), x1.t())
    torch.manual_seed(0)
    w = torch.randn(8, 16); b = torch.randn(8); xin = torch.randn(4, 16)
    f = torch.compile(lambda x, w_, b_: F.linear(x, w_, b_), backend="mlx_vulkan_train")
    gw, gb, gx = w.clone().requires_grad_(True), b.clone().requires_grad_(True), xin.clone().requires_grad_(True)
    f(gx, gw, gb).sum().backward()
    rw, rb, rx = w.clone().requires_grad_(True), b.clone().requires_grad_(True), xin.clone().requires_grad_(True)
    F.linear(rx, rw, rb).sum().backward()
    check("linear grad_x", gx.grad, rx.grad, tol=1e-6)
    check("linear grad_w", gw.grad, rw.grad, tol=1e-6)
    check("linear grad_b", gb.grad, rb.grad, tol=1e-6)
    no_fallback("linear_backward")


def t_unsafe_view_mv_dot():
    torch.manual_seed(0)
    a = torch.randn(2, 3, 4)
    f = torch.compile(lambda t: t.view(-1), backend="mlx_vulkan")
    check("view", f(a), a.view(-1))
    m = torch.randn(4, 8); v = torch.randn(8)
    f2 = torch.compile(torch.mv, backend="mlx_vulkan")
    check("mv", f2(m, v), torch.mv(m, v))
    u, w = torch.randn(16), torch.randn(16)
    f3 = torch.compile(torch.dot, backend="mlx_vulkan")
    check("dot", f3(u, w), torch.dot(u, w), tol=1e-5)
    no_fallback("view_mv_dot")


def t_slice_clone_silu():
    torch.manual_seed(0)
    # aten.slice with implicit step=1 (4-arg form — the LoRA fwd emits this)
    a = torch.randn(4, 16, 32, 128)
    f = torch.compile(lambda t: t[:, :, :, :64], backend="mlx_vulkan")
    check("slice implicit step", f(a), a[:, :, :, :64])
    # open-ended slice lowers to end=INT64_MAX — must clamp to dim size
    f2 = torch.compile(lambda t: t[:, :, :, 64:], backend="mlx_vulkan")
    check("slice INT64_MAX end", f2(a), a[:, :, :, 64:])
    # clone(memory_format=...) kwarg — peft/rope emits it
    c = torch.randn(4, 8, 16)
    fc = torch.compile(lambda t: t.clone(memory_format=torch.contiguous_format),
                       backend="mlx_vulkan")
    check("clone memory_format", fc(c), c)
    # slice_backward + silu_backward via train backend
    sb = a.clone().requires_grad_(True)
    g = torch.compile(lambda t: t[:, :, :, :64].sum(), backend="mlx_vulkan_train")
    g(sb).backward()
    rb = a.clone().requires_grad_(True)
    rb[:, :, :, :64].sum().backward()
    check("slice_backward", sb.grad, rb.grad, tol=1e-6)
    x = torch.randn(8, 32)
    sx = x.clone().requires_grad_(True)
    h = torch.compile(lambda t: F.silu(t).sum(), backend="mlx_vulkan_train")
    h(sx).backward()
    rx = x.clone().requires_grad_(True)
    F.silu(rx).sum().backward()
    check("silu_backward", sx.grad, rx.grad, tol=1e-5)
    no_fallback("slice_clone_silu")


def t_sdpa_flash_pair():
    # F.sdpa on fp32 CPU dispatches to _scaled_dot_product_flash_attention_for_cpu
    # and its _backward — the pair the LoRA train step was still falling back on.
    torch.manual_seed(0)
    for sq, sk, causal, qh, kvh in ((64, 64, True, 4, 4), (32, 64, False, 4, 4),
                                   (48, 64, True, 4, 4), (64, 64, True, 8, 4)):
        q = torch.randn(2, qh, sq, 32)
        k = torch.randn(2, kvh, sk, 32)
        v = torch.randn(2, kvh, sk, 32)
        g = torch.randn(2, qh, sq, 32)

        def run(fn):
            qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
            o = fn(qq, kk, vv)
            o.backward(g)
            return o.detach(), qq.grad, kk.grad, vv.grad

        gqa = qh != kvh  # HF emits enable_gqa=True when kv heads differ
        ref = run(lambda a, b, c: F.scaled_dot_product_attention(
            a, b, c, is_causal=causal, enable_gqa=gqa))
        f = torch.compile(lambda a, b, c: F.scaled_dot_product_attention(
            a, b, c, is_causal=causal, enable_gqa=gqa),
            backend="mlx_vulkan_train")
        got = run(f)
        for i, nm in enumerate(("out", "dq", "dk", "dv")):
            check(f"sdpa_flash sq={sq} sk={sk} causal={causal} "
                  f"qh={qh} kvh={kvh} {nm}", got[i], ref[i], tol=5e-5)
    no_fallback("sdpa_flash")


def t_review_fixes():
    import numpy as np
    import mlx.core as mx
    torch.manual_seed(0)

    # w8F: the rebase dropped lift_fresh_copy; aot lifts module params through
    # it, so both the spec entry and a compiled module call are covered.
    ok = "lift_fresh_copy" in mlx_vulkan._FUNC_SPECS
    RESULTS.append(("lift_fresh_copy in _FUNC_SPECS", ok, 0))
    print(f"{'PASS' if ok else 'FAIL'} lift_fresh_copy in _FUNC_SPECS", flush=True)
    lin = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    f = torch.compile(lin, backend="mlx_vulkan_train")
    check("lift_fresh_copy module lift", f(x), lin(x))

    # w8F: _slice crashed on open (None) bounds after the int32 clamp.
    a = torch.randn(4, 8)
    f2 = torch.compile(lambda t: t[2:], backend="mlx_vulkan")
    check("slice None end", f2(a), a[2:])
    f3 = torch.compile(lambda t: t[:3], backend="mlx_vulkan")
    check("slice None start", f3(a), a[:3])
    got = torch.from_numpy(np.array(mlx_vulkan._slice(mx.array(np.arange(8.0)), 0, None, None)))
    check("slice None-None direct", got, torch.arange(8.0))

    # w8F: _log_softmax ignored half_to_float / input_dtype.
    f4 = torch.compile(
        lambda t: torch.ops.aten._log_softmax.default(t, -1, True),
        backend="mlx_vulkan")
    xh = torch.randn(4, 8).half()
    got4, want4 = f4(xh), F.log_softmax(xh.float(), dim=-1)
    check("_log_softmax half_to_float", got4, want4, tol=5e-4)
    ok = got4.dtype == torch.float32
    RESULTS.append(("_log_softmax half_to_float dtype fp32", ok, 0))
    print(f"{'PASS' if ok else 'FAIL'} _log_softmax half_to_float dtype fp32 "
          f"(got {got4.dtype})", flush=True)
    g = torch.randn(2, 8).half()
    out = torch.randn(2, 8).half()
    r = mlx_vulkan._log_softmax_bwd(g, out, -1, torch.float32)
    ok = r.dtype == mx.float32
    RESULTS.append(("_log_softmax_bwd input_dtype fp32", ok, 0))
    print(f"{'PASS' if ok else 'FAIL'} _log_softmax_bwd input_dtype fp32 "
          f"(got {r.dtype})", flush=True)
    no_fallback("review_fixes")


if __name__ == "__main__":
    t_lsm_fwd()
    t_lsm_bwd()
    t_t_backward()
    t_unsafe_view_mv_dot()
    t_slice_clone_silu()
    t_sdpa_flash_pair()
    t_review_fixes()
    fails = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(fails)}/{len(RESULTS)} pass; {len(fails)} fail", flush=True)
    sys.exit(1 if fails else 0)
