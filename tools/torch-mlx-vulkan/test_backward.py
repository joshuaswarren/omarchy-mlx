#!/usr/bin/env python3
"""PAR-5.2 backward lowering parity tests vs CPU aten ops (allclose 1e-5/1e-5)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import mlx.core as mx
import mlx_vulkan

TOL = 1e-5


def _tm(a):
    if getattr(a, "dtype", None) in (mx.bfloat16, mx.float16):
        a = a.astype(mx.float32)
    return torch.from_numpy(np.array(a))


def _am(t):
    # torch tensor -> mx array preserving bf16 (numpy has no bf16).
    if t.dtype == torch.bfloat16:
        return mx.array(t.float().numpy()).astype(mx.bfloat16)
    return mx.array(t.numpy())


def _cmp(tag, mine, ref):
    d = float((_tm(mine) - ref).abs().max())
    assert torch.allclose(_tm(mine), ref, atol=TOL, rtol=TOL), f"{tag}: {d:.3e}"
    print(f"PASS {tag} maxdiff={d:.3e}")


def test_native_layer_norm_backward():
    torch.manual_seed(0)
    for ns_shape, xshape in [((128,), (4, 64, 128)), ((64,), (8, 64)), ((8, 16), (4, 8, 16))]:
        x = torch.randn(*xshape)
        w = torch.randn(*ns_shape)
        b = torch.randn(*ns_shape)
        out, mean, rstd = torch.native_layer_norm(x, list(ns_shape), w, b, 1e-5)
        g = torch.randn_like(out)
        ref = torch.ops.aten.native_layer_norm_backward(
            g, x, list(ns_shape), mean, rstd, w, b, [True, True, True])
        my = mlx_vulkan.ATEN_TABLE["native_layer_norm_backward"](
            mx.array(g.numpy()), mx.array(x.numpy()), list(ns_shape),
            mx.array(mean.numpy()), mx.array(rstd.numpy()),
            mx.array(w.numpy()), mx.array(b.numpy()), [True, True, True])
        for tag, m, r in zip(("dx", "dw", "db"), my, ref):
            _cmp(f"ln_bwd{ns_shape}.{tag}", m, r)


def test_native_layer_norm_backward_bf16_and_fp32():
    """PT-C: the training-path backward at both dtypes, including the
    weightless/biasless graph shapes aot_autograd emits when the norm has no
    affine parameters, and the output_maskFalse arms for the grads torch
    does not need. The bf16 reference runs the aten op in fp32 (torch 2.4
    has no bf16 CPU kernel for this backward - it raises a device error)
    and compares upcast with a tolerance that covers the bf16 mantissa;
    mean/rstd stay fp32, which is what aot_autograd saves for bf16 inputs."""
    torch.manual_seed(0)
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        for ns_shape, xshape in [((64,), (8, 64)), ((32,), (4, 6, 32))]:
            for affine in (True, False):
                x = torch.randn(*xshape, dtype=dtype)
                w = torch.randn(*ns_shape, dtype=dtype) if affine else None
                b = torch.randn(*ns_shape, dtype=dtype) if affine else None
                out, mean, rstd = torch.native_layer_norm(
                    x.float(), list(ns_shape), None if w is None else w.float(),
                    None if b is None else b.float(), 1e-5)
                out = out.to(dtype)
                g = torch.randn_like(out)

                def mx_or_none(t):
                    if t is None:
                        return None
                    if t.dtype == torch.bfloat16:
                        return mx.array(t.float().numpy()).astype(mx.bfloat16)
                    return mx.array(t.numpy())

                for mask in ([True, True, True], [True, False, False]):
                    my = mlx_vulkan.ATEN_TABLE["native_layer_norm_backward"](
                        mx_or_none(g), mx_or_none(x), list(ns_shape),
                        mx_or_none(mean), mx_or_none(rstd),
                        mx_or_none(w), mx_or_none(b), mask)
                    # torch 2.4 CPU has no kernel for weight=None on this
                    # backward ("tensor does not have a device"), so the
                    # weightless reference is autograd's dx through
                    # F.layer_norm; dw/db have no CPU reference and torch
                    # never requests them for an affine-free norm, so only
                    # dx is compared there.
                    if affine:
                        ref_m = torch.ops.aten.native_layer_norm_backward(
                            g.float(), x.float(), list(ns_shape), mean, rstd,
                            w.float(), b.float(), mask)
                        pairs = [(tag, my[i], ref_m[i])
                                 for i, tag in enumerate(("dx", "dw", "db"))
                                 if mask[i]]
                    else:
                        xg = x.float().clone().requires_grad_(True)
                        F.layer_norm(xg, list(ns_shape), None, None, 1e-5) \
                            .backward(g.float())
                        pairs = [("dx", my[0], xg.grad)]
                    for tag, m, r in pairs:
                        m32 = m.astype(mx.float32) if m.dtype == mx.bfloat16 else m
                        d = float((_tm(m32) - r).abs().max())
                        assert torch.allclose(_tm(m32), r, atol=tol, rtol=tol), \
                            f"ln_bwd {dtype} {ns_shape} affine={affine} {tag}: {d:.3e}"
                        print(f"PASS ln_bwd {dtype} {ns_shape} affine={affine} "
                              f"mask={mask}.{tag} maxdiff={d:.3e}")


def test_flash_cpu_fwd():
    torch.manual_seed(1)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    for causal in (False, True):
        ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            q, k, v, 0.0, causal)
        my_out, my_lse = mlx_vulkan.ATEN_TABLE["_scaled_dot_product_flash_attention_for_cpu"](
            mx.array(q.numpy()), mx.array(k.numpy()), mx.array(v.numpy()),
            0.0, causal)
        _cmp(f"flash_fwd out causal={causal}", my_out, ref_out)
        _cmp(f"flash_fwd lse causal={causal}", my_lse, ref_lse)


def test_flash_cpu_bwd_gqa():
    torch.manual_seed(3)
    q = torch.randn(2, 16, 32, 64)
    k = torch.randn(2, 8, 32, 64)
    v = torch.randn(2, 8, 32, 64)
    g = torch.randn(2, 16, 32, 64)
    out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        q, k, v, 0.0, True)
    ref = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
        g, q, k, v, out, lse, 0.0, True)
    my = mlx_vulkan.ATEN_TABLE["_scaled_dot_product_flash_attention_for_cpu_backward"](
        mx.array(g.numpy()), mx.array(q.numpy()), mx.array(k.numpy()),
        mx.array(v.numpy()), mx.array(out.numpy()), mx.array(lse.numpy()),
        0.0, True)
    for tag, m, r in zip(("dq", "dk", "dv"), my, ref):
        _cmp(f"flash_bwd_gqa {tag}", m, r)


def test_slice_backward():
    torch.manual_seed(4)
    x = torch.randn(2, 8, 64, 128, requires_grad=True)
    y = x[:, :, 4:36, :]
    g = torch.randn_like(y)
    y.backward(g)
    ref = torch.ops.aten.slice_backward(
        g, [2, 8, 64, 128], 2, 4, 36, 1)
    my = mlx_vulkan.ATEN_TABLE["slice_backward"](
        mx.array(g.numpy()), [2, 8, 64, 128], 2, 4, 36, 1)
    _cmp("slice_backward", my, ref)
    g2 = torch.randn(2, 8, 60, 128)
    ref2 = torch.ops.aten.slice_backward(
        g2, [2, 8, 64, 128], 2, 4, 9223372036854775807, 1)
    my2 = mlx_vulkan.ATEN_TABLE["slice_backward"](
        mx.array(g2.numpy()), [2, 8, 64, 128], 2, 4, 9223372036854775807, 1)
    _cmp("slice_backward[max_end]", my2, ref2)


def test_slice_backward_clamp():
    # bounds below -N clamp to 0 (no wraparound)
    g = torch.ones(4)
    ref = torch.zeros(10); ref[0:4] = 1  # slice(-20, 4) -> [0:4]
    my = mlx_vulkan.ATEN_TABLE["slice_backward"](
        mx.array(g.numpy()), (10,), 0, -20, 4, 1)
    _cmp("slice_backward[start<-N]", my, ref)
    g6 = torch.ones(6)
    ref = torch.zeros(10); ref[0:6] = 1  # slice(-25, -4) -> [0:6]
    my = mlx_vulkan.ATEN_TABLE["slice_backward"](
        mx.array(g6.numpy()), (10,), 0, -25, -4, 1)
    _cmp("slice_backward[both<-N]", my, ref)


def test_flash_cpu_causal_rect():
    # T != S: torch causal mask is top-left aligned
    torch.manual_seed(3)
    q = torch.randn(2, 4, 48, 32)
    k = torch.randn(2, 4, 64, 32)
    v = torch.randn(2, 4, 64, 32)
    g = torch.randn(2, 4, 48, 32)
    out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        q, k, v, 0.0, True)
    my_out, my_lse = mlx_vulkan.ATEN_TABLE["_scaled_dot_product_flash_attention_for_cpu"](
        mx.array(q.numpy()), mx.array(k.numpy()), mx.array(v.numpy()), 0.0, True)
    _cmp("flash_fwd causal T<S out", my_out, out)
    _cmp("flash_fwd causal T<S lse", my_lse, lse)
    ref = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
        g, q, k, v, out, lse, 0.0, True)
    my = mlx_vulkan.ATEN_TABLE["_scaled_dot_product_flash_attention_for_cpu_backward"](
        mx.array(g.numpy()), mx.array(q.numpy()), mx.array(k.numpy()),
        mx.array(v.numpy()), mx.array(out.numpy()), mx.array(lse.numpy()),
        0.0, True)
    for tag, m, r in zip(("dq", "dk", "dv"), my, ref):
        _cmp(f"flash_bwd causal T<S {tag}", m, r)


def test_silu_backward():
    torch.manual_seed(5)
    x = torch.randn(4, 64)
    g = torch.randn(4, 64)
    ref = torch.ops.aten.silu_backward(g, x)
    my = mlx_vulkan.ATEN_TABLE["silu_backward"](
        mx.array(g.numpy()), mx.array(x.numpy()))
    _cmp("silu_backward", my, ref)


def test_flash_cpu_bwd():
    torch.manual_seed(2)
    q = torch.randn(2, 4, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    g = torch.randn(2, 4, 16, 32)
    for causal in (False, True):
        out, lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
            q, k, v, 0.0, causal)
        ref = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
            g, q, k, v, out, lse, 0.0, causal)
        my = mlx_vulkan.ATEN_TABLE["_scaled_dot_product_flash_attention_for_cpu_backward"](
            mx.array(g.numpy()), mx.array(q.numpy()), mx.array(k.numpy()),
            mx.array(v.numpy()), mx.array(out.numpy()), mx.array(lse.numpy()),
            0.0, causal)
        for tag, m, r in zip(("dq", "dk", "dv"), my, ref):
            _cmp(f"flash_bwd {tag} causal={causal}", m, r)




def test_flash_cpu_fwd_gqa():
    torch.manual_seed(11)
    q = torch.randn(2, 8, 16, 32)
    k = torch.randn(2, 4, 16, 32)
    v = torch.randn(2, 4, 16, 32)
    ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        q, k.repeat_interleave(2, dim=1), v.repeat_interleave(2, dim=1), 0.0, False)
    my_out, my_lse = mlx_vulkan.ATEN_TABLE["_scaled_dot_product_flash_attention_for_cpu"](
        mx.array(q.numpy()), mx.array(k.numpy()), mx.array(v.numpy()), 0.0, False)
    _cmp("flash_gqa_fwd out", my_out, ref_out)
    _cmp("flash_gqa_fwd lse", my_lse, ref_lse)


def test_flash_cpu_mask_and_dtype():
    torch.manual_seed(12)
    for dt in (torch.float32, torch.bfloat16):
        q = torch.randn(2, 4, 8, 16, dtype=dt)
        k = torch.randn(2, 4, 8, 16, dtype=dt)
        v = torch.randn(2, 4, 8, 16, dtype=dt)
        bmask = torch.ones(8, 8, dtype=torch.bool).tril()
        fmask = torch.where(bmask, 0.0, float("-inf")).expand(2, 4, 8, 8)
        for tag, am in (("bool", bmask), ("float", fmask)):
            if tag == "float":
                ref_out, ref_lse = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
                    q, k, v, 0.0, False, attn_mask=am.to(dt))
            else:
                # aten flash_cpu on this torch rejects bool masks; math sdpa
                # gives the same keep/drop semantics, lse scored manually.
                ref_out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=am)
                sc = torch.matmul(q.float(), k.float().transpose(-2, -1)) / 16 ** 0.5
                sc = sc.masked_fill(~am, float("-inf"))
                ref_lse = torch.logsumexp(sc, dim=-1)
            my_out, my_lse = mlx_vulkan.ATEN_TABLE["_scaled_dot_product_flash_attention_for_cpu"](
                _am(q), _am(k), _am(v), 0.0, False, attn_mask=_am(am))
            assert my_out.dtype == (mx.bfloat16 if dt == torch.bfloat16 else mx.float32), \
                f"out dtype {my_out.dtype}"
            assert my_lse.dtype == mx.float32, f"lse dtype {my_lse.dtype}"
            tol = 2e-2 if dt == torch.bfloat16 else TOL
            d_out = float((_tm(my_out).float() - ref_out.float()).abs().max())
            d_lse = float((_tm(my_lse) - ref_lse).abs().max())
            assert d_out < tol and d_lse < tol, f"{tag}/{dt}: {d_out:.3e} {d_lse:.3e}"
            print(f"PASS flash mask {tag} {dt} out={d_out:.3e} lse={d_lse:.3e}")


def _eff_ref(q, k, v, causal_off=0):
    # manual fp32 math reference: top-left causal, softmax@v + logsumexp.
    sc = torch.matmul(q.float(), k.float().transpose(-2, -1)) / q.shape[-1] ** 0.5
    sq, sk = q.shape[-2], k.shape[-2]
    keep = torch.arange(sk)[None, :] <= torch.arange(sq)[:, None] + causal_off
    sc = sc.masked_fill(~keep, float("-inf"))
    lse = torch.logsumexp(sc, dim=-1)
    return torch.softmax(sc, dim=-1).matmul(v.float()), lse


def test_efficient_attention_arity_and_causal():
    torch.manual_seed(13)
    q = torch.randn(1, 4, 6, 16)
    k = torch.randn(1, 4, 10, 16)
    v = torch.randn(1, 4, 10, 16)
    # no CPU kernel for the aten op on this build; manual top-left ref.
    ref_out, ref_lse = _eff_ref(q, k, v)
    my = mlx_vulkan.ATEN_TABLE["_scaled_dot_product_efficient_attention"](
        mx.array(q.numpy()), mx.array(k.numpy()), mx.array(v.numpy()),
        None, True, 0.0, True)
    assert len(my) == 4, f"efficient_attention must return 4-tuple, got {len(my)}"
    _cmp("eff out", my[0], ref_out)
    _cmp("eff lse", my[1], ref_lse)


def test_flash_cpu_bwd_dtype():
    torch.manual_seed(15)
    q32 = torch.randn(2, 4, 8, 16)
    k32 = torch.randn(2, 4, 8, 16)
    v32 = torch.randn(2, 4, 8, 16)
    g32 = torch.randn(2, 4, 8, 16)
    qb, kb, vb, gb = (t.bfloat16() for t in (q32, k32, v32, g32))
    out32, lse32 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu(
        q32, k32, v32, 0.0, False)
    ref = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu_backward(
        g32, q32, k32, v32, out32, lse32, 0.0, False)
    my_out, my_lse = mlx_vulkan.ATEN_TABLE["_scaled_dot_product_flash_attention_for_cpu"](
        _am(qb), _am(kb), _am(vb), 0.0, False)
    my = mlx_vulkan.ATEN_TABLE["_scaled_dot_product_flash_attention_for_cpu_backward"](
        _am(gb), _am(qb), _am(kb), _am(vb), my_out, my_lse, 0.0, False)
    for tag, m, r in zip(("dq", "dk", "dv"), my, ref):
        assert m.dtype == mx.bfloat16, f"{tag} dtype {m.dtype}, want bfloat16"
        d = float((_tm(m) - r).abs().max())
        assert d < 2e-2, f"bwd bf16 {tag}: {d:.3e}"
        print(f"PASS flash_bwd bf16 {tag} maxdiff={d:.3e}")


def test_efficient_attention_forward_schema():
    torch.manual_seed(14)
    # aten._efficient_attention_forward takes BSHD (S != H catches swaps).
    B, S, H, D = 1, 5, 3, 8
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    ref_out, ref_lse = _eff_ref(q.transpose(1, 2), k.transpose(1, 2),
                                v.transpose(1, 2))
    ref_out = ref_out.transpose(1, 2)  # back to BSHD; lse stays BHS
    my = mlx_vulkan.ATEN_TABLE["_efficient_attention_forward"](
        mx.array(q.numpy()), mx.array(k.numpy()), mx.array(v.numpy()),
        None, None, None, S, S, 0.0, 1, True)
    assert len(my) == 6, f"efficient_attention_forward must return 6-tuple, got {len(my)}"
    assert type(my[4]) is int and type(my[5]) is int,         f"max_seqlens must be SymInt, got {type(my[4])}, {type(my[5])}"
    assert my[4] == S and my[5] == S
    _cmp("eff_fwd out", my[0], ref_out)
    _cmp("eff_fwd lse", my[1], ref_lse)
    # 4-D bias stays (B, H, Sq, Sk) - H != Sq catches layout swaps.
    bias = torch.randn(B, H, S, S) * 0.5
    qb, kb, vb = (t_.transpose(1, 2) for t_ in (q, k, v))
    sc = torch.matmul(qb.float(), kb.float().transpose(-2, -1)) / D ** 0.5
    sc = sc + bias.float()
    ref_out_b = torch.softmax(sc, dim=-1).matmul(vb.float()).transpose(1, 2)
    ref_lse_b = torch.logsumexp(sc, dim=-1)
    my_b = mlx_vulkan.ATEN_TABLE["_efficient_attention_forward"](
        mx.array(q.numpy()), mx.array(k.numpy()), mx.array(v.numpy()),
        _am(bias), None, None, S, S, 0.0, 0, True)
    _cmp("eff_fwd bias out", my_b[0], ref_out_b)
    _cmp("eff_fwd bias lse", my_b[1], ref_lse_b)


def test_no_duplicate_table_keys():
    import ast as _ast, collections
    src = open(mlx_vulkan.__file__).read()
    tree = _ast.parse(src)
    dups = []
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Dict):
            keys = collections.Counter(
                kk.value for kk in node.keys
                if isinstance(kk, _ast.Constant) and isinstance(kk.value, str))
            dups += [k for k, c in keys.items() if c > 1]
    # "log"/"t"/"tril" are pre-existing intentional overrides inherited from
    # main (later entries refine earlier fallbacks); this test guards against
    # NEW shadowing duplicates like the flash-for-cpu one this PR removes.
    dups = [k for k in set(dups) - {"log", "t", "tril"}]
    assert not dups, f"new duplicate keys shadowed within one table literal: {dups}"
    assert mlx_vulkan.ATEN_TABLE["_scaled_dot_product_flash_attention_for_cpu"] is \
        mlx_vulkan._sdpa_flash_cpu


def test_native_layer_norm_backward_lowered_in_training_graph():
    """PT-C census rerun: a LayerNorm training step through aot_autograd
    must dispatch native_layer_norm_backward through the MLX lowering -
    the wrapped ATEN_TABLE entry counts the calls and no
    'FALLBACK native_layer_norm_backward' record may appear."""
    import logging
    from torch._dynamo.backends.common import aot_autograd

    class TinyLN(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln = nn.LayerNorm(8)
            self.fc = nn.Linear(8, 4)

        def forward(self, x):
            return self.fc(self.ln(x))

    torch.manual_seed(0)
    model = TinyLN()
    calls = []
    real = mlx_vulkan.ATEN_TABLE["native_layer_norm_backward"]

    def counting(*a, **k):
        calls.append(a)
        return real(*a, **k)

    mlx_vulkan.ATEN_TABLE["native_layer_norm_backward"] = counting
    records = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())
    log = logging.getLogger("mlx_vulkan")
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        def lower(gm, ex):
            runner = mlx_vulkan.mlx_vulkan_backend(gm, ex,
                                                   return_all_outputs=True)

            def call(*inputs):
                flat = inputs
                if (len(inputs) == 1 and isinstance(inputs[0], (list, tuple))
                        and not torch.is_tensor(inputs[0])):
                    flat = tuple(inputs[0])
                return runner(*flat)

            call._boxed_call = True
            return call

        step_fn = torch.compile(model, backend=aot_autograd(
            fw_compiler=lower, bw_compiler=lower))
        x = torch.randn(2, 8)
        target = torch.randn(2, 4)
        loss = torch.nn.functional.mse_loss(step_fn(x), target)
        loss.backward()
    finally:
        mlx_vulkan.ATEN_TABLE["native_layer_norm_backward"] = real
        log.removeHandler(handler)

    assert calls, "native_layer_norm_backward never reached the MLX lowering"
    fallbacks = [r for r in records
                 if "FALLBACK" in r and "native_layer_norm_backward" in r]
    assert not fallbacks, f"the backward CPU-fell back: {fallbacks}"
    print(f"PASS census: {len(calls)} native_layer_norm_backward dispatches "
          f"through the MLX lowering, 0 fallback records")
def test_embedding_dense_backward():
    torch.manual_seed(6)
    cases = [
        (32, 16, 48, -1, False),
        (32, 16, 48, 3, False),
        (64, 32, 128, 7, True),
        (16, 8, 24, 0, True),
    ]
    for num_weights, feat, rows, padding_idx, scale in cases:
        idx = torch.randint(0, num_weights, (rows,))
        idx[0] = padding_idx if padding_idx >= 0 else idx[0]
        g = torch.randn(rows, feat)
        ref = torch.ops.aten.embedding_dense_backward(
            g, idx, num_weights, padding_idx, scale)
        my = mlx_vulkan.ATEN_TABLE["embedding_dense_backward"](
            mx.array(g.numpy()), mx.array(idx.numpy().astype(np.int64)),
            num_weights, padding_idx, scale)
        _cmp(f"emb_dense_bwd W{num_weights} f{feat} pad{padding_idx} scale={scale}",
             my, ref)


def test_select_backward():
    torch.manual_seed(7)
    cases = [
        ((2, 8, 64), 2, 3),
        ((4, 6), 1, 2),
        ((10,), 0, 9),
        ((4,), 0, 1),
        ((3, 5), -1, 4),
    ]
    for shape, dim, index in cases:
        g_shape = list(shape)
        del g_shape[dim]
        g = torch.randn(*g_shape) if g_shape else torch.randn(1)[0:0] + torch.zeros(())
        g = torch.randn(*g_shape) if g_shape else torch.zeros(())
        ref = torch.ops.aten.select_backward(g, list(shape), dim, index)
        my = mlx_vulkan.ATEN_TABLE["select_backward"](
            mx.array(g.numpy()), list(shape), dim, index)
        _cmp(f"select_backward {shape} dim={dim} idx={index}", my, ref)


def test_relu_inplace_and_boolean_dispatch():
    """PT-M salvage (superseded #812): the relu_/relu_ aten lowerings match
    CPU and sit in the always-lower gate, and a boolean_dispatch'd
    F.max_pool2d(return_indices=True) call unwraps to the inner aten
    function so the executor lowers it instead of CPU-falling back."""
    torch.manual_seed(7)
    x = torch.randn(2, 4, 16, 16)
    ref = torch.relu(x)
    _cmp("relu", mlx_vulkan.ATEN_TABLE["relu"](mx.array(x.numpy())), ref)
    _cmp("relu_", mlx_vulkan.ATEN_TABLE["relu_"](mx.array(x.numpy())), ref)
    assert {"relu", "relu_"} <= mlx_vulkan._ALWAYS_LOWER

    tgt = mlx_vulkan._bd_inner(
        F.max_pool2d, (x,), {"kernel_size": 2, "stride": 2, "return_indices": True})
    assert tgt is not F.max_pool2d, "F.max_pool2d is not boolean_dispatch'd on this torch"
    tn = getattr(tgt, "__name__", "")
    assert "max_pool2d" in tn, tn
    lowered = mlx_vulkan.FUNC_TABLE_BY_NAME.get(tn) or mlx_vulkan.ATEN_TABLE.get(tn)
    assert lowered is not None, f"no lowering for unwrapped {tn}"
    out, idx = F.max_pool2d(x, 2, return_indices=True)
    my_out, my_idx = lowered(mx.array(x.numpy()), 2, 2, 0, 1, False)
    _cmp("bd.max_pool2d", my_out, out)
    assert np.array_equal(np.array(my_idx), idx.numpy())


if __name__ == "__main__":
    test_native_layer_norm_backward()
    test_embedding_dense_backward()
    test_select_backward()
    test_flash_cpu_fwd()
    test_flash_cpu_bwd()
    test_flash_cpu_bwd_gqa()
    test_slice_backward()
    test_slice_backward_clamp()
    test_flash_cpu_causal_rect()
    test_silu_backward()
    test_flash_cpu_fwd_gqa()
    test_flash_cpu_mask_and_dtype()
    test_efficient_attention_arity_and_causal()
    test_flash_cpu_bwd_dtype()
    test_efficient_attention_forward_schema()
    test_no_duplicate_table_keys()
    test_relu_inplace_and_boolean_dispatch()
    print("ALL PASS")


