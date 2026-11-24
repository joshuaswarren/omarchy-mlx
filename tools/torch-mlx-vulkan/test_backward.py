#!/usr/bin/env python3
"""PAR-5.2 backward lowering parity tests vs CPU aten ops (allclose 1e-5/1e-5)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
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


if __name__ == "__main__":
    test_native_layer_norm_backward()
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
    print("ALL PASS")
