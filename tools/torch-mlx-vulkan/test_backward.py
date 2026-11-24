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
    return torch.from_numpy(np.array(a))


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


if __name__ == "__main__":
    test_native_layer_norm_backward()
    test_flash_cpu_fwd()
    test_flash_cpu_bwd()
    print("ALL PASS")
