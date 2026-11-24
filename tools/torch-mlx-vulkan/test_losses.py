#!/usr/bin/env python3
"""PAR-5 loss lowering tests: each lowered op vs torch, allclose(1e-5, 1e-5)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
import mlx_vulkan

TOL = 1e-5


def _check(name, low_args, fn, kwargs=None):
    kwargs = kwargs or {}
    tt = fn(*low_args)
    my = mlx_vulkan.ATEN_TABLE[name.split("[")[0]](
        *[mx.array(a.numpy()) if isinstance(a, torch.Tensor) else a for a in low_args],
        **kwargs)
    my = torch.from_numpy(np.array(my))
    d = float((my - tt).abs().max())
    assert torch.allclose(my, tt, atol=TOL, rtol=TOL), f"{name}: maxdiff {d:.3e} > {TOL}"
    print(f"PASS {name}: maxdiff={d:.3e}")


def test_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(64, 256)
    tgt = torch.randint(0, 256, (64,))
    for red in ("none", "mean", "sum"):
        _check("cross_entropy", (logits, tgt),
               lambda l, t: F.cross_entropy(l, t, reduction=red),
               {"reduction": red})
    w = torch.rand(256) + 0.5
    for red in ("none", "mean", "sum"):
        _check(f"cross_entropy[weight={red}]", (logits, tgt),
               lambda l, t: F.cross_entropy(l, t, weight=w, reduction=red),
               {"weight": mx.array(w.numpy()), "reduction": red})


def test_nll_loss():
    torch.manual_seed(1)
    logp = torch.log_softmax(torch.randn(64, 256), dim=-1)
    tgt = torch.randint(0, 256, (64,))
    for red in ("none", "mean", "sum"):
        _check("nll_loss", (logp, tgt),
               lambda l, t: F.nll_loss(l, t, reduction=red),
               {"reduction": red})


def test_mse_loss():
    torch.manual_seed(2)
    a = torch.randn(8, 32)
    b = torch.randn(8, 32)
    for red in ("none", "mean", "sum"):
        _check("mse_loss", (a, b),
               lambda x, y: F.mse_loss(x, y, reduction=red),
               {"reduction": red})


if __name__ == "__main__":
    test_cross_entropy()
    test_nll_loss()
    test_mse_loss()
    print("ALL PASS")
