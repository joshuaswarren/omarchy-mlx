#!/usr/bin/env python3
"""PAR-5 loss lowering tests: each lowered op vs torch, maxdiff <= 1e-5."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F
import mlx.core as mx
import mlx_vulkan

TOL = 1e-5


def _check(name, low_args, fn):
    tt = fn(*low_args)
    my = mlx_vulkan.FUNC_TABLE_BY_NAME[name](*[mx.array(a.numpy()) if isinstance(a, torch.Tensor) else a for a in low_args])
    my = torch.from_numpy(np.array(my))
    d = float((my - tt).abs().max())
    assert d <= TOL, f"{name}: maxdiff {d:.3e} > {TOL}"
    print(f"PASS {name}: maxdiff={d:.3e}")


def test_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(64, 256)
    tgt = torch.randint(0, 256, (64,))
    for red in ("none", "mean", "sum"):
        _check("cross_entropy", (logits, tgt), lambda l, t: F.cross_entropy(l, t, reduction=red))
    # default mean path via kwargs
    my = mlx_vulkan.FUNC_TABLE_BY_NAME["cross_entropy"](mx.array(logits.numpy()), mx.array(tgt.numpy()))
    assert abs(float(np.array(my)) - float(F.cross_entropy(logits, tgt))) <= TOL


def test_nll_loss():
    torch.manual_seed(1)
    logp = torch.log_softmax(torch.randn(64, 256), dim=-1)
    tgt = torch.randint(0, 256, (64,))
    for red in ("none", "mean", "sum"):
        _check("nll_loss", (logp, tgt), lambda l, t: F.nll_loss(l, t, reduction=red))


def test_mse_loss():
    torch.manual_seed(2)
    a = torch.randn(8, 32)
    b = torch.randn(8, 32)
    for red in ("none", "mean", "sum"):
        _check("mse_loss", (a, b), lambda x, y: F.mse_loss(x, y, reduction=red))


if __name__ == "__main__":
    # reduction is passed per-call; exercise it via kwargs here
    import functools
    for name, args, fn in [
        ("cross_entropy", (torch.randn(64, 256), torch.randint(0, 256, (64,))), F.cross_entropy),
        ("nll_loss", (torch.log_softmax(torch.randn(64, 256), -1), torch.randint(0, 256, (64,))), F.nll_loss),
        ("mse_loss", (torch.randn(8, 32), torch.randn(8, 32)), F.mse_loss),
    ]:
        for red in ("none", "mean", "sum"):
            my = mlx_vulkan.FUNC_TABLE_BY_NAME[name](
                *[mx.array(a.numpy()) for a in args], reduction=red)
            my = torch.from_numpy(np.array(my))
            tt = fn(*args, reduction=red)
            d = float((my - tt).abs().max())
            assert d <= TOL, f"{name}[{red}]: {d:.3e} > {TOL}"
            print(f"PASS {name}[{red}] maxdiff={d:.3e}")
    print("ALL PASS")
