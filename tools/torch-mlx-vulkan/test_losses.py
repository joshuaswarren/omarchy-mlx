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


def test_cross_entropy_edge():
    torch.manual_seed(1)
    logits = torch.randn(6, 8)
    tgt = torch.randint(0, 8, (6,))
    w = torch.rand(8)
    # ignore_index: default -100 masks masked targets; custom value too.
    tgt_ig = tgt.clone(); tgt_ig[1] = -100; tgt_ig[4] = -100
    for red in ("none", "mean", "sum"):
        for wi, sm in ((None, 0.0), (w, 0.0), (w, 0.15)):
            _check(f"cross_entropy[ig w={wi is not None} sm={sm} {red}]",
                   (logits, tgt_ig),
                   lambda l, t: F.cross_entropy(l, t, weight=wi, reduction=red,
                                                label_smoothing=sm),
                   {"weight": mx.array(wi.numpy()) if wi is not None else None,
                    "reduction": red, "label_smoothing": sm})
    tgt99 = tgt.clone(); tgt99[2] = 99
    _check("cross_entropy[ig99]", (logits, tgt99),
           lambda l, t: F.cross_entropy(l, t, ignore_index=99),
           {"ignore_index": 99})
    # weight + label_smoothing: mean denom is sum(w[t]) over non-ignored only
    for red in ("none", "mean", "sum"):
        _check(f"cross_entropy[w+sm {red}]", (logits, tgt),
               lambda l, t: F.cross_entropy(l, t, weight=w, reduction=red,
                                            label_smoothing=0.15),
               {"weight": mx.array(w.numpy()), "reduction": red,
                "label_smoothing": 0.15})
    # 4-D NCHW: class dim is 1, not last
    l4 = torch.randn(2, 8, 3, 4)
    t4 = torch.randint(0, 8, (2, 3, 4)); t4[0, 1, 2] = -100
    _check("cross_entropy[4d w+sm]", (l4, t4),
           lambda l, t: F.cross_entropy(l, t, weight=w, label_smoothing=0.1),
           {"weight": mx.array(w.numpy()), "label_smoothing": 0.1})


def test_nll_loss_edge():
    torch.manual_seed(1)
    logp = F.log_softmax(torch.randn(6, 8), dim=-1)
    tgt = torch.randint(0, 8, (6,)); tgt[1] = -100
    w = torch.rand(8)
    for red in ("none", "mean", "sum"):
        for wi in (None, w):
            _check(f"nll_loss[ig w={wi is not None} {red}]", (logp, tgt),
                   lambda l, t: F.nll_loss(l, t, weight=wi, reduction=red),
                   {"weight": mx.array(wi.numpy()) if wi is not None else None,
                    "reduction": red})


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
    test_cross_entropy_edge()
    test_nll_loss_edge()
    test_mse_loss()
    print("ALL PASS")
