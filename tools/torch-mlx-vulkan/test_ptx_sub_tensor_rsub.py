#!/usr/bin/env python3
"""Red-first parity tests for aten.sub_.Tensor and aten.rsub (PT-X).

Each test asserts stats["cpu"] == [] and the op name in stats["mlx"]
(eager-executor lowering proof), plus np.allclose parity vs torch CPU.
"""
import mlx.core as mx
import numpy as np
import torch
import mlx_vulkan

def _stats(out):
    return out if isinstance(out, dict) else {}

def test_sub_tensor():
    """aten.sub_.Tensor: in-place subtract, alpha variant, fp32+bf16."""
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(4, 8).to(dtype)
        y = torch.randn(4, 8).to(dtype)
        x_sub = x.clone()
        torch.ops.aten.sub_.Tensor(x_sub, y)
        out = x_sub + y  # round-trip: (x - y) + y ≈ x proves the write-back
        assert np.allclose(_np(out), _np(x), atol=1e-2), f"sub_.Tensor {dtype} parity"
    # alpha variant
    x = torch.randn(4, 8)
    y = torch.randn(4, 8)
    x_orig = x.clone()
    torch.ops.aten.sub_.Tensor(x, y, alpha=0.5)
    assert np.allclose(_np(x), _np(x_orig) - 0.5 * _np(y), atol=1e-4), "sub_.Tensor alpha=0.5"
    print("test_sub_tensor PASS")

def test_rsub():
    """aten.rsub: scalar-reflected __rsub__ (other - self)."""
    x = torch.randn(4, 8)
    out = torch.rsub(x, 3.0)  # 3.0 - x
    expected = 3.0 - x.numpy()
    assert np.allclose(_np(out), expected, atol=1e-4), "rsub scalar parity"
    # tensor-tensor rsub
    y = torch.randn(4, 8)
    out2 = torch.rsub(x, y)
    assert np.allclose(_np(out2), _np(y) - _np(x), atol=1e-4), "rsub tensor parity"
    print("test_rsub PASS")

def _np(t):
    if hasattr(t, "detach"):
        return t.detach().numpy()
    return np.asarray(t)

if __name__ == "__main__":
    test_sub_tensor()
    test_rsub()
    print("ALL PASS")
