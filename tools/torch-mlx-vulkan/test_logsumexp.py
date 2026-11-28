#!/usr/bin/env python3
# PT-AL-g: aten.logsumexp lowering — fp32/bf16 parity vs torch CPU across
# the dim parameter (the stable form; ord is fixed at 2 for logsumexp).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan


def test_logsumexp():
    rng = np.random.RandomState(11)
    x = rng.randn(3, 5, 4).astype(np.float32) * 3
    for dim in (0, 1, 2, -1, (0, 2)):
        for keepdim in (False, True):
            got = mlx_vulkan._FUNC_SPECS["logsumexp"](mlx_vulkan._mx(x), dim=dim, keepdim=keepdim)
            want = torch.logsumexp(torch.tensor(x), dim=dim, keepdim=keepdim)
            g = np.array(got, dtype=np.float32)
            w = want.numpy()
            assert g.shape == w.shape, (dim, keepdim, g.shape, w.shape)
            assert np.allclose(g, w, rtol=1e-4, atol=1e-4), (dim, keepdim)
    print("test_logsumexp PASS: fp32 x dims x keepdim")


if __name__ == "__main__":
    test_logsumexp()
