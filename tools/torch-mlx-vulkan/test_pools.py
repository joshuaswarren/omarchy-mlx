"""Op parity tests for the pool lowerings in mlx_vulkan.py.

Compares each lowered pool op (fwd + bwd) against the torch CPU
reference. Run with the backend's venv:

    python test_pools.py

Covers the review cases: ceil_mode windows that overrun the padded
input (w74: k3 s2 p1 on (2,3,8,10); w7W: k2 s2 p0 on 9x9) and
adaptive_avg_pool2d output_size=None entries.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mlx.core as mx
import mlx_vulkan as mv

import torch
import torch.nn.functional as F

FAILS = []


def _check(name, mx_arr, torch_t, tol=3e-6):
    got = np.asarray(mx_arr.astype(mx.float32))
    want = torch_t.detach().cpu().numpy().astype(np.float32)
    if got.shape != want.shape:
        FAILS.append(f"{name}: shape {got.shape} != {want.shape}")
        print(f"FAIL {name}: shape {got.shape} != {want.shape}")
        return
    d = float(np.abs(got - want).max()) if got.size else 0.0
    ok = d <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: maxdiff {d:.3e}")
    if not ok:
        FAILS.append(f"{name}: maxdiff {d:.3e}")


def _max_pool_case(shape, k, s, p, ceil):
    x = torch.randn(*shape, dtype=torch.float32)
    g = torch.randn(*F.max_pool2d(x, k, s, p, 1, ceil).shape)
    xt = x.clone().requires_grad_(True)
    out_t = F.max_pool2d(xt, k, s, p, 1, ceil)
    out_t.backward(g)
    tag = f"maxpool {shape} k{k} s{s} p{p} ceil={ceil}"
    _check(tag + " fwd", mv._max_pool2d(mx.array(x.numpy()), k, s, p, 1,
                                        ceil), out_t)
    _check(tag + " bwd", mv._max_pool2d_bwd(mx.array(g.numpy()),
                                            mx.array(x.numpy()), k, s, p,
                                            1, ceil), xt.grad)


def _avg_pool_case(shape, k, s, p, ceil, include_pad):
    x = torch.randn(*shape, dtype=torch.float32)
    ref = F.avg_pool2d(x, k, s, p, ceil, include_pad)
    g = torch.randn(*ref.shape)
    xt = x.clone().requires_grad_(True)
    out_t = F.avg_pool2d(xt, k, s, p, ceil, include_pad)
    out_t.backward(g)
    tag = f"avgpool {shape} k{k} s{s} p{p} ceil={ceil} inc={include_pad}"
    _check(tag + " fwd", mv._avg_pool2d(mx.array(x.numpy()), k, s, p,
                                        ceil, include_pad), out_t)
    _check(tag + " bwd", mv._avg_pool2d_backward(mx.array(g.numpy()),
                                                 mx.array(x.numpy()), k, s,
                                                 p, ceil, include_pad),
           xt.grad)


def main():
    torch.manual_seed(0)

    # w74 review case: ceil window overruns the padded input.
    for include_pad in (True, False):
        _avg_pool_case((2, 3, 8, 10), 3, 2, 1, True, include_pad)
    _max_pool_case((2, 3, 8, 10), 3, 2, 1, True)

    # w7W case: 9x9, k2 s2 p0, ceil last window overruns by one.
    for include_pad in (True, False):
        _avg_pool_case((1, 1, 9, 9), 2, 2, 0, True, include_pad)
    _max_pool_case((1, 1, 9, 9), 2, 2, 0, True)

    # Sweep: k2/k3, s1-s3, p0-1, ceil on/off.
    for k in (2, 3):
        for s in (1, 2, 3):
            for p in (0, 1):
                for ceil in (False, True):
                    if p * 2 >= k:
                        continue
                    _max_pool_case((1, 2, 8, 9), k, s, p, ceil)
                    _avg_pool_case((1, 2, 8, 9), k, s, p, ceil, True)
                    _avg_pool_case((1, 2, 8, 9), k, s, p, ceil, False)

    # adaptive_avg_pool2d: output sizes and None entries.
    x = torch.randn(2, 3, 7, 9)
    for os_ in ((2, 2), (1, 1), (7, 9), (None, 3), (4, None), (None, None)):
        out_t = F.adaptive_avg_pool2d(x, os_)
        _check(f"adaptive os={os_}", mv._adaptive_avg_pool2d(
            mx.array(x.numpy()), os_), out_t)

    print(f"\n{len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
