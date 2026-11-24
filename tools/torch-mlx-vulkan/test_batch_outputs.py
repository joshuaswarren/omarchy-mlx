"""Batch>1 output-shape regression tests for the mlx_vulkan backend.

Regression for omarchy-mlx #814 / #815: the compiled executor returned a
bare tensor for single-output graphs, and the caller-side unpack sliced it
over dim 0 — a (B,C,H,W) conv result came back as the batch-0 (C,H,W)
slice. Masked at B=1 (singleton slice is invisible) and by (1,1,V) decode
logits. w74's #814 repro was exactly this: F.conv2d (2,3,16,16) in ->
(4,16,16) back, only when the conv is the graph's whole output.

Run with the backend's venv:

    python test_batch_outputs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx_vulkan as mv  # noqa: F401  (registers backend "mlx_vulkan")

import torch
import torch.nn as nn
import torch.nn.functional as F

FAILS = []


def _check(name, out, ref, tol=2e-2):
    if tuple(out.shape) != tuple(ref.shape):
        FAILS.append(f"{name}: shape {tuple(out.shape)} != {tuple(ref.shape)}")
        print(f"FAIL {name}: shape {tuple(out.shape)} != {tuple(ref.shape)}")
        return
    d = float((out.float() - ref.float()).abs().max()) if out.numel() else 0.0
    ok = d <= tol
    print(f"{'ok  ' if ok else 'FAIL'} {name}: maxdiff {d:.3e}")
    if not ok:
        FAILS.append(f"{name}: maxdiff {d:.3e}")


def main():
    torch.manual_seed(0)
    # w74's exact repro: output-is-the-conv single-output graph.
    w = torch.randn(4, 3, 3, 3)
    x = torch.randn(2, 3, 16, 16)
    fn = lambda t: F.conv2d(t, w, padding=1)
    f = torch.compile(fn, backend="mlx_vulkan", dynamic=False)
    _check("w74 repro conv (2,3,16,16)", f(x), fn(x))

    # batch sweep on a module conv: the leading dim must survive.
    net = nn.Conv2d(3, 16, 3, padding=1)
    fc = torch.compile(net, backend="mlx_vulkan", dynamic=False)
    for bs in (1, 2, 4, 8):
        xb = torch.randn(bs, 3, 32, 32)
        _check(f"conv module bs={bs}", fc(xb), net(xb))

    # conv output feeding another op (regression guard: intermediates).
    net2 = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(4, 4, 3, padding=1))
    x2 = torch.randn(2, 4, 16, 16)
    f2 = torch.compile(net2, backend="mlx_vulkan", dynamic=False)
    _check("conv-relu-conv (2,4,16,16)", f2(x2), net2(x2))

    print(f"\n{len(FAILS)} failures")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
