#!/usr/bin/env python3
"""PT-AD softmax parity: aten._softmax / aten._log_softmax (the exported
pair behind F.softmax) and the public softmax/log_softmax with their dtype
kwarg, vs torch CPU 2.15. Last-dim and dim=1 shapes; fp32, bf16, and the
fp32-accumulation paths (half/half_to_float, dtype=fp32). The lowerings do
an explicit max-subtract with fp32 accumulation for half inputs so the
reduction order is pinned.

bf16 cases compare with torch.allclose(atol=1e-2, rtol=1e-2): two
fp32-ordered reductions can straddle one bf16 quantization boundary, and
the rtol admits exactly that single-quantum straddle.

Run: python3 test_softmax.py   (prints PASS lines; exits nonzero on the
first failing group with its max diff).
"""
import mlx.core as mx
import numpy as np
import torch

import mlx_vulkan


def _np(v):
    if v.dtype == mx.bfloat16:
        return np.asarray(v.astype(mx.float32), dtype=np.float32)
    return np.array(v, dtype=np.float32)


def _pair(rng, shape, bf16):
    x = rng.randn(*shape).astype(np.float32) * 3.0  # *3: exercise the max-subtract
    t = torch.tensor(x)
    m = mx.array(x)
    if bf16:
        t = t.to(torch.bfloat16)
        m = m.astype(mx.bfloat16)
    return t, m


def _close(tag, got, want, tol, rtol=0.0):
    g = torch.tensor(_np(got), dtype=torch.float32)
    w = want.detach().cpu().float()
    if not torch.allclose(g, w, rtol=rtol, atol=tol):
        d = (g - w).abs().max().item()
        print(f"{tag} FAILED max diff {d:g} (atol {tol:g} rtol {rtol:g})")
        print("got :", [round(x, 5) for x in g.ravel()[:8].tolist()])
        print("want:", [round(x, 5) for x in w.ravel()[:8].tolist()])
        assert False, f"{tag}: max diff {d:g} (atol {tol} rtol {rtol})"


SHAPES = ((4, 6, 3, 3), (2, 6))


def test_softmax_parity():
    rng = np.random.RandomState(23)
    for shape in SHAPES:
        for dim in (-1, 1):
            for tag, bf16, tol in (("fp32", False, 2e-5), ("bf16", True, 1e-2)):
                kw = {"rtol": 1e-2} if bf16 else {}
                t, m = _pair(rng, shape, bf16)
                want = torch.ops.aten._softmax(t, dim, False)
                got = mlx_vulkan.ATEN_TABLE["_softmax"](m, dim)
                _close(f"_softmax {tag} {shape} dim={dim}", got, want, tol, **kw)
            # half=True: fp16/bf16 inputs compute in fp32 (htf). torch CPU
            # refuses the private-op htf flag, so reference it on the
            # fp32-upcast input - the exact htf semantics.
            t, m = _pair(rng, shape, True)
            want = torch.ops.aten._softmax(t.float(), dim, False)
            got = mlx_vulkan.ATEN_TABLE["_softmax"](m, dim, half=True)
            _close(f"_softmax htf {shape} dim={dim}", got, want, 2e-5)
            # public softmax carries the dtype kwarg (input-cast rule)
            want = torch.softmax(t, dim, dtype=torch.float32)
            got = mlx_vulkan.ATEN_TABLE["softmax"](m, dim, dtype=torch.float32)
            _close(f"softmax dtype-fp32 {shape} dim={dim}", got, want, 2e-5)
    print("PASS softmax parity")


def test_log_softmax_parity():
    rng = np.random.RandomState(29)
    for shape in SHAPES:
        for dim in (-1, 1):
            for tag, bf16, tol in (("fp32", False, 2e-5), ("bf16", True, 1e-2)):
                kw = {"rtol": 1e-2} if bf16 else {}
                t, m = _pair(rng, shape, bf16)
                want = torch.ops.aten._log_softmax(t, dim, False)
                got = mlx_vulkan.ATEN_TABLE["_log_softmax"](m, dim)
                _close(f"_log_softmax {tag} {shape} dim={dim}", got, want, tol, **kw)
            t, m = _pair(rng, shape, True)
            want = torch.ops.aten._log_softmax(t.float(), dim, False)
            got = mlx_vulkan.ATEN_TABLE["_log_softmax"](m, dim, half_to_float=True)
            _close(f"_log_softmax htf {shape} dim={dim}", got, want, 2e-5)
            want = torch.log_softmax(t, dim, dtype=torch.float32)
            got = mlx_vulkan.ATEN_TABLE["log_softmax"](m, dim, dtype=torch.float32)
            _close(f"log_softmax dtype-fp32 {shape} dim={dim}", got, want, 2e-5)
    print("PASS log_softmax parity")


if __name__ == "__main__":
    test_softmax_parity()
    test_log_softmax_parity()
    print("ALL PASS")
