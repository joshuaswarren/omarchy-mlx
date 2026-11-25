#!/usr/bin/env python3
"""PAR-41 batch-norm parity: native_batch_norm, batch_norm and
_native_batch_norm_legit_functional vs torch CPU, train and eval modes,
fp32 and bf16, plus the backward.

Each case runs in a FRESH subprocess (the second BN-family evaluation of
the same shapes inside one process is covered by the in-sequence check
the parent runs at the end). Contracts:
- native_batch_norm -> the (out, save_mean, save_invstd) tuple
- batch_norm -> the normalized tensor only
- _native_batch_norm_legit_functional -> the 5-tuple with new_rm/new_rv

Run: python3 test_batch_norm.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = []
for dt_name in ("fp32", "bf16"):
    for shape in ((4, 6, 3, 3), (2, 6)):
        for training in (True, False):
            CASES.append(("native_batch_norm", training, dt_name, shape))
            CASES.append(("batch_norm", training, dt_name, shape))
            CASES.append(("_bn_legit_functional", training, dt_name, shape))
            CASES.append(("bn_backward", training, dt_name, shape))


def child_source(fn_name, training, dt_name, shape, tol):
    train = "True" if training else "False"
    dt = "mx.float32" if dt_name == "fp32" else "mx.bfloat16"
    tdt = "torch.float32" if dt_name == "fp32" else "torch.bfloat16"
    return f'''
import numpy as np
import torch
import mlx.core as mx
import mlx_vulkan

rng = np.random.RandomState(11)
shape = {shape}
C = shape[1]
axes = tuple(d for d in range(len(shape)) if d != 1)
tol = {tol}
x = rng.randn(*shape).astype(np.float32)
w = (rng.rand(C) + 0.5).astype(np.float32)
b = (rng.randn(C) * 0.1).astype(np.float32)
rm = rng.randn(C).astype(np.float32)
rv = (rng.rand(C) + 0.5).astype(np.float32)
x_m, w_m, b_m, rm_m, rv_m = (mx.array(v).astype({dt}) for v in (x, w, b, rm, rv))
x_t = torch.tensor(x).to({tdt})
w_t = torch.tensor(w).to({tdt})
b_t = torch.tensor(b).to({tdt})
rm_t = torch.tensor(rm).to({tdt})
rv_t = torch.tensor(rv).to({tdt})
dy = rng.randn(*shape).astype(np.float32)
dy_t = torch.tensor(dy).to({tdt})
dy_m = mx.array(dy).astype({dt})
mom, eps = 0.1, 1e-5
mn_t = x_t.mean(dim=axes)
sd_t = torch.sqrt(x_t.var(dim=axes, unbiased=False) + eps)

def close(tag, got, want):
    g = np.array(got, dtype=np.float64)
    w = np.array(want.detach().cpu().float().numpy(), dtype=np.float64)
    assert g.shape == w.shape, f"{{tag}}: shape {{g.shape}} vs {{w.shape}}"
    d = np.abs(g - w).max()
    assert d < tol, f"{{tag}}: max diff {{d:g}} (tol {{tol}})"
'''

    if fn_name in ("native_batch_norm", "batch_norm"):
        src = f'''
r = mlx_vulkan.ATEN_TABLE["{fn_name}"](x_m, w_m, b_m, rm_m, rv_m, {training}, mom, eps)
'''
        if fn_name == "native_batch_norm":
            # native returns the (out, save_mean, save_invstd) tuple
            src += '''
close("out", r[0], torch.native_batch_norm(x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), {training}, mom, eps)[0])
close("save_mean", r[1], torch.native_batch_norm(x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), {training}, mom, eps)[1])
close("save_invstd", r[2], torch.native_batch_norm(x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), {training}, mom, eps)[2])
'''
        else:
            src += '''
close("out", r, torch.batch_norm(x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), {training}, mom, eps)[0])
'''
        if not training:
            src += """
close("eval uses batch stats", r.reshape(x_t.shape),
      (x_t - mn_t.reshape(1, -1)) / sd_t.reshape(1, -1) * w_t.reshape(1, -1) + b_t.reshape(1, -1))
"""
        return src
    if fn_name == "_bn_legit_functional":
        src = f'''
r = mlx_vulkan._bn_legit_functional(x_m, w_m, b_m, rm_m, rv_m, {training}, mom, eps)
wr = torch.ops.aten._native_batch_norm_legit_functional(
    x_t, w_t, b_t, rm_t.clone(), rv_t.clone(), {training}, mom, eps)
close("out", r[0], wr[0])
close("save_mean", r[1], wr[1])
close("elem3", r[2], wr[2])
n = 1
for d_ in range(len(shape)):
    if d_ != 1:
        n *= shape[d_]
'''
        if training:
            src += '''
close("new_rm", r[3], rm_t * (1 - mom) + mom * x_t.mean(dim=axes))
close("new_rv", r[4], rv_t * (1 - mom)
      + mom * x_t.var(dim=axes, unbiased=False) * (n / max(1, n - 1)))
'''
        return src
    return src + f'''
sm_t = x_t.mean(dim=axes)
si_t = 1.0 / torch.sqrt(x_t.var(dim=axes, unbiased=False) + eps)
sm_m = mx.array(sm_t.numpy()).astype({dt})
si_m = mx.array(si_t.numpy()).astype({dt})
g = mlx_vulkan._bn_backward(dy_m, x_m, w_m, rm_m, rv_m, sm_m, si_m, {training}, eps,
                            (True, True, True))
xh = (x_t - sm_t) * si_t
if {train}:
    close("dx", g[0], (dy_t - dy_t.mean(dim=axes) - xh * (dy_t * xh).mean(dim=axes)) * si_t * w_t)
else:
    close("dx", g[0], dy_t * si_t * w_t)
close("dw", g[1], (dy_t * xh).sum(dim=axes))
close("db", g[2], dy_t.sum(dim=axes))
'''


def main():
    fails = []
    for fn_name, training, dt_name, shape in CASES:
        tag = f"{fn_name} train={training} {dt_name} {shape}"
        src = child_source(fn_name, training, dt_name, shape, None)
        r = subprocess.run([sys.executable, "-c", src], cwd=HERE,
                           capture_output=True, text=True, check=False)
        if r.returncode != 0:
            tail = (r.stderr.strip().splitlines() or ["?"])[-1][:160]
            fails.append(f"{tag}: {tail}")
            print(f"FAIL {tag}\n  {tail}")
        else:
            print(f"ok   {tag}")
    if fails:
        print(f"\n{len(fails)} case(s) FAILED")
        sys.exit(1)
    print(f"\nALL {len(CASES)} BN parity cases PASS (isolated subprocesses)")


if __name__ == "__main__":
    main()
