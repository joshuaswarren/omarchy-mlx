"""Exactness check (jwm1 2026-09-28): does mx.fast.rms_norm_gated / rms_norm_scaled match the composed mlx-lm path? Exit 1 on a systematic deviation (rate > 1e-4).

Compares, on bf16 GDN-shaped inputs (16 rows x 128 and 1 row x 2048, model-like scales):
  composed  = mlx_lm qwen3_next._precise_swiglu(h, z, mx.fast.rms_norm(h, w, eps))
  fused     = mx.fast.rms_norm_gated(h, z, w, eps)
  ref_f32   = all-f32 chain from the bf16-rounded normed value (no sigmoid/silu rounding)
  ref_bf16r = same but sigmoid and silu rounded through bf16 (what the fused shader does)
Prints mismatch counts (bit patterns) of fused and each ref against composed.
"""
import sys
import mlx.core as mx
from mlx_lm.models.qwen3_next import _precise_swiglu

mx.random.seed(0)
eps = 1e-6


def bits(a):
    return a.astype(mx.bfloat16).view(mx.uint16)


def run(rows, dim, n_trials=20):
    tot = dict(fused=0, ref_f32=0, ref_bf16r=0, n=0)
    for t in range(n_trials):
        h = (mx.random.normal((rows, dim)) * 3.0).astype(mx.bfloat16)
        z = (mx.random.normal((rows, dim)) * 2.0).astype(mx.bfloat16)
        w = (mx.random.normal((dim,)) * 0.3 + 1.0).astype(mx.bfloat16)
        x = mx.fast.rms_norm(h, w, eps)
        composed = _precise_swiglu(h, z, x)
        fused = mx.fast.rms_norm_gated(h, z, w, eps)
        zf = z.astype(mx.float32)
        xf = x.astype(mx.float32)
        s = mx.sigmoid(zf)
        ref_f32 = ((zf * s) * xf).astype(mx.bfloat16)
        sb = s.astype(mx.bfloat16).astype(mx.float32)
        silu_b = (zf * sb).astype(mx.bfloat16).astype(mx.float32)
        ref_bf16r = (silu_b * xf).astype(mx.bfloat16)
        mx.eval(composed, fused, ref_f32, ref_bf16r)
        bc = bits(composed)
        tot["fused"] += int(mx.sum(bits(fused) != bc).item())
        tot["ref_f32"] += int(mx.sum(bits(ref_f32) != bc).item())
        tot["ref_bf16r"] += int(mx.sum(bits(ref_bf16r) != bc).item())
        tot["n"] += rows * dim
    print(f"rows={rows} dim={dim}: n={tot['n']} mismatches vs composed: fused={tot['fused']} ref_f32={tot['ref_f32']} ref_bf16r={tot['ref_bf16r']}")
    return tot["fused"] / tot["n"]


if not hasattr(mx.fast, "rms_norm_gated"):
    sys.exit("no rms_norm_gated in this build")
def run_scaled(rows, dim, n_trials=20):
    """q/k site: (inv_scale**2) * rms_norm(q, None, 1e-6) vs rms_norm_scaled."""
    inv = 128 ** -0.5
    bad = 0
    for _ in range(n_trials):
        q = (mx.random.normal((rows, dim)) * 3.0).astype(mx.bfloat16)
        composed = (inv**2) * mx.fast.rms_norm(q, None, 1e-6)
        fused = mx.fast.rms_norm_scaled(q, None, inv * inv, 1e-6)
        mx.eval(composed, fused)
        bad += int(mx.sum(bits(fused) != bits(composed)).item())
    print(f"scaled rows={rows} dim={dim}: n={n_trials * rows * dim} mismatches vs composed: {bad}")
    return bad / (n_trials * rows * dim)


# Fails on a systematic deviation (the pre-fix bug flipped ~36% of outputs). A rare ~1e-5
# last-bit epilogue deviation of the fused gated kernel still exists at every shape checked
# (see receipts/2026-09-28-jwm1-gdn-qk-scaled.md), so the bar is a rate, not exact zero.
LIMIT = 1e-4
rates = [run_scaled(rows, 128) for rows in (16, 32, 64, 128, 256)]
rates += [run(rows, dim) for rows, dim in ((16, 128), (1, 2048), (2, 2048), (7, 128), (32, 128), (64, 128), (256, 128))]
print("max mismatch rate:", max(rates), "limit:", LIMIT)
sys.exit(1 if max(rates) > LIMIT else 0)
