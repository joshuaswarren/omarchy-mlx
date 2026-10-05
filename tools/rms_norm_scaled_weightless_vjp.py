#!/usr/bin/env python3
"""Op-level reproducer for the weightless rms_norm_scaled backward defect.

The qwen3.5 qk-scaled patch (mlx-lm-qwen35-qk-scaled) calls
mx.fast.rms_norm_scaled(q, None, inv_scale**2, eps): a WEIGHTLESS scaled
norm. Its fused forward ran fine on the omarchy backend while the
fallback fed the {1} 0-D placeholder to fast::rms_norm, which refuses
0-D weights — so the FIRST BACKWARD of every weightless scaled norm
threw "[rms_norm] (*weight) must have 1 dimension but has 0 dimensions."
This is the second LoRA blocker on the hybrid 2B (the first, the
[RoPE::vjp] fence, is fixed on agent/m2lane-rope-vjp).

Run: python rms_norm_scaled_weightless_vjp.py
OK  -> prints SCALED_WEIGHTLESS_VJP_OK with the gradient checksum.
Old -> raises the rms_norm 0-dimension ValueError (the defect).
"""
import mlx.core as mx

def main():
    if not hasattr(mx.fast, "rms_norm_scaled"):
        print("SKIP: no mx.fast.rms_norm_scaled on this build")
        return
    B, H, T, D = 1, 2, 8, 128
    x = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)
    cot = mx.random.normal((B, H, T, D)).astype(mx.bfloat16)

    def f(a):
        y = mx.fast.rms_norm_scaled(a, None, 0.7, 1e-6)
        return mx.sum(y * cot)

    lg = mx.value_and_grad(f)
    loss, gx = lg(x)
    mx.eval(loss, gx)
    flat = mx.reshape(gx, (-1,))
    checksum = float(mx.sum(mx.abs(flat)))
    finite = bool(mx.all(mx.isfinite(gx)))
    print(f"SCALED_WEIGHTLESS_VJP_OK loss={float(loss):.4f} "
          f"grad_abs_sum={checksum:.4f} finite={finite}")
    assert finite, "non-finite gradient"

if __name__ == "__main__":
    main()
