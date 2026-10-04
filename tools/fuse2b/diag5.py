#!/usr/bin/env python3
"""diag5: read back the push-constant dump from y (bf16-bit patterns)."""
import mlx.core as mx
import numpy as np

B, HV, HK, DK, DV = 1, 16, 16, 128, 128
KEY_DIM, VALUE_DIM = HK * DK, HV * DV
C, K = 2 * KEY_DIM + VALUE_DIM, 4

rng = np.random.default_rng(3)
conv_state = mx.array(rng.standard_normal((B, K - 1, C)).astype(np.float32)).astype(mx.bfloat16)
state = mx.array(rng.standard_normal((B, HV, DV, DK)).astype(np.float32))
qkv = mx.array(rng.standard_normal((B, 1, C)).astype(np.float32)).astype(mx.bfloat16)
w = mx.array(rng.standard_normal((C, K, 1)).astype(np.float32)).astype(mx.bfloat16)
a = mx.array(rng.standard_normal((B, 1, HV)).astype(np.float32)).astype(mx.bfloat16)
b = mx.array(rng.standard_normal((B, 1, HV)).astype(np.float32)).astype(mx.bfloat16)
A_log = mx.array((-rng.random(HV)).astype(np.float32)).astype(mx.bfloat16)
dt_bias = mx.array(rng.standard_normal(HV).astype(np.float32)).astype(mx.bfloat16)
mx.eval(conv_state, state, qkv, w, a, b, A_log, dt_bias)

f_out, f_ring, f_state = mx.fast.gdn_conv_delta_update(
    conv_state, state, qkv, w, a, b, A_log, dt_bias,
    activate=True, qk_key_dim=KEY_DIM, qk_scale_q=DK ** -1.0,
    qk_scale_k=DK ** -0.5, qk_eps=1e-6)
mx.eval(f_out)

o = np.array(f_out.astype(mx.float32), dtype=np.float32)
expect = {
    "count=B*C=6144": 6144.0,
    "operation=K=4": 4.0,
    "lhs_size=K-1=3": 3.0,
    "rhs_size=qk_key_dim=2048": 2048.0,
    "reduce_size=C=6144": 6144.0,
    "matrix_m=Dk=128": 128.0,
    "matrix_k=Hv=16": 16.0,
    "flags": None,
}
names = list(expect)
for g in range(2):
    vals = o[g * 8:(g + 1) * 8]
    print(f"WG{g}:", {n: float(v) for n, v in zip(names, vals)})
print("expected:", {n: v for n, v in expect.items() if v is not None}, "(flags=8|2=10)")
