#!/usr/bin/env python3
"""diag3: exact nonzero indices + input-vs-output state comparison."""
import mlx.core as mx
import numpy as np

B, HV, HK, DK, DV = 1, 16, 16, 128, 128
KEY_DIM, VALUE_DIM = HK * DK, HV * DV
C, K = 2 * KEY_DIM + VALUE_DIM, 4

rng = np.random.default_rng(12345)
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
mx.eval(f_out, f_ring, f_state)

np.set_printoptions(precision=3, suppress=True, linewidth=220)
o = np.array(f_out.astype(mx.float32))
nz = np.nonzero(o)
print("out nonzero idx (first 24):", list(zip(nz[1], nz[2], nz[3]))[:24], "count", len(nz[0]))
print("state in [0,0,0,:4]:", np.array(state[0, 0, 0, :4]))
print("state out[0,0,0,:4]:", np.array(f_state[0, 0, 0, :4]))
print("state out[0,1,0,:4]:", np.array(f_state[0, 1, 0, :4]))
print("state out[0,15,0,:4]:", np.array(f_state[0, 15, 0, :4]))
sd = np.abs(np.array(f_state.astype(mx.float32)) - np.array(state.astype(mx.float32)))
print("state |out-in| per head mean:", sd.reshape(B, HV, -1).mean(-1))
r = np.array(f_ring.astype(mx.float32))
print("ring nonzero count:", int(np.count_nonzero(r)), "of", r.size)
print("ring[:8]:", r.flat[:8])
print("ring[1024:1032]:", r.flat[1024:1032])
