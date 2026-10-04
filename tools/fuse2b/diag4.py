#!/usr/bin/env python3
"""diag4: store-only bisect check. The dirty wheel's kernel only copies
conv_state rows and the x row into the ring and x into y. If those do not
land, the dispatch/store path itself is broken for this kernel."""
import mlx.core as mx
import numpy as np

B, HV, HK, DK, DV = 1, 16, 16, 128, 128
KEY_DIM, VALUE_DIM = HK * DK, HV * DV
C, K = 2 * KEY_DIM + VALUE_DIM, 4

rng = np.random.default_rng(7)
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

cs = np.array(conv_state.astype(mx.float32))
x = np.array(qkv.astype(mx.float32))
r = np.array(f_ring.astype(mx.float32))
o = np.array(f_out.astype(mx.float32))
# ring rows 0..2 = conv_state rows 0..2 (channels head*128..): check head 0
seg = r[0, 0, :128]
ref = cs[0, 0, :128]
print("ring[0,:128]==state[0,:128]:", bool(np.all(seg == ref)))
seg2 = r[0, 0, 9216:9216+128]
print("ring[9216:9344]==x[0,:128]:", bool(np.all(seg2 == x[0, 0, :128])))
# y: head*128+row over rows 0..127 (grid y covers 4 blocks x 32)
ok_y = True
for h in range(HV):
    if not np.all(o[0, 0, h, :] == x[0, 0, h * 128:(h + 1) * 128]):
        ok_y = False
        break
print("y==x pattern:", ok_y)
print("ring nonzero:", int(np.count_nonzero(r)), "y nonzero:", int(np.count_nonzero(o)))
