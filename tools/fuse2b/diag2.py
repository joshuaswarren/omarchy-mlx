#!/usr/bin/env python3
"""Fuse2B diag v2: is the fused kernel reading garbage (input readiness)?
Probes: raw output values, with and without pre-evaluated inputs."""
import mlx.core as mx
import numpy as np

B, HV, HK, DK, DV = 1, 16, 16, 128, 128
KEY_DIM, VALUE_DIM = HK * DK, HV * DV
C, K = 2 * KEY_DIM + VALUE_DIM, 4


def build():
    import random
    rng = np.random.default_rng(12345)
    conv_state = mx.array(rng.standard_normal((B, K - 1, C)).astype(np.float32)).astype(mx.bfloat16)
    state = mx.array(rng.standard_normal((B, HV, DV, DK)).astype(np.float32))
    qkv = mx.array(rng.standard_normal((B, 1, C)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.standard_normal((C, K, 1)).astype(np.float32)).astype(mx.bfloat16)
    a = mx.array(rng.standard_normal((B, 1, HV)).astype(np.float32)).astype(mx.bfloat16)
    b = mx.array(rng.standard_normal((B, 1, HV)).astype(np.float32)).astype(mx.bfloat16)
    A_log = mx.array((-rng.random(HV)).astype(np.float32)).astype(mx.bfloat16)
    dt_bias = mx.array(rng.standard_normal(HV).astype(np.float32)).astype(mx.bfloat16)
    return conv_state, state, qkv, w, a, b, A_log, dt_bias


for pre in (False, True):
    conv_state, state, qkv, w, a, b, A_log, dt_bias = build()
    if pre:
        mx.eval(conv_state, state, qkv, w, a, b, A_log, dt_bias)
    f_out, f_ring, f_state = mx.fast.gdn_conv_delta_update(
        conv_state, state, qkv, w, a, b, A_log, dt_bias,
        activate=True, qk_key_dim=KEY_DIM, qk_scale_q=DK ** -1.0,
        qk_scale_k=DK ** -0.5, qk_eps=1e-6)
    mx.eval(f_out, f_ring, f_state)
    print("pre-eval:", pre)
    print("  f_out[0,0,:4]  =", np.array(f_out[0, 0, :4].astype(mx.float32)))
    print("  f_ring[0,0,:4] =", np.array(f_ring[0, 0, :4].astype(mx.float32)))
    print("  f_state[0,0,0,:4] =", np.array(f_state[0, 0, 0, :4]))
    print("  f_out nonzero:", int(np.count_nonzero(np.array(f_out.astype(mx.float32)))))
    print("  f_ring nonzero:", int(np.count_nonzero(np.array(f_ring.astype(mx.float32)))))
    print("  f_state nonzero:", int(np.count_nonzero(np.array(f_state.astype(mx.float32)))))
