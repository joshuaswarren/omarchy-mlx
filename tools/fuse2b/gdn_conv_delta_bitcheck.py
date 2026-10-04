#!/usr/bin/env python3
"""Captured-operand bitcheck for the fused GDN decode conv + state update.

F7 candidate (Fuse2B lane): mx.fast.gdn_conv_delta_update must be
bit-identical (out, new conv ring, new GDN state) to the composed chain it
replaces on the Qwen3.8-2B decode geometry:

  conv_out, ring = mx.fast.gdn_conv_update(conv_state, qkv, w, activate=True,
                                           qk_key_dim=key_dim, ...)
  q, k, v = split/reshape(conv_out)
  out, state = gated_delta_update(q, k, v, a, b, A_log, dt_bias, state, None)

Shapes (config text_config): B=1, Hv=Hk=16, Dk=Dv=128, key_dim=2048,
value_dim=2048, C=6144, conv kernel K=4. Operands are drawn from a seeded
RNG; bit-identity must hold for ANY operands, so synthetic values are a
valid captured-operand stand-in and the production digest pins carry the
in-model proof. A_log rides bf16 (the 2B checkpoint) and f32 (bit 9 route,
the 9B conversion shape) in separate cases.

usage: gdn_conv_delta_bitcheck.py OUT.json
"""
import hashlib
import json
import sys
import zlib

import mlx.core as mx
import numpy as np

out_path = sys.argv[1]
results = {}

B = 1
HV = 16
HK = 16
DK = 128
DV = 128
KEY_DIM = HK * DK
VALUE_DIM = HV * DV
C = 2 * KEY_DIM + VALUE_DIM
K = 4


def bits(a):
    if a.dtype in (mx.bfloat16, mx.float16):
        return np.array(a.view(mx.uint16)).tobytes()
    return np.array(a.view(mx.uint32)).tobytes()


def composed(conv_state, state, qkv, w, a, b, A_log, dt_bias,
             scale_q, scale_k, eps):
    conv_out, new_ring = mx.fast.gdn_conv_update(
        conv_state, qkv, w, activate=True,
        qk_key_dim=KEY_DIM, qk_scale_q=scale_q, qk_scale_k=scale_k,
        qk_eps=eps)
    parts = mx.split(conv_out, [KEY_DIM, 2 * KEY_DIM], -1)
    q = parts[0].reshape(B, 1, HK, DK)
    k = parts[1].reshape(B, 1, HK, DK)
    v = parts[2].reshape(B, 1, HV, DV)
    out, new_state = mx.fast.gated_delta_update_raw(
        q, k, v, a, b, A_log, dt_bias, state, None)
    return out, new_ring, new_state


def fused(conv_state, state, qkv, w, a, b, A_log, dt_bias,
          scale_q, scale_k, eps):
    out, new_ring, new_state = mx.fast.gdn_conv_delta_update(
        conv_state, state, qkv, w, a, b, A_log, dt_bias,
        activate=True, qk_key_dim=KEY_DIM, qk_scale_q=scale_q,
        qk_scale_k=scale_k, qk_eps=eps)
    return out, new_ring, new_state


def case(name, a_log_dtype=mx.bfloat16, seed_salt=""):
    rng = np.random.default_rng(zlib.crc32((name + seed_salt).encode()))
    scale = float(DK ** -0.5)
    conv_state = mx.array(
        rng.standard_normal((B, K - 1, C)).astype(np.float32)).astype(mx.bfloat16)
    state = mx.array(
        rng.standard_normal((B, HV, DV, DK)).astype(np.float32))
    qkv = mx.array(
        rng.standard_normal((B, 1, C)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(
        rng.standard_normal((C, K, 1)).astype(np.float32)).astype(mx.bfloat16)
    a = mx.array(
        rng.standard_normal((B, 1, HV)).astype(np.float32)).astype(mx.bfloat16)
    b = mx.array(
        rng.standard_normal((B, 1, HV)).astype(np.float32)).astype(mx.bfloat16)
    A_log_np = -rng.random(HV).astype(np.float32) * 2.0
    A_log = mx.array(A_log_np).astype(a_log_dtype)
    dt_bias = mx.array(
        rng.standard_normal(HV).astype(np.float32)).astype(mx.bfloat16)

    r_out, r_ring, r_state = composed(
        conv_state, state, qkv, w, a, b, A_log, dt_bias,
        scale * scale, scale, 1e-6)
    f_out, f_ring, f_state = fused(
        conv_state, state, qkv, w, a, b, A_log, dt_bias,
        scale * scale, scale, 1e-6)
    mx.eval(r_out, r_ring, r_state, f_out, f_ring, f_state)
    row = {
        "name": name,
        "a_log_dtype": str(a_log_dtype),
        "out_bit_identical": bits(r_out) == bits(f_out),
        "ring_bit_identical": bits(r_ring) == bits(f_ring),
        "state_bit_identical": bits(r_state) == bits(f_state),
    }
    if not all(row[k] for k in
               ("out_bit_identical", "ring_bit_identical", "state_bit_identical")):
        on = np.array(r_out, dtype=np.float32) - np.array(f_out, dtype=np.float32)
        row["out_max_abs_diff"] = float(np.abs(on).max())
    results[name] = row
    print(json.dumps(row))


has_fused = hasattr(mx.fast, "gdn_conv_delta_update")
results["has_gdn_conv_delta_update"] = has_fused
print("has_gdn_conv_delta_update:", has_fused)

if has_fused:
    case("2b-bf16-alog")
    case("2b-bf16-alog-salt2", seed_salt="s2")
    case("f32-alog-bit9", a_log_dtype=mx.float32)
    case("f32-alog-bit9-salt2", a_log_dtype=mx.float32, seed_salt="s2")

ok = has_fused and all(
    results[k][k2]
    for k in results
    if isinstance(results[k], dict) and k != "has_gdn_conv_delta_update"
    for k2 in ("out_bit_identical", "ring_bit_identical", "state_bit_identical")
)
results["PASS"] = bool(ok)
with open(out_path, "w") as f:
    json.dump(results, f, indent=1)
print("GDN-CONV-DELTA-BITCHECK", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
