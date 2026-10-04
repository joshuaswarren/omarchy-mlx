#!/usr/bin/env python3
"""Fuse2B bitcheck DIAG: which output differs, where, by how much.
Runs the same composed-vs-fused case as gdn_conv_delta_bitcheck.py but
reports per-output mismatch counts, max abs diff, and first-mismatch
indices (mx-native comparisons, no numpy bf16).
"""
import sys

import mlx.core as mx
import numpy as np

B, HV, HK, DK, DV = 1, 16, 16, 128, 128
KEY_DIM, VALUE_DIM = HK * DK, HV * DV
C, K = 2 * KEY_DIM + VALUE_DIM, 4


def bits(a):
    return np.array(a.view(mx.uint16), dtype=np.uint16) if a.dtype in (mx.bfloat16, mx.float16) else np.array(a.view(mx.uint32), dtype=np.uint32)


def case(tag, a_log_dtype=mx.bfloat16, salt=""):
    rng = np.random.default_rng(abs(hash((tag, salt))) % (2**31))
    scale = float(DK ** -0.5)
    conv_state = mx.array(rng.standard_normal((B, K - 1, C)).astype(np.float32)).astype(mx.bfloat16)
    state = mx.array(rng.standard_normal((B, HV, DV, DK)).astype(np.float32))
    qkv = mx.array(rng.standard_normal((B, 1, C)).astype(np.float32)).astype(mx.bfloat16)
    w = mx.array(rng.standard_normal((C, K, 1)).astype(np.float32)).astype(mx.bfloat16)
    a = mx.array(rng.standard_normal((B, 1, HV)).astype(np.float32)).astype(mx.bfloat16)
    b = mx.array(rng.standard_normal((B, 1, HV)).astype(np.float32)).astype(mx.bfloat16)
    A_log = mx.array((-rng.random(HV)).astype(np.float32)).astype(a_log_dtype)
    dt_bias = mx.array(rng.standard_normal(HV).astype(np.float32)).astype(mx.bfloat16)

    r_conv, r_ring = mx.fast.gdn_conv_update(
        conv_state, qkv, w, activate=True, qk_key_dim=KEY_DIM,
        qk_scale_q=scale * scale, qk_scale_k=scale, qk_eps=1e-6)
    r_q = r_conv[..., :KEY_DIM].reshape(B, 1, HK, DK)
    r_k = r_conv[..., KEY_DIM:2 * KEY_DIM].reshape(B, 1, HK, DK)
    r_v = r_conv[..., 2 * KEY_DIM:].reshape(B, 1, HV, DV)
    r_out, r_state = mx.fast.gated_delta_update_raw(
        r_q, r_k, r_v, a, b, A_log, dt_bias, state, None)

    f_out, f_ring, f_state = mx.fast.gdn_conv_delta_update(
        conv_state, state, qkv, w, a, b, A_log, dt_bias,
        activate=True, qk_key_dim=KEY_DIM, qk_scale_q=scale * scale,
        qk_scale_k=scale, qk_eps=1e-6)
    mx.eval(r_conv, r_ring, r_out, r_state, f_out, f_ring, f_state)

    report = {"tag": tag, "a_log": str(a_log_dtype)}
    for name, r, f, wide in (
        ("out", r_out, f_out, False),
        ("ring", r_ring, f_ring, False),
        ("state", r_state, f_state, True),
    ):
        rb, fb = bits(r), bits(f)
        same = rb.tobytes() == fb.tobytes()
        rd = np.array(r.astype(mx.float32), dtype=np.float64)
        fd = np.array(f.astype(mx.float32), dtype=np.float64)
        d = np.abs(rd - fd)
        idx = np.unravel_index(np.argmax(d), d.shape) if d.size else ()
        report[name] = {
            "bit_identical": bool(same),
            "max_abs": float(d.max()),
            "mismatches": int((rb != fb).sum()),
            "first_bad_index": [int(i) for i in idx],
            "ref_at_idx": float(rd[idx]) if d.size else None,
            "fus_at_idx": float(fd[idx]) if d.size else None,
        }
    print(tag, a_log_dtype, {k: report[k] for k in ("out", "ring", "state")}, flush=True)
    return report


results = [case("bf16-alog"), case("bf16-alog-s2", salt="s2")]
ok = all(all(results[i][k]["bit_identical"] for k in ("out", "ring", "state")) for i in range(2))
print("DIAG", "PASS" if ok else "MISMATCH")
