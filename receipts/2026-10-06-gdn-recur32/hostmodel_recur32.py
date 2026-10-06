#!/usr/bin/env python3
# Host model of shaders/gated_delta_prefill_recur32.comp (GdnRecur32).
#
# Reproduces the kernel's exact fp32 operation order - bf16 activations
# widened to f32, per-lane sequential 4-term partials, a pairwise tree
# over the 32 lane partials (hardware subgroupAdd order is unspecified;
# the tree is the canonical form), y stored bf16 RNE - and checks the
# result against an fp64 reference at the test_gdn_maskless_correctness
# tolerances (y 0.02 abs, state 2e-4 abs).
#
# Kernel order per token t, per (hv, dv) row:
#   st *= g; acc = sum_i st[i]*k[i] (per lane); kv = tree(acc);
#   delta = (v - kv)*beta; st += k*delta;
#   acc = sum_i st[i]*q[i]; o = tree(acc); y = bf16_rne(o)

import numpy as np

DK = 128
LANES = 32
SLICE = 4
TOL_Y = 0.02
TOL_STATE = 2e-4


def values(n, seed, scale):
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        out[i] = np.float32((seed % 20001 / 10000.0 - 1.0) * scale)
    return out


def round_bf16(data):
    bits = data.view(np.uint32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.view(np.float32)


def bf16_rne(x):
    bits = np.float32(x).view(np.uint32)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.view(np.float32)


def tree_sum(partials, axis=-1):
    """Pairwise binary tree over the given axis, like subgroupAdd."""
    p = np.moveaxis(partials, axis, -1).copy()
    while p.shape[-1] > 1:
        if p.shape[-1] % 2:
            raise ValueError("tree needs a power-of-two width")
        p = p[..., 0::2] + p[..., 1::2]
    return p[..., 0]


def kernel_f32(q, k, v, g, beta, h0, T, Hv):
    """Exactly the recur32 kernel in fp32, vectorized over dv rows.
    Shapes: q/k/v [T, Hv, Dk], g/beta [T, Hv], h0/state [Hv, Dv, Dk]."""
    Dv = h0.shape[1]
    state = h0.astype(np.float32).copy()
    y = np.empty((T, Hv, Dv), dtype=np.float32)
    for t in range(T):
        for h in range(Hv):
            kp = k[t, h].reshape(LANES, SLICE)       # [lane, slice]
            qp = q[t, h].reshape(LANES, SLICE)
            st = state[h].reshape(Dv, LANES, SLICE)  # [dv, lane, slice]
            st *= g[t, h]
            acc = np.zeros((Dv, LANES), dtype=np.float32)
            for i in range(SLICE):
                acc += st[:, :, i] * kp[None, :, i]
            kv = tree_sum(acc)                        # [dv]
            delta = (v[t, h] - kv) * beta[t, h]
            for i in range(SLICE):
                st[:, :, i] += kp[None, :, i] * delta[:, None]
            acc = np.zeros((Dv, LANES), dtype=np.float32)
            for i in range(SLICE):
                acc += st[:, :, i] * qp[None, :, i]
            y[t, h] = bf16_rne(tree_sum(acc))
    return y, state


def reference_f64(q, k, v, g, beta, h0, T, Hv):
    """fp64 mirror of the test's reference()."""
    state = h0.astype(np.float64).copy()
    y = np.empty((T, Hv, DK), dtype=np.float64)
    for t in range(T):
        for h in range(Hv):
            gate = np.float64(g[t, h])
            rate = np.float64(beta[t, h])
            kv = (state[h] * gate * k[t, h]).sum(axis=1)
            delta = (v[t, h] - kv) * rate
            updated = state[h] * gate + delta[:, None] * k[t, h]
            y[t, h] = (updated * q[t, h]).sum(axis=1)
            state[h] = updated
    return y, state


def check_case(T, rep, nonzero_state):
    Hv = 16 * rep
    n = T * Hv * DK
    q = round_bf16(values(n, 0x10203040 + T + rep, 0.25))
    k = round_bf16(values(n, 0x50607080 + T + rep, 0.25))
    v = round_bf16(values(n, 0x90A0B0C0 + T + rep, 0.25))
    g = np.abs(values(T * Hv, 0xD0E0F000 + T + rep, 0.07)) + 0.92
    beta = round_bf16(values(T * Hv, 0x12345678 + T + rep, 0.2) + 0.3)
    h0 = (values(Hv * DK * DK, 0xBEEF1234 + T + rep, 0.1)
          if nonzero_state else np.zeros(Hv * DK * DK, np.float32))

    q4, k4, v4 = (a.reshape(T, Hv, DK) for a in (q, k, v))
    g2, b2 = g.reshape(T, Hv), beta.reshape(T, Hv)
    y_ref, s_ref = reference_f64(q4, k4, v4, g2, b2, h0.reshape(Hv, DK, DK), T, Hv)
    y_k, s_k = kernel_f32(q4, k4, v4, g2, b2, h0.reshape(Hv, DK, DK), T, Hv)

    err_y = np.abs(y_k.astype(np.float64) - y_ref).max()
    err_s = np.abs(s_k.astype(np.float64) - s_ref).max()
    ok = err_y <= TOL_Y and err_s <= TOL_STATE
    print(f"T={T:4d} rep={rep} nz={int(nonzero_state)} "
          f"y_err={err_y:.3e} state_err={err_s:.3e} "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ok = True
    # mode 2 (MLX_OMARCHY_GDN_RECUR32=2): every prefill T the contract
    # allows, down to the T >= 2 route minimum (T=1 is the decode kernel).
    for rep in (1, 2, 3):
        for T in (2, 3, 5, 11, 16, 31, 32, 33, 63):
            ok &= check_case(T, rep, nonzero_state=False)
    for rep in (1, 2):
        for T in (11, 63):
            ok &= check_case(T, rep, nonzero_state=True)
    # mode 1 (MLX_OMARCHY_GDN_RECUR32=1): the T >= 64 sweep.
    for rep in (1, 2, 3):
        for T in (63, 64, 65, 96, 352, 512, 519):
            ok &= check_case(T, rep, nonzero_state=False)
    for rep in (1, 2):
        for T in (512, 519):
            ok &= check_case(T, rep, nonzero_state=True)
    print("ALL PASS" if ok else "FAILURES PRESENT")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
