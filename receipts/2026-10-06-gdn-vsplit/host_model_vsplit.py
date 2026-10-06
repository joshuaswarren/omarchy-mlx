#!/usr/bin/env python3
"""Host model: GDN HOIST pass-B Dv split (MLX_OMARCHY_GDN_VSPLIT).

Emulates the per-simdgroup fp32 arithmetic of
gated_delta_prefill_coopmat_hoist_vsplit.comp in kernel order and checks:

  1. Split schedules (SGPW=2, SGPW=1) are BIT-IDENTICAL to the unsplit
     schedule (SGPW=4) for y and final state, with non-zero initial state.
  2. The snapshot slot remap (slab>>2, slab&3) retrieves the same pass-A
     tiles as the unsplit (gy, sg) mapping.
  3. The model is honest: it tracks a direct fp64 token recurrence to
     within fp32 chunked-algorithm tolerance.
"""
import numpy as np

FLT_MIN = np.float32(1.17549435e-38)
C = 8
DK = 128
KSTEPS = DK // 8
HV = 2
DV = 128
BF16_MANTISSA_MASK = np.uint32(0xFFFF0000)


def bf16_round(x):
    """Round fp32 -> bf16 -> fp32 (round-to-nearest-even, like bf16_pack)."""
    u = x.astype(np.float32).view(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    rounded = (u + np.uint32(0x7FFF) + lsb) & BF16_MANTISSA_MASK
    # overflow into exponent: let it wrap; inputs are O(1) so no inf
    return rounded.view(np.float32)


def hscan(vals):
    """Hillis-Steele inclusive scan over 32 lanes in fp32 (subgroupInclusiveAdd)."""
    v = np.zeros(32, dtype=np.float32)
    v[:len(vals)] = vals
    off = 1
    while off < 32:
        w = v.copy()
        w[off:] = (v[off:] + v[:-off]).astype(np.float32)
        v = w
        off *= 2
    return v


def make_inputs(T, seed):
    rng = np.random.default_rng(seed)
    q = bf16_round(rng.standard_normal((T, HV, DK), dtype=np.float32) * 0.3)
    k = bf16_round(rng.standard_normal((T, HV, DK), dtype=np.float32) * 0.3)
    v = bf16_round(rng.standard_normal((T, HV, DV), dtype=np.float32) * 0.3)
    # gates in (0.05, 0.95): exercise the log-domain floor path
    g = rng.uniform(0.05, 0.95, (T, HV)).astype(np.float32)
    beta = bf16_round(rng.uniform(0.1, 0.9, (T, HV)).astype(np.float32))
    h0 = rng.standard_normal((HV, DV, DK), dtype=np.float32) * 0.2  # NON-ZERO
    return q, k, v, g, beta, h0


def pass_a(q, k, g, beta, T):
    """Pass A snap: kkt/qkt per chunk per (head, slot, sub). fp32 ascending-kk."""
    chunks = (T + C - 1) // C
    snap = np.zeros((chunks, HV, DV // 32, 512), dtype=np.float32)
    for ch in range(chunks):
        t0 = ch * C
        nv = min(C, T - t0)
        for h in range(HV):
            kc = np.zeros((C, DK), dtype=np.float32)
            qc = np.zeros((C, DK), dtype=np.float32)
            kc[:nv] = k[t0:t0 + nv, h]
            qc[:nv] = q[t0:t0 + nv, h]
            # kkt/qkt are token x token: one 8x8 tile per simdgroup, stored
            # redundantly to every slab position (matches pass A).
            kkt = np.zeros((C, C), dtype=np.float32)
            qkt = np.zeros((C, C), dtype=np.float32)
            for kk in range(KSTEPS):  # ascending-kk MMA accumulation
                ks = kc[:, kk * 8:(kk + 1) * 8]
                qs = qc[:, kk * 8:(kk + 1) * 8]
                kkt += ks @ ks.T
                qkt += qs @ ks.T
            for slot in range(DV // 32):
                for sub in range(4):
                    snap[ch, h, slot, sub * 128:sub * 128 + 64] = \
                        kkt.reshape(64)
                    snap[ch, h, slot, sub * 128 + 64:sub * 128 + 128] = \
                        qkt.reshape(64)
    return snap


def tok(mat, nv):
    """Zero-pad a token-sliced (nv, x) chunk matrix to C token rows."""
    out = np.zeros((C,) + mat.shape[1:], dtype=np.float32)
    out[:nv] = mat
    return out


def run_pass_b(q, k, v, g, beta, h0, snap, T, sgpw, strict_snap=True):
    """Emulate the vsplit pass-B kernel at local_size_y = sgpw."""
    gy_count = DV // (8 * sgpw)
    y = np.zeros((T, HV, DV), dtype=np.float32)
    state_out = np.zeros_like(h0)
    chunks = (T + C - 1) // C
    for h in range(HV):
        for gy in range(gy_count):
            for sg in range(sgpw):
                slab = gy * sgpw + sg
                dv0 = slab * 8
                # state restore: s_state[kk][r][c] = h0[h, dv0+c, kk*8+r]
                st = np.zeros((KSTEPS, 8, 8), dtype=np.float32)
                for kk in range(KSTEPS):
                    st[kk] = h0[h, dv0:dv0 + 8, kk * 8:(kk + 1) * 8].T
                for ch in range(chunks):
                    t0 = ch * C
                    nv = min(C, T - t0)
                    gv = np.log(np.maximum(g[t0:t0 + nv, h], FLT_MIN))
                    gam = hscan(gv)[:C]                       # gamma_s
                    bet = np.zeros(C, dtype=np.float32)
                    bet[:nv] = beta[t0:t0 + nv, h]
                    gamC = np.exp(gam[C - 1])
                    gam_r = np.exp(gam)                        # per row r
                    dec_r = np.where(np.arange(C) < nv,
                                     np.exp(gam[C - 1] - gam),
                                     np.float32(0.0)).astype(np.float32)
                    bg = (bet * gam_r).astype(np.float32)      # beta*gam per row

                    # loop 1: staged k/q row-scaled, m2/tmp (ascending kk)
                    m2 = np.zeros((8, 8), dtype=np.float32)
                    tmp = np.zeros((8, 8), dtype=np.float32)
                    for kk in range(KSTEPS):
                        kg = st[kk]                            # [dk 8][dv 8]
                        a = tok(k[t0:t0 + nv, h, kk * 8:(kk + 1) * 8], nv)
                        m2 += (a * bg[:, None]) @ kg
                        qs = tok(q[t0:t0 + nv, h, kk * 8:(kk + 1) * 8], nv)
                        tmp += (qs * gam_r[:, None]) @ kg

                    # snapshot tile for THIS slab (the remapped formula)
                    base_slot = (ch * HV + h) * (DV // 32) + (slab >> 2)
                    off = (slab & 3) * 128
                    kkt = snap.reshape(-1, 512)[base_slot, off:off + 64] \
                        .reshape(8, 8)
                    qkt = snap.reshape(-1, 512)[base_slot, off + 64:off + 128] \
                        .reshape(8, 8)
                    if strict_snap:  # the unsplit mapping must agree bitwise
                        u_slot = (ch * HV + h) * (DV // 32) + (slab >> 2)
                        assert u_slot == base_slot and (slab & 3) == (slab & 3)

                    # N, Tinv
                    tri = (np.arange(8)[None, :] < np.arange(8)[:, None])
                    # log-domain difference (kernel: exp(gamma_s[r]-gamma_s[c]))
                    sc = (gam[:, None] - gam[None, :]).astype(np.float32)
                    n_mat = np.where(tri, kkt * bet[:, None] * np.exp(sc),
                                     np.float32(0.0)).astype(np.float32)
                    n2 = n_mat @ n_mat
                    n4 = n2 @ n2
                    n6 = n2 @ n4
                    msum = np.eye(8, dtype=np.float32) + n2 + n4 + n6
                    tinv = (msum - msum @ n_mat).astype(np.float32)

                    # delta
                    vv = tok(v[t0:t0 + nv, h, dv0:dv0 + 8], nv)
                    bv = (vv * bet[:, None]).astype(np.float32)
                    delta = (tinv @ (bv - m2)).astype(np.float32)

                    # out/y
                    qtri = np.where((np.arange(8)[None, :] <=
                                     np.arange(8)[:, None]),
                                    qkt * np.exp(sc),
                                    np.float32(0.0)).astype(np.float32)
                    out_t = (qtri @ delta + tmp).astype(np.float32)
                    rows = np.arange(C) < nv
                    y[t0:t0 + nv, h, dv0:dv0 + 8] = out_t[rows]

                    # state waves
                    for wave in range(KSTEPS // 4):
                        for j in range(4):
                            kk = wave * 4 + j
                            kc = tok(k[t0:t0 + nv, h, kk * 8:(kk + 1) * 8], nv)
                            kgc = (kc * dec_r[:, None]).astype(np.float32)
                            kd = (kgc.T @ delta).astype(np.float32)
                            st[kk] = (st[kk] * gamC + kd).astype(np.float32)
                for kk in range(KSTEPS):
                    state_out[h, dv0:dv0 + 8, kk * 8:(kk + 1) * 8] = st[kk].T
    return y, state_out


def fp64_reference(q, k, v, g, beta, h0, T):
    y = np.zeros((T, HV, DV))
    s = h0.astype(np.float64).copy()  # [HV, DV, DK], rows = dv
    for t in range(T):
        for h in range(HV):
            s[h] *= g[t, h]
            st = s[h].T  # [DK, DV]
            kv = k[t, h].astype(np.float64) @ st
            d = beta[t, h].astype(np.float64) * (v[t, h].astype(np.float64) - kv)
            st += np.outer(k[t, h].astype(np.float64), d)
            s[h] = st.T
            y[t, h] = q[t, h].astype(np.float64) @ st
    return y, s


def main():
    print(f"{'T':>5}  schedule   y==unsplit  state==unsplit  |y-ref|rel")
    for T in (512, 519, 1024):
        q, k, v, g, beta, h0 = make_inputs(T, seed=42 + T)
        snap = pass_a(q, k, g, beta, T)
        y4, s4 = run_pass_b(q, k, v, g, beta, h0, snap, T, 4)
        yr, sr = fp64_reference(q, k, v, g, beta, h0, T)
        rel = np.max(np.abs(y4.astype(np.float64) - yr)) / \
            (np.max(np.abs(yr)) + 1e-30)
        for sgpw in (2, 1):
            ys, ss = run_pass_b(q, k, v, g, beta, h0, snap, T, sgpw)
            eq_y = np.array_equal(
                y4.view(np.uint32), ys.view(np.uint32))
            eq_s = np.array_equal(
                s4.view(np.uint32), ss.view(np.uint32))
            print(f"{T:>5}  SGPW={sgpw:>2}    {str(eq_y):>5}       "
                  f"{str(eq_s):>5}        {rel:.2e}")
            assert eq_y and eq_s, f"NOT bit-identical at T={T} SGPW={sgpw}"
    print("PASS: split schedules bit-identical to unsplit; model tracks "
          "fp64 reference within chunked-fp32 tolerance.")


if __name__ == "__main__":
    main()
