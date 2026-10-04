#!/usr/bin/env python3
"""Bit-exact chase, phase 2: stage attribution on one failing GDU call.

Loads failing-operands.npz (bitexact_bisect.py), then on the GPU:
  1. composed reference = mx.fast.gated_delta_update_raw on Hk<Hv operands
     (use_fallback -> C++ composed chain); must equal the captured bits;
  2. op-by-op Python replica of the C++ fallback, dumping every
     intermediate (beta, x, sp, g, state_next, kv, delta, state, o);
  3. fused = the same call with q/k repeated to Hv (the RAW_REPEAT route).
Then numpy f32 emulations seeded with the GPU intermediates attribute the
mismatch: kv reduction order, the state update, the o reduction, and the
shader's gate formula vs the composed elementwise gates.

usage: gdu_stage_dump.py <failing-operands.npz> <out.json>
"""
import json
import sys

import mlx.core as mx
import numpy as np

d = np.load(sys.argv[1])
out_path = sys.argv[2]


def bf16(name):
    return mx.array(d[name]).view(mx.bfloat16)


q, k, v = bf16("q_bf16bits"), bf16("k_bf16bits"), bf16("v_bf16bits")
a, b, dt = bf16("a_bf16bits"), bf16("b_bf16bits"), bf16("dt_bf16bits")
A_log = mx.array(d["A_log"])
s0 = mx.array(d["state_in_bits"]).view(mx.float32)
ref_out_bits, ref_st_bits = d["out_bits"], d["state_bits"]
Hk, Hv = q.shape[2], v.shape[2]
rep = Hv // Hk
R = {}


def bits16(x):
    return np.array(x.view(mx.uint16))


def bits32(x):
    return np.array(x.view(mx.uint32))


def f32(x):
    return np.array(x.astype(mx.float32))


def frac_diff(x, y):
    return float((x != y).mean())


# 1. composed reference through the primitive (Hk<Hv -> fallback)
o_c, s_c = mx.fast.gated_delta_update_raw(q, k, v, a, b, A_log, dt, s0)
mx.eval(o_c, s_c)
R["composed_matches_capture"] = bool((bits16(o_c) == ref_out_bits).all() and
                                     (bits32(s_c) == ref_st_bits).all())

# 2. op-by-op replica of the C++ fallback (T == 1)
beta = mx.sigmoid(b)
x = mx.add(a, dt)
sp = mx.logaddexp(mx.array(0.0, mx.float32), x)
g = mx.exp(mx.negative(mx.multiply(mx.exp(A_log.astype(mx.float32)), sp)))
qr = mx.repeat(q, rep, 2)
kr = mx.repeat(k, rep, 2)
q_t, k_t, vv = qr[:, 0], kr[:, 0], v[:, 0]
v_t = vv
g_t, beta_t = g[:, 0], beta[:, 0]
decay = mx.expand_dims(g_t, (-1, -2))
st_next = mx.multiply(s0, decay)
kv_terms = mx.multiply(st_next, mx.expand_dims(k_t, -2))
kv = mx.sum(kv_terms, -1)
delta = mx.multiply(mx.subtract(v_t, kv), mx.expand_dims(beta_t, -1))
upd = mx.multiply(mx.expand_dims(delta, -1), mx.expand_dims(k_t, -2))
st = mx.add(st_next, upd)
o_terms = mx.multiply(st, mx.expand_dims(q_t, -2))
o = mx.sum(o_terms, -1).astype(q.dtype)
mx.eval(beta, x, sp, g, st_next, kv_terms, kv, delta, upd, st, o_terms, o)
R["dtypes"] = {n: str(t.dtype) for n, t in
               dict(beta=beta, x=x, sp=sp, g=g, st_next=st_next, kv=kv,
                    delta=delta, st=st, o=o).items()}
R["replica_matches_composed"] = {
    "out": bool((bits16(o)[:, None] == bits16(o_c)).all()),
    "state": bool((bits32(st) == bits32(s_c)).all()),
}

# 3. fused route (q/k repeated to Hv -> fused tiled kernel)
o_f, s_f = mx.fast.gated_delta_update_raw(qr, kr, v, a, b, A_log, dt, s0)
mx.eval(o_f, s_f)
sf_b, sc_b = bits32(s_f), bits32(s_c)
R["fused_vs_composed"] = {
    "out_frac_diff": frac_diff(bits16(o_f), bits16(o_c)),
    "state_frac_diff": frac_diff(sf_b, sc_b),
    "state_heads_with_diff": int((sf_b != sc_b).reshape(Hv, -1).any(1).sum()),
    "state_rows_with_diff": int((sf_b != sc_b).reshape(Hv * 128, 128).any(1).sum()),
}
rows_diff = (sf_b != sc_b).reshape(Hv * 128, 128)
R["fused_vs_composed"]["mean_cols_diff_in_bad_rows"] = (
    float(rows_diff[rows_diff.any(1)].mean()) if rows_diff.any() else 0.0)

# 4. numpy emulation seeded with GPU intermediates (all f32, IEEE RN)
S0 = np.array(s0)[0]                       # [Hv, Dv, Dk]
G = f32(g)[0, 0]                           # [Hv]
B = f32(beta)[0, 0]                        # [Hv] (bf16-exact)
K = f32(k_t)[0]                            # [Hv, Dk]
Q = f32(q_t)[0]
V = f32(vv)[0]                             # [Hv, Dv]
SN_gpu = np.array(st_next)[0]
KV_gpu = np.array(kv)[0]
ST_gpu = np.array(st)[0]

SN = (S0 * G[:, None, None]).astype(np.float32)
R["emu_state_next_matches"] = bool((SN.view(np.uint32) == SN_gpu.view(np.uint32)).all())
terms = (SN * K[:, None, :]).astype(np.float32)
R["emu_kv_terms_match"] = bool((terms.view(np.uint32) ==
                                np.array(kv_terms)[0].view(np.uint32)).all())


def serial(t):
    acc = np.zeros(t.shape[:-1], np.float32)
    for i in range(t.shape[-1]):
        acc = (acc + t[..., i]).astype(np.float32)
    return acc


def pairwise(t):
    while t.shape[-1] > 1:
        t = (t[..., 0::2] + t[..., 1::2]).astype(np.float32)
    return t[..., 0]


def strided(t, lanes):
    # each of `lanes` lanes sums i = lane, lane+lanes, ... serially, then a
    # pairwise (butterfly-equivalent) tree over lanes
    part = np.stack([serial(t[..., l::lanes]) for l in range(lanes)], -1)
    return pairwise(part)


def chunked(t, chunk):
    # each lane sums a contiguous chunk serially, then pairwise over lanes
    n = t.shape[-1] // chunk
    part = np.stack([serial(t[..., c * chunk:(c + 1) * chunk]) for c in range(n)], -1)
    return pairwise(part)


def chunked_serial(t, chunk):
    n = t.shape[-1] // chunk
    part = np.stack([serial(t[..., c * chunk:(c + 1) * chunk]) for c in range(n)], -1)
    return serial(part)


cands = {"serial": serial, "pairwise": pairwise}
for lanes in (2, 4, 8, 16, 32, 64):
    cands[f"strided{lanes}"] = (lambda t, L=lanes: strided(t, L))
for chunk in (2, 4, 8, 16, 32, 64):
    cands[f"chunk{chunk}_tree"] = (lambda t, C=chunk: chunked(t, C))
    cands[f"chunk{chunk}_serial"] = (lambda t, C=chunk: chunked_serial(t, C))
R["kv_reduction_match"] = {
    n: float((f(terms).view(np.uint32) == KV_gpu.view(np.uint32)).mean())
    for n, f in cands.items()}

# shader-order emulation (serial kv, per-element products) with GPU g/beta
kv_ser = serial(terms)
delta_e = ((V - kv_ser).astype(np.float32) * B[:, None]).astype(np.float32)
ST_e = (SN + (delta_e[:, :, None] * K[:, None, :]).astype(np.float32)).astype(np.float32)
R["shader_emu_state_vs_fused_frac_diff"] = frac_diff(ST_e.view(np.uint32), sf_b[0])
R["shader_emu_state_vs_composed_frac_diff"] = frac_diff(ST_e.view(np.uint32), sc_b[0])
# composed-order emulation with GPU kv
delta_c = ((V - KV_gpu).astype(np.float32) * B[:, None]).astype(np.float32)
R["emu_delta_matches_gpu"] = bool((delta_c.view(np.uint32) ==
                                   np.array(delta)[0].view(np.uint32)).all())
ST_c = (SN + (delta_c[:, :, None] * K[:, None, :]).astype(np.float32)).astype(np.float32)
R["emu_state_from_gpu_kv_matches_composed"] = bool(
    (ST_c.view(np.uint32) == ST_gpu.view(np.uint32)).all())
o_terms_e = (ST_gpu * Q[:, None, :]).astype(np.float32)
R["o_reduction_match"] = {
    n: float((f(o_terms_e).view(np.uint32) ==
              np.array(mx.sum(o_terms, -1))[0].view(np.uint32)).mean())
    for n, f in cands.items()}

# per-head gate search: which (g, beta) reproduces the FUSED state head?
np.save(out_path.replace(".json", "-fused-state.npy"), np.array(s_f.view(mx.uint32)))
np.save(out_path.replace(".json", "-fused-out.npy"), np.array(o_f.view(mx.uint16)))
np.savez(out_path.replace(".json", "-gpu-gates.npz"), g=G, beta=B, x=f32(x)[0, 0], sp=f32(sp)[0, 0])
SF = np.array(s_f)[0]
bad_heads = [int(h) for h in np.nonzero((sf_b[0] != sc_b[0]).reshape(Hv, -1).any(1))[0]]


def head_state(h, gv, bv):
    sn = (S0[h] * np.float32(gv)).astype(np.float32)
    kvh = serial((sn * K[h][None, :]).astype(np.float32))
    dl = ((V[h] - kvh).astype(np.float32) * np.float32(bv)).astype(np.float32)
    return (sn + (dl[:, None] * K[h][None, :]).astype(np.float32)).astype(np.float32)


def bf16_step(x, n):
    u = int(np.array([x], np.float32).view(np.uint32)[0]) >> 16
    return float(np.array([(u + n) << 16], np.uint32).view(np.float32)[0])


search = {}
for h in bad_heads:
    hit = []
    for dg in range(-6, 7):
        gv = np.nextafter(np.float32(G[h]), np.float32(np.inf if dg > 0 else -np.inf))
        gv = np.float32(G[h])
        for _ in range(abs(dg)):
            gv = np.nextafter(gv, np.float32(np.inf if dg > 0 else -np.inf))
        for db in range(-2, 3):
            bv = bf16_step(float(B[h]), db)
            if (head_state(h, gv, bv).view(np.uint32) == SF[h].view(np.uint32)).all():
                hit.append({"g_ulp": dg, "beta_bf16_ulp": db})
    search[h] = {"hits": hit, "g": float(G[h]), "beta": float(B[h]),
                 "x": float(f32(x)[0, 0, h]), "sp": float(f32(sp)[0, 0, h]),
                 "b": float(f32(b)[0, 0, h]), "a": float(f32(a)[0, 0, h]),
                 "dt": float(f32(dt)[h]), "A_log": float(np.array(A_log)[h])}
R["bad_head_gate_search"] = search

# gate attribution: does the fused state equal the shader emulation seeded
# with the COMPOSED g (then the gates agree) or not?
R["gates"] = {"g": G.tolist()[:4], "beta": B.tolist()[:4]}
print(json.dumps(R, indent=1))
json.dump(R, open(out_path, "w"), indent=1)
