"""Per-kernel recheck configs: realistic shapes/dtypes from the upstream call
sites, deterministic input builders, and NumPy fp64 (or integer-exact)
references per docs/numerics-gate.md.

No mlx import here: the dev-box dry-run (--reference-only) uses numpy only.
Input builders pre-round float inputs onto the exact bf16/fp16 grid the GPU
run will see, so references and GPU outputs consume identical operands.
"""
import numpy as np

UPSTREAM = {
    "bitlinear_matmul": "bitlinear_matmul.msl",
    "fused_double_norm_rope": "fused_double_norm_rope.msl",
    "fused_single_norm_rope": "fused_single_norm_rope.msl",
    "inkling_banded_mask": "inkling_banded_mask.msl",
    "inkling_banded_mask_v2": "inkling_banded_mask_v2.msl",
    "inkling_sconv_decode": "inkling_sconv_decode.msl",
    "inkling_moe_route": "inkling_moe_route.msl",
    "inkling_moe_down_combine": "inkling_moe_down_combine.msl",
    "mlx_vlm_llguidance_mask": "mlx_vlm_llguidance_mask.msl",
    "custom_depthwise_conv1d": "custom_depthwise_conv1d.msl",
    "qk_relu_squared": "qk_relu_squared.msl",
    "mlx_audio_phonon_unpack_base5_v1": "mlx_audio_phonon_unpack_base5_v1.msl",
    "cbq_gather_mm": "cbq_gather_mm.msl",
    "cbq_gather_mm_v2": "cbq_gather_mm_v2.msl",
    "cbq_gather_mm_v3": "cbq_gather_mm_v3.msl",
    "cbq_gather_mm_v4": "cbq_gather_mm_v4.msl",
    "cbq_gather_mm_v3_situ": "cbq_gather_mm_v3_situ.msl",
    "cbq_gather_mm_v4_situ": "cbq_gather_mm_v4_situ.msl",
    "cbq_gather_mm_glu": "cbq_gather_mm_glu.msl",
    "cbq_grad_d": "cbq_grad_d.msl",
    "cbq_grad_x": "cbq_grad_x.msl",
    "kda_glue_pre": "kda_glue_pre.msl",
    "kda_glue_post": "kda_glue_post.msl",
    "moe_route_fused": "moe_route_fused.msl",
    "situ_fused": "situ_fused.msl",
    "situ_pair_fused": "situ_pair_fused.msl",
}

# Static prediction from reading custom_kernel.cpp at 09da0917a, recorded in
# the notebook entry BEFORE any run. checked against the phase-2 result.
PREDICT = {
    "bitlinear_matmul": "pass",
    "fused_double_norm_rope": "pass",
    "fused_single_norm_rope": "pass",
    "inkling_banded_mask": "pass",
    "inkling_banded_mask_v2": "pass",
    "inkling_sconv_decode": "pass",
    "inkling_moe_route": "pass",
    "inkling_moe_down_combine": "pass",
    "mlx_vlm_llguidance_mask": "crash",   # numeric_limits<T>::infinity()
    "custom_depthwise_conv1d": "pass",
    "qk_relu_squared": "pass",
    "mlx_audio_phonon_unpack_base5_v1": "pass",
    "cbq_gather_mm": "refused",           # int64* deref guard
    "cbq_gather_mm_v2": "refused",        # threadgroup pointer alias guard
    "cbq_gather_mm_v3": "refused",        # as_type<char4>
    "cbq_gather_mm_v4": "refused",
    "cbq_gather_mm_v3_situ": "refused",
    "cbq_gather_mm_v4_situ": "refused",
    "cbq_gather_mm_glu": "refused",
    "cbq_grad_d": "refused",              # pointer guard / atomic_float buffer
    "cbq_grad_x": "refused",
    "kda_glue_pre": "crash",              # header `inline` + `constant T*`
    "kda_glue_post": "crash",
    "moe_route_fused": "crash",           # `constant int* mp`
    "situ_fused": "crash",
    "situ_pair_fused": "crash",
}


def bf16(x):
    """Round float32 values to the bf16 grid (round-to-nearest-even), the
    same arithmetic as _mlx_float_to_bf16 / Metal's bfloat16_t stores.
    Keeps the input shape: scalars stay 0-d (numpy 2.5 refuses float() on
    size-1 arrays but accepts 0-d)."""
    arr = np.asarray(x, dtype=np.float32)
    bits = arr.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) \
        & np.uint32(0xFFFF0000)
    return rounded.view(np.float32).reshape(arr.shape)


def f16(x):
    return np.asarray(x, dtype=np.float16)


def rng(seed):
    return np.random.default_rng(seed)


REL = ("rel", 2.0 ** -7)      # 2 bf16 ULP
REL16 = ("rel", 2.0 ** -9)    # 2 fp16 ULP
EXACT = ("exact",)


_GRID_CACHE = None


def _grid_lut():
    """The upstream repo reads iq1s_grid.npz (llama.cpp IQ1_S grid) from its own
    directory but does not ship it. The kernel treats `grid` as an opaque
    int8 LUT, so any deterministic 2048x8 int8 table exercises the identical
    code path; the reference reads the same table. Seed fixed for
    reproducibility across processes."""
    global _GRID_CACHE
    if _GRID_CACHE is None:
        _GRID_CACHE = np.random.default_rng(20261007).integers(
            -128, 128, (2048, 8), dtype=np.int8)
    return _GRID_CACHE


# ---------------------------------------------------------------- bitlinear
def build_bitlinear(seed=1):
    r = rng(seed)
    B, K, O = 4, 2048, 2048
    x = f16(r.standard_normal((B, K)) * 0.5)
    pw = r.integers(0, 4, (O // 4, K)).astype(np.uint8) * np.uint8(0)
    # ternary codes in the low 2 bits of each 2-bit lane: values 0..3 (code-1)
    pw = r.integers(0, 4, (O // 4, K)).astype(np.uint8)
    for q in range(1, 4):
        pw |= r.integers(0, 4, (O // 4, K)).astype(np.uint8) << np.uint8(2 * q)
    ws = f16([0.05])
    return {"x": x, "packed_weights": pw, "weight_scale": ws}


def ref_bitlinear(inp):
    x = inp["x"].astype(np.float64)
    pw = inp["packed_weights"]
    ws = float(inp["weight_scale"].astype(np.float32)[0])
    B, K = x.shape
    O4 = pw.shape[0]
    w = np.stack([((pw >> np.uint8(2 * q)) & np.uint8(3)).astype(np.int64) - 1
                  for q in range(4)])                       # (4,O4,K)
    acc = np.einsum("bi,qri->bqr", x, w.astype(np.float64))
    out = np.concatenate([acc[:, q, :] for q in range(4)], axis=1)  # cols r+q*O4
    return [out * ws]


# ---------------------------------------------------------------- flux2
FLUX = dict(dim=3072, heads=24, hd=128, eps=1e-6, img=64, txt=32)


def build_flux_double(seed=2):
    r = rng(seed)
    d = FLUX
    img = bf16(r.standard_normal((1, d["img"] * 3 * d["dim"])) * 0.4)
    txt = bf16(r.standard_normal((1, d["txt"] * 3 * d["dim"])) * 0.4)
    nq, nk = bf16(r.standard_normal(d["hd"])), bf16(r.standard_normal(d["hd"]))
    naq, nak = bf16(r.standard_normal(d["hd"])), bf16(r.standard_normal(d["hd"]))
    s_tot = d["img"] + d["txt"]
    cos = bf16(r.standard_normal((s_tot, d["hd"] // 2)))
    sin = bf16(r.standard_normal((s_tot, d["hd"] // 2)))
    return {"img_qkv": img, "txt_qkv": txt, "norm_q": nq, "norm_k": nk,
            "norm_added_q": naq, "norm_added_k": nak, "cos_vals": cos, "sin_vals": sin}


def _flux_kernel_emulation(rows_flat, nq, nk, cos_flat, sin_flat, meta, eps):
    """Literal per-thread emulation of the fused norm+rope kernel body.
    rows_flat is the concatenated qkv plane laid out exactly as the buffers
    the kernel indexes (txt rows first for the double kernel); nq/nk are
    (S, hd) selected exactly as the kernel's is_txt branch does."""
    dim, hd, heads, s_tot = meta
    hd_half = hd // 2
    ELEMS = hd // 32
    outq = np.zeros((heads, s_tot, hd))
    outk = np.zeros_like(outq)
    outv = np.zeros_like(outq)
    for s in range(s_tot):
        for h in range(heads):
            q_col, k_col, v_col = h * hd, dim + h * hd, 2 * dim + h * hd
            lq = np.zeros((32, ELEMS))
            lk = np.zeros((32, ELEMS))
            lv = np.zeros((32, ELEMS))
            lane_sq_q = np.zeros(32)
            lane_sq_k = np.zeros(32)
            for tid in range(32):
                for i in range(ELEMS):
                    d = tid * ELEMS + i
                    base = s * (3 * dim)
                    lq[tid, i] = rows_flat[base + q_col + d]
                    lk[tid, i] = rows_flat[base + k_col + d]
                    lv[tid, i] = rows_flat[base + v_col + d]
                    lane_sq_q[tid] += lq[tid, i] * lq[tid, i]
                    lane_sq_k[tid] += lk[tid, i] * lk[tid, i]
            inv_q = 1.0 / np.sqrt(lane_sq_q.sum() / hd + eps)
            inv_k = 1.0 / np.sqrt(lane_sq_k.sum() / hd + eps)
            for tid in range(32):
                for i in range(ELEMS):
                    d = tid * ELEMS + i
                    lq[tid, i] = lq[tid, i] * inv_q * nq[s, d]
                    lk[tid, i] = lk[tid, i] * inv_k * nk[s, d]
            cos_base = s * hd_half
            for tid in range(32):
                for p in range(0, ELEMS, 2):
                    d = tid * ELEMS + p
                    d_half = d >> 1
                    c = cos_flat[cos_base + d_half]
                    sn = sin_flat[cos_base + d_half]
                    for arr in (lq, lk):
                        a0, a1 = arr[tid, p], arr[tid, p + 1]
                        arr[tid, p] = a0 * c - a1 * sn
                        arr[tid, p + 1] = a1 * c + a0 * sn
            for tid in range(32):
                for i in range(ELEMS):
                    d = tid * ELEMS + i
                    outq[h, s, d] = lq[tid, i]
                    outk[h, s, d] = lk[tid, i]
                    outv[h, s, d] = lv[tid, i]
    return [o[None] for o in (outq, outk, outv)]


def ref_flux_double(inp):
    d = FLUX
    dim, hd, heads, eps = d["dim"], d["hd"], d["heads"], d["eps"]
    s_tot = d["img"] + d["txt"]
    txt = inp["txt_qkv"][0].astype(np.float64).reshape(d["txt"], 3 * dim)
    img = inp["img_qkv"][0].astype(np.float64).reshape(d["img"], 3 * dim)
    rows = np.concatenate([txt, img], axis=0).reshape(-1)
    nq = np.concatenate([np.repeat(inp["norm_added_q"][None], d["txt"], axis=0),
                         np.repeat(inp["norm_q"][None], d["img"], axis=0)], axis=0)
    nk = np.concatenate([np.repeat(inp["norm_added_k"][None], d["txt"], axis=0),
                         np.repeat(inp["norm_k"][None], d["img"], axis=0)], axis=0)
    return _flux_kernel_emulation(
        rows, nq, nk,
        inp["cos_vals"].reshape(-1).astype(np.float64),
        inp["sin_vals"].reshape(-1).astype(np.float64),
        (dim, hd, heads, s_tot), eps)


def build_flux_single(seed=3):
    r = rng(seed)
    d = FLUX
    s_tot = d["img"] + d["txt"]
    fused_dim = 3 * d["dim"] + 2 * int(d["dim"] * 3.0)
    fused = bf16(r.standard_normal((1, s_tot, fused_dim)) * 0.3)
    nq, nk = bf16(r.standard_normal(d["hd"])), bf16(r.standard_normal(d["hd"]))
    cos = bf16(r.standard_normal((s_tot, d["hd"] // 2)))
    sin = bf16(r.standard_normal((s_tot, d["hd"] // 2)))
    return {"fused": fused, "norm_q": nq, "norm_k": nk, "cos_vals": cos, "sin_vals": sin}


def ref_flux_single(inp):
    d = FLUX
    dim, hd, heads, eps = d["dim"], d["hd"], d["heads"], d["eps"]
    s_tot = d["img"] + d["txt"]
    rows = inp["fused"][0].astype(np.float64).reshape(-1)
    nq = np.repeat(inp["norm_q"][None], s_tot, axis=0)
    nk = np.repeat(inp["norm_k"][None], s_tot, axis=0)
    return _flux_kernel_emulation(
        rows, nq, nk,
        inp["cos_vals"].reshape(-1).astype(np.float64),
        inp["sin_vals"].reshape(-1).astype(np.float64),
        (dim, hd, heads, s_tot), eps)


# ---------------------------------------------------------------- inkling
def build_mask(seed=4):
    r = rng(seed)
    B, LQ, H, D_REL, S = 2, 16, 4, 8, 32
    rel = bf16(r.standard_normal((B, LQ, H, D_REL)))
    proj = bf16(r.standard_normal((D_REL, 16)))
    kshape = np.zeros((B, S, 4), dtype=np.float32)   # shape carrier, unread
    return {"rel": rel, "proj": proj, "kshape": kshape}


def _mask_ref(inp, q_offset, sliding, rel_extent, S):
    rel = inp["rel"].astype(np.float64)
    proj = inp["proj"].astype(np.float64)
    B, LQ, H, d_rel = rel.shape
    dist = np.arange(LQ)[:, None] + q_offset - np.arange(S)[None, :]
    plane = np.where(dist < 0, -1e30, 0.0)
    if sliding > 0:
        plane = np.where(dist >= sliding, -1e30, plane)
    val = np.broadcast_to(plane, (B, H, LQ, S)).copy()
    band = (dist >= 0) & (dist < rel_extent)
    acc = np.einsum("bihd,de->bhie", rel, proj)     # acc[b,h,i,e] = sum_d rel*proj[d,e]
    for b in range(B):
        for h in range(H):
            for i in range(LQ):
                for j in range(S):
                    if band[i, j]:
                        val[b, h, i, j] = acc[b, h, i, dist[i, j]]
    return [val]


def ref_mask(inp):
    return _mask_ref(inp, q_offset=16, sliding=32, rel_extent=16, S=32)


def ref_mask_v2(inp):
    S = inp["kshape"].shape[1]
    return _mask_ref(inp, q_offset=S - inp["rel"].shape[1], sliding=32,
                     rel_extent=16, S=S)


def build_sconv(seed=5):
    r = rng(seed)
    B, L, C, K = 2, 4, 512, 4
    x = bf16(r.standard_normal((B, L, C)) * 0.5)
    state = r.standard_normal((B, K - 1, C)).astype(np.float32)
    w = r.standard_normal((C * K,)).astype(np.float32)
    res = bf16(r.standard_normal((B, L, C)) * 0.5)
    return {"x": x, "state": state, "w": w, "res": res}


def ref_sconv(inp):
    x = inp["x"].astype(np.float64)
    state = inp["state"].astype(np.float64)
    w = inp["w"].astype(np.float64).reshape(-1, 4)
    res = inp["res"].astype(np.float64)
    B, L, C = x.shape
    K = 4
    xp = np.concatenate([state, x], axis=1)          # (B, K-1+L, C)
    out = np.zeros((B, L, C))
    nstate = np.zeros((B, K - 1, C))
    for i in range(L):
        acc = np.zeros((B, C))
        for k in range(K):
            acc += w[:, k] * xp[:, i + k, :]
        conv_r = bf16(acc)
        inner = bf16(conv_r + x[:, i, :])
        out[:, i, :] = bf16(inner + res[:, i, :])
    nstate = xp[:, L:, :]
    return [out, nstate]


def build_route(seed=6):
    r = rng(seed)
    N, R, SH, K, I = 2, 64, 2, 8, 256
    logits = bf16(np.round(r.standard_normal((N, R + SH)) * 4.0) * 0.05)  # spaced
    corr = bf16(r.standard_normal(R) * 0.01)
    wscale = np.array([0.001], dtype=np.float32)
    return {"logits": logits, "corr": corr, "wscale": wscale}


def ref_route(inp):
    logits = inp["logits"].astype(np.float64)
    corr = inp["corr"].astype(np.float64)
    ws = float(inp["wscale"][0])
    N, RSH = logits.shape
    R = corr.shape[0]
    SH = RSH - R
    K, I = 8, 256
    PER = (R + 31) // 32
    idx = np.zeros((N, K), dtype=np.uint32)
    wk = np.zeros((N, K))
    gamma = np.zeros((N, SH * I))
    for n in range(N):
        sc = np.full(32 * PER, -np.inf)
        tl = np.zeros(32 * PER)
        for lane in range(32):
            for t in range(PER):
                j = lane + 32 * t
                if j < R:
                    l = logits[n, j]
                    sc[j] = 1.0 / (1.0 + np.exp(-l)) + corr[j]
                    tl[j] = l
        taken = np.zeros(32 * PER, bool)
        bidx, btl = [], []
        for _kk in range(K):
            best_j, best = -1, -np.inf
            for j in range(32 * PER):     # lowest lane then t => lowest j wins ties
                if not taken[j] and sc[j] > best:
                    best, best_j = sc[j], j
            taken[best_j] = True
            bidx.append(best_j)
            btl.append(tl[best_j])
        vals = [btl[k] for k in range(K)] + [logits[n, R + s] for s in range(SH)]
        lp = []
        for tv in vals:
            a = -tv
            lad = max(a, 0.0) + np.log1p(np.exp(-abs(a)))
            lp.append(-lad)
        m = max(lp)
        lse = m + np.log(sum(np.exp(v - m) for v in lp))
        for lane in range(K):
            idx[n, lane] = bidx[lane]
            wk[n, lane] = np.exp(lp[lane] - lse) * ws
        for s_i in range(SH):
            gamma[n, s_i * I:(s_i + 1) * I] = np.exp(lp[K + s_i] - lse) * ws
    return [idx, wk, gamma]


def build_down_combine(seed=7):
    r = rng(seed)
    E, OUT, IN, GROUPS, K, N = 8, 512, 2048, 32, 8, 1
    act = bf16(r.standard_normal((N * K, IN)) * 0.5)
    wq = r.integers(0, 2 ** 32, (E, OUT, IN // 8), dtype=np.uint32)
    sc = bf16(r.standard_normal((E, OUT, GROUPS)) * 0.1)
    bi = bf16(r.standard_normal((E, OUT, GROUPS)) * 0.1)
    idx = r.integers(0, E, (N, K)).astype(np.uint32)
    wk = bf16(np.abs(r.standard_normal((N, K)) * 0.1) + 0.05)
    return {"xin": act, "wq": wq, "sc": sc, "bi": bi, "idx": idx, "wk": wk}


def ref_down_combine(inp):
    act = inp["xin"].astype(np.float64)          # (N*K, IN)
    wq = inp["wq"]                               # (E, OUT, IN/8) uint32
    sc = inp["sc"].astype(np.float64)
    bi = inp["bi"].astype(np.float64)
    idx = inp["idx"]
    wk = inp["wk"].astype(np.float64)
    N, K = idx.shape
    E, OUT, IN8 = wq.shape
    IN = IN8 * 8
    shifts = np.uint32(4 * np.arange(8))
    out = np.zeros((N, OUT))
    for n in range(N):
        partial = np.zeros(8)
        for sg in range(K):
            e = int(idx[n, sg])
            xr = act[n * K + sg]                                  # (IN,)
            actx_l = xr.reshape(32, 64).sum(axis=1)               # (32,)
            for row in range(OUT):
                w8 = wq[e, row]                                   # (IN/8,)
                nib = np.stack([(w8 >> s) & np.uint32(0xF)
                                for s in shifts], axis=-1).reshape(IN)  # (IN,)
                accq_l = (nib.reshape(32, 64) * xr.reshape(32, 64)).sum(axis=1)
                dot = float((sc[e, row, :] * accq_l + bi[e, row, :] * actx_l).sum())
                dv = float(bf16(np.float32(dot)))
                partial[sg] = float(bf16(np.float32(dv * float(wk[n, sg]))))
        out[n] = bf16(np.float32(partial.sum()))
    return [out]


def build_llg(seed=8):
    r = rng(seed)
    B, V = 2, 4096
    logits = bf16(r.standard_normal((B, V)))
    mask = r.integers(0, 2 ** 32, (B, (V + 31) // 32), dtype=np.uint32).view(np.int32)
    return {"logits": logits, "mask": mask}


def ref_llg(inp):
    logits = inp["logits"].astype(np.float64)
    mask = inp["mask"].view(np.uint32)
    B, V = logits.shape
    out = np.full((B, V), -np.inf)
    for b in range(B):
        for token in range(V):
            word, bit = token >> 5, token & 31
            if word < mask.shape[1] and (int(mask[b, word]) >> bit) & 1:
                out[b, token] = logits[b, token]
    return [out]


def build_depthwise(seed=9):
    r = rng(seed)
    B, L, C, K = 1, 64, 32, 3
    x = r.standard_normal((B, L, C)).astype(np.float32)
    w = r.standard_normal((C, K, 1)).astype(np.float32)
    params = np.array([B, L, C, K, 1, L], dtype=np.int32)
    return {"inp": x, "weight": w, "params": params}


def ref_depthwise(inp):
    x = inp["inp"].astype(np.float64)
    w = inp["weight"].astype(np.float64)[:, :, 0]
    B, L, C = x.shape
    K = w.shape[1]
    pad = 1
    out = np.zeros((B, L, C))
    for t in range(L):
        for k in range(K):
            it = t + k - pad
            if 0 <= it < L:
                out[:, t, :] += x[:, it, :] * w[:, k]
    return [out]


def build_qk(seed=10):
    r = rng(seed)
    B, G, Q, D = 2, 4, 16, 64
    q = f16(r.standard_normal((B, G, Q, D)) * 0.5)
    k = f16(r.standard_normal((B, G, Q, D)) * 0.5)
    scale = f16([1.0 / Q])
    return {"q": q, "k": k, "scale": scale}


def ref_qk(inp):
    """Metal executes `half` arithmetic in fp32 on Apple GPUs (half storage,
    fp32 compute), so the T accumulator is fp32-wide: emulate with an fp64
    dot (well inside fp32 noise) and round to fp16 at the T boundaries."""
    q = inp["q"].astype(np.float64)
    k = inp["k"].astype(np.float64)
    scale = float(inp["scale"].astype(np.float32)[0])
    dot = np.einsum("bgqd,bgkd->bgqk", q, k)
    scaled = np.float16(dot * np.float16(scale))
    pos = np.maximum(scaled.astype(np.float64), 0.0)
    return [np.float16(pos * pos)]


def build_phonon(seed=11):
    r = rng(seed)
    rows, in_features = 512, 2048
    words = in_features // 16
    syms = r.integers(0, 5, (rows, in_features)).astype(np.uint64)
    bpr = ((in_features + 9) // 10) * 3
    quint = np.zeros((rows, bpr), dtype=np.uint8)
    for g in range((in_features + 9) // 10):
        payload = np.zeros(rows, dtype=np.uint64)
        for off in range(10):
            col = g * 10 + off
            if col < in_features:
                payload += syms[:, col] * (5 ** off)
        quint[:, g * 3] = (payload & 0xFF).astype(np.uint8)
        quint[:, g * 3 + 1] = ((payload >> 8) & 0xFF).astype(np.uint8)
        quint[:, g * 3 + 2] = ((payload >> 16) & 0xFF).astype(np.uint8)
    return {"quint5_q": quint, "in_features": np.array([in_features], dtype=np.uint32)}


def ref_phonon(inp):
    quint = inp["quint5_q"]
    in_features = int(inp["in_features"][0])
    rows = quint.shape[0]
    words = in_features // 16
    base = np.zeros((rows, words), dtype=np.uint32)
    resid = np.zeros((rows, words), dtype=np.uint32)
    divisors = [1, 5, 25, 125, 625, 3125, 15625, 78125, 390625, 1953125]
    table = {0: (0, 0), 1: (0, 2), 2: (1, 1), 3: (2, 0), 4: (2, 2)}
    for word in range(words):
        for j in range(16):
            li = word * 16 + j
            group, off = li // 10, li % 10
            payload = (quint[:, group * 3].astype(np.uint32)) | \
                      (quint[:, group * 3 + 1].astype(np.uint32) << 8) | \
                      (quint[:, group * 3 + 2].astype(np.uint32) << 16)
            symbol = (payload // divisors[off]) % 5
            b_code, r_code = np.vectorize(table.get)(symbol)
            base[:, word] |= b_code.astype(np.uint32) << np.uint32(2 * j)
            resid[:, word] |= r_code.astype(np.uint32) << np.uint32(2 * j)
    return [base, resid]


# ---------------------------------------------------------------- CBQ shared
CBQ = dict(E=8, O=512, K=4096, R=8, N=1)


def build_cbq_stack(seed=12):
    r = rng(seed)
    E, O, K = CBQ["E"], CBQ["O"], CBQ["K"]
    qs = r.integers(0, 256, (E, O, K // 8)).astype(np.uint8)
    qh = r.integers(0, 2 ** 16, (E, O, K // 32)).astype(np.uint16)
    d = bf16(r.standard_normal((E, O, K // 256)) * 0.05 + 0.1)
    return qs, qh, d


def build_cbq_common(seed=12):
    r = rng(seed)
    qs, qh, d = build_cbq_stack(seed)
    K = CBQ["K"]
    x = bf16(r.standard_normal((CBQ["N"], K)) * 0.5)
    eidx = r.integers(0, CBQ["E"], CBQ["R"]).astype(np.uint32)
    # decode convention: one token, all routed experts share tokidx 0
    # (upstream: tokidx = mx.repeat(mx.arange(B*T), Kt))
    tokidx = np.zeros(CBQ["R"], dtype=np.uint32)
    grid_lut = _grid_lut()                                # (2048,8) int8
    return {"x": x, "eidx": eidx, "tokidx": tokidx,
            "cb_qs": qs, "cb_qh": qh, "cb_d": d,
            "grid": grid_lut.reshape(-1)}


def _cbq_w(inp, e):
    """Dequantize expert e's rows to (O,K) fp64 per dequant_ref semantics."""
    qs, qh, dd = inp["cb_qs"], inp["cb_qh"], inp["cb_d"]
    grid = _grid_lut()
    O, K8 = qs.shape[1], qs.shape[2]
    K = K8 * 8
    hi = np.stack([(qh[e] >> s) & 7 for s in (0, 3, 6, 9)], axis=-1)   # (O,K/32,4)
    gi = qs[e].astype(np.uint32).reshape(O, K // 32, 4) | (hi << 8)     # (O,K/32,4)
    dl = dd[e].astype(np.float64).repeat(8, axis=1) * \
        (2 * ((qh[e] >> 12) & 7).astype(np.float64) + 1)                # (O,K/32)
    dsg = np.where(((qh[e] >> 15) & 1).astype(bool), -0.125, 0.125)
    pat = grid[gi.reshape(-1)].reshape(O, K // 32, 4, 8).astype(np.float64)
    w = (pat + dsg[..., None, None]) * dl[..., None, None]
    return w.reshape(O, K)


def ref_cbq_mm(inp):
    x = inp["x"].astype(np.float64)
    eidx, tokidx = inp["eidx"], inp["tokidx"]
    R, O = eidx.shape[0], CBQ["O"]
    y = np.zeros((R, O))
    for r in range(R):
        w = _cbq_w(inp, int(eidx[r]))
        y[r] = bf16(np.float32(x[int(tokidx[r])] @ w.T))
    return [y]


def _situ(yv, gg):
    uu = float(bf16(np.float32(yv)))
    return float(bf16(np.float32(
        (4.0 * np.tanh(gg / 4.0)) * (1.0 / (1.0 + np.exp(-gg))) * (25.0 * np.tanh(uu / 25.0)))))


def ref_cbq_situ(inp):
    base = ref_cbq_mm(inp)
    gbuf = inp["gbuf"].astype(np.float64)
    R, O = base[0].shape
    out = np.zeros((R, O))
    for r in range(R):
        for o in range(O):
            out[r, o] = _situ(base[0][r, o], gbuf[r, o])
    return [out]


def ref_cbq_glu(inp):
    # two independent stacks through the same math
    keys_a = {k: inp[k] for k in ("x", "eidx", "tokidx")}
    a = {**keys_a, "cb_qs": inp["a_qs"], "cb_qh": inp["a_qh"], "cb_d": inp["a_d"], "grid": inp["grid"]}
    b = {**keys_a, "cb_qs": inp["b_qs"], "cb_qh": inp["b_qh"], "cb_d": inp["b_d"], "grid": inp["grid"]}
    return ref_cbq_mm(a) + ref_cbq_mm(b)


def build_cbq_glu(seed=13):
    base = build_cbq_common(seed)
    qs2, qh2, d2 = build_cbq_stack(seed + 1)
    out = dict(base)
    out.update({"a_qs": base["cb_qs"], "a_qh": base["cb_qh"], "a_d": base["cb_d"],
                "b_qs": qs2, "b_qh": qh2, "b_d": d2})
    return out


def build_cbq_situ(seed=14):
    out = build_cbq_common(seed)
    r = rng(seed + 100)
    out["gbuf"] = bf16(r.standard_normal((CBQ["R"], CBQ["O"])) * 0.5)
    return out


def build_cbq_gd(seed=15):
    out = build_cbq_common(seed)
    r = rng(seed + 200)
    out["cot"] = bf16(r.standard_normal((CBQ["R"], CBQ["O"])) * 0.5)
    return out


def ref_cbq_gd(inp):
    cot = inp["cot"].astype(np.float64)
    grid64 = _grid_lut().reshape(-1).view(np.int64)       # (2048,)
    qs, qh = inp["cb_qs"], inp["cb_qh"]
    x = inp["x"].astype(np.float64)
    K, O = CBQ["K"], CBQ["O"]
    gd = np.zeros((CBQ["E"], O, K // 256))
    byte_shifts = 8 * np.arange(8)
    for r_i in range(CBQ["R"]):
        e = int(inp["eidx"][r_i])
        tok = int(inp["tokidx"][r_i])
        xv_base = x[tok]
        for o in range(O):
            ct = float(cot[r_i, o])
            if ct == 0.0:
                continue
            for sup in range(K // 256):
                sbs = sup * 8 + np.arange(8)
                hs = qh[e, o, sbs].astype(np.int64)
                mult = (2 * ((hs >> 12) & 7) + 1).astype(np.float64)      # (8,)
                dele = np.where((hs & 0x8000) != 0, -0.125, 0.125)
                gi = qs[e, o, sbs[:, None] * 4 + np.arange(4)].astype(np.int64)
                gi |= (((hs[:, None] >> (3 * np.arange(4))) & 7) << 8)
                gv = grid64[gi]                                           # (8,4)
                gbytes = ((gv[..., None] >> byte_shifts) & 0xFF).astype(np.int64)
                xs = np.stack([xv_base[s * 32 + np.arange(4)[:, None] * 8 + np.arange(8)]
                               for s in sbs])                             # (8,4,8)
                p32 = ((gbytes + dele[..., None, None]) * xs).sum(axis=(1, 2))
                gd[e, o, sup] += ct * float((mult * p32).sum())
    return [gd]


def ref_cbq_gx(inp):
    cot = inp["cot"].astype(np.float64)
    grid64 = _grid_lut().reshape(-1).view(np.int64)
    qs, qh, dd = inp["cb_qs"], inp["cb_qh"], inp["cb_d"]
    K, O = CBQ["K"], CBQ["O"]
    n_tok = int(inp["tokidx"].max()) + 1
    gx = np.zeros((n_tok, K))
    byte_shifts = 8 * np.arange(8)
    for r_i in range(CBQ["R"]):
        e = int(inp["eidx"][r_i])
        tok = int(inp["tokidx"][r_i])
        for g8 in range(K // 8):
            sb, l, sup = g8 // 4, g8 % 4, g8 // 32
            ct = cot[r_i]                                                  # (O,)
            nz = ct != 0.0
            hs = qh[e, :, sb].astype(np.int64)
            dele = np.where((hs & 0x8000) != 0, -0.125, 0.125)
            gi = qs[e, :, sb * 4 + l].astype(np.int64) | \
                (((hs >> (3 * l)) & 7) << 8)
            gv = grid64[gi]
            gj = ((gv[:, None] >> byte_shifts) & 0xFF).astype(np.int64)    # (O,8)
            dsc = dd[e, :, sup].astype(np.float64) * \
                (2 * ((hs >> 12) & 7) + 1).astype(np.float64)
            wgt = np.where(nz, ct * dsc, 0.0)
            # grad wrt x carries no x factor: d(part)/dx = (g + delta)
            gx[tok, g8 * 8:(g8 + 1) * 8] += \
                (wgt[:, None] * (gj + dele[:, None])).sum(axis=0)
    return [gx]


# ---------------------------------------------------------------- kda glue
GLUE_H, GLUE_DIM = 96, 128
GLUE_C = GLUE_H * GLUE_DIM            # 12288


def _mlx_sigmoid_bf16(v):
    e = bf16(np.float32(np.exp(abs(float(v)))))
    den = bf16(np.float32(1.0 + float(e)))
    y = bf16(np.float32(1.0 / float(den)))
    return float(y) if float(v) < 0 else float(bf16(np.float32(1.0 - float(y))))


def build_glue_pre(seed=16):
    r = rng(seed)
    c = GLUE_C
    hp = bf16(r.standard_normal(49376) * 0.5)
    z = bf16(r.standard_normal(c) * 0.5)
    s0 = bf16(r.standard_normal(3 * c) * 0.5)
    s1 = bf16(r.standard_normal(3 * c) * 0.5)
    s2 = bf16(r.standard_normal(3 * c) * 0.5)
    wq = bf16(r.standard_normal(4 * c) * 0.3)
    wk = bf16(r.standard_normal(4 * c) * 0.3)
    wv = bf16(r.standard_normal(4 * c) * 0.3)
    dtb = r.standard_normal(c).astype(np.float32)
    alog = r.standard_normal(GLUE_DIM).astype(np.float32)
    cons = np.array([-2.0, 0.7, 0.49, 1e-6 / GLUE_DIM], dtype=np.float32)
    return {"hp": hp, "z": z, "s0": s0, "s1": s1, "s2": s2, "wq": wq, "wk": wk,
            "wv": wv, "dtb": dtb, "alog": alog, "cons": cons}


def ref_glue_pre(inp):
    c = GLUE_C
    f = lambda k: inp[k].astype(np.float64)
    hp, z, s0, s1, s2 = f("hp"), f("z"), f("s0"), f("s1"), f("s2")
    wq, wk, wv = f("wq"), f("wk"), f("wv")
    dtb, alog = f("dtb"), f("alog")
    cs = inp["cons"].astype(np.float32).astype(np.float64)
    q = np.zeros((1, 1, GLUE_H, GLUE_DIM))
    k = np.zeros_like(q)
    v = np.zeros_like(q)
    g = np.zeros_like(q)
    beta = np.zeros((1, 1, GLUE_H))
    ns0 = np.zeros((1, 3, c))
    ns1 = np.zeros((1, 3, c))
    ns2 = np.zeros((1, 3, c))
    for h in range(GLUE_H):
        conv_q = np.zeros(GLUE_DIM)
        conv_k = np.zeros(GLUE_DIM)
        conv_v = np.zeros(GLUE_DIM)
        for d in range(GLUE_DIM):
            ci = h * GLUE_DIM + d
            taps_q = [s0[ci], s0[c + ci], s0[2 * c + ci], hp[ci]]
            taps_k = [s1[ci], s1[c + ci], s1[2 * c + ci], hp[c + ci]]
            taps_v = [s2[ci], s2[c + ci], s2[2 * c + ci], hp[2 * c + ci]]
            aq = sum(taps_q[t] * wq[ci * 4 + t] for t in range(4))
            ak = sum(taps_k[t] * wk[ci * 4 + t] for t in range(4))
            av = sum(taps_v[t] * wv[ci * 4 + t] for t in range(4))
            cq, ck, cv = float(bf16(np.float32(aq))), float(bf16(np.float32(ak))), float(bf16(np.float32(av)))
            sq_, sk_, sv_ = _mlx_sigmoid_bf16(cq), _mlx_sigmoid_bf16(ck), _mlx_sigmoid_bf16(cv)
            conv_q[d] = float(bf16(np.float32(cq * sq_)))
            conv_k[d] = float(bf16(np.float32(ck * sk_)))
            conv_v[d] = float(bf16(np.float32(cv * sv_)))
            ns0[0, 0, ci], ns0[0, 1, ci], ns0[0, 2, ci] = s0[c + ci], s0[2 * c + ci], hp[ci]
            ns1[0, 0, ci], ns1[0, 1, ci], ns1[0, 2, ci] = s1[c + ci], s1[2 * c + ci], hp[c + ci]
            ns2[0, 0, ci], ns2[0, 1, ci], ns2[0, 2, ci] = s2[c + ci], s2[2 * c + ci], hp[2 * c + ci]
            zf = z[ci] + dtb[ci]
            s = 1.0 / (1.0 + np.exp(-(np.exp(alog[d]) * zf)))
            g[0, 0, h, d] = np.exp(cs[0] * s)
        mq = conv_q * conv_q
        mk = conv_k * conv_k
        # simd_sum per 32-lane group, then the 4 group sums: fp32 tree either way
        mq = mq.reshape(4, 32).sum(axis=1).sum() / GLUE_DIM + cs[3]
        mk = mk.reshape(4, 32).sum(axis=1).sum() / GLUE_DIM + cs[3]
        nq = bf16(conv_q * (1.0 / np.sqrt(mq)))
        nk = bf16(conv_k * (1.0 / np.sqrt(mk)))
        q[0, 0, h] = bf16(nq.astype(np.float32) * np.float32(cs[2]))
        k[0, 0, h] = bf16(nk.astype(np.float32) * np.float32(cs[1]))
        v[0, 0, h] = bf16(conv_v.astype(np.float32))
        beta[0, 0, h] = _mlx_sigmoid_bf16(hp[49280 + h])
    return [q, k, v, g, beta, ns0, ns1, ns2]


def build_glue_post(seed=17):
    r = rng(seed)
    return {"inp": bf16(r.standard_normal(GLUE_C) * 0.5),
            "hp": bf16(r.standard_normal(49376) * 0.5),
            "w": bf16(r.standard_normal(GLUE_DIM) * 0.3),
            "cons": np.array([1e-6], dtype=np.float32)}


def ref_glue_post(inp):
    f = lambda k: inp[k].astype(np.float64)
    inp_b, hp, w = f("inp"), f("hp"), f("w")
    eps = float(inp["cons"].astype(np.float32)[0])
    y = np.zeros(GLUE_C)
    # per head reduction over 128 channels
    for h in range(GLUE_H):
        xs = inp_b[h * GLUE_DIM:(h + 1) * GLUE_DIM]
        m = (xs * xs).sum() / GLUE_DIM + eps
        inv = 1.0 / np.sqrt(m)
        for d in range(GLUE_DIM):
            ci = h * GLUE_DIM + d
            nx0 = float(bf16(np.float32(xs[d] * inv)))
            nx = float(bf16(np.float32(nx0 * float(w[d]))))
            sg_ = _mlx_sigmoid_bf16(hp[36864 + ci])
            y[ci] = float(bf16(np.float32(nx * sg_)))
    return [y.reshape(1, 1, GLUE_C)]


def build_route_fused(seed=18):
    r = rng(seed)
    NE = 896
    gate = bf16(np.round(r.standard_normal(NE) * 20.0) * 0.05)  # spaced, bf16 grid
    bias = bf16(r.standard_normal(NE) * 0.01)
    lo_map = r.integers(0, 512, NE).astype(np.int32)
    hi_map = r.integers(0, 512, NE).astype(np.int32)
    meta = np.array([16, 4], dtype=np.int32)
    return {"gate_out": gate, "bias": bias, "lo_map": lo_map, "hi_map": hi_map, "meta": meta}


def ref_route_fused(inp):
    gate = inp["gate_out"].astype(np.float32).astype(np.float64)
    bias = inp["bias"].astype(np.float32).astype(np.float64)
    lo_map, hi_map = inp["lo_map"], inp["hi_map"]
    K, KHI = int(inp["meta"][0]), int(inp["meta"][1])
    sc = 1.0 / (1.0 + np.exp(-gate))
    bi = sc + bias
    sel, selw = [], []
    for _ in range(K):
        best_i = int(np.argmax(bi))          # max value, lowest index on ties
        sel.append(best_i)
        selw.append(sc[best_i])
        bi[best_i] = -1e30
    tot = sum(selw) + 1e-20
    lo_idx = np.zeros(K, dtype=np.uint32)
    w_lo = np.zeros(K)
    hi_idx = np.zeros(KHI, dtype=np.uint32)
    w_hi = np.zeros(KHI)
    pool = []
    for kk in range(K):
        ge = sel[kk]
        wn = selw[kk] / tot
        lo, hi = int(lo_map[ge]), int(hi_map[ge])
        lo_idx[kk] = lo
        w_lo[kk] = float(bf16(np.float32(0.0 if hi >= 0 else wn)))
        pool.append(wn if hi >= 0 else -1.0)
    for j in range(KHI):
        best_k, bv = -1, 0.0
        for kk in range(K):
            if pool[kk] > bv:
                bv, best_k = pool[kk], kk
        if best_k >= 0:
            hi_idx[j] = hi_map[sel[best_k]] + 1
            w_hi[j] = float(bf16(np.float32(bv)))
            pool[best_k] = -1.0
        else:
            hi_idx[j] = 0
            w_hi[j] = 0.0
    return [lo_idx.reshape(1, 1, K), w_lo.reshape(1, 1, K),
            hi_idx.reshape(1, 1, KHI), w_hi.reshape(1, 1, KHI)]


def build_situ(seed=19):
    r = rng(seed)
    H = 512
    return {"gu": bf16(r.standard_normal(2 * H) * 2.0),
            "cons": np.array([4.0, 25.0, float(H)], dtype=np.float32)}


def _situ_ref(g, u, b1, b2):
    return float(bf16(np.float32((b1 * np.tanh(g / b1)) * (1.0 / (1.0 + np.exp(-g))) * (b2 * np.tanh(u / b2)))))


def ref_situ(inp):
    gu = inp["gu"].astype(np.float64)
    cs = inp["cons"].astype(np.float32).astype(np.float64)
    H = int(cs[2])
    out = np.zeros(H)
    for i in range(H):
        out[i] = _situ_ref(gu[i], gu[H + i], cs[0], cs[1])
    return [out.reshape(1, 1, H)]


def build_situ_pair(seed=20):
    r = rng(seed)
    n = 256
    return {"ga": bf16(r.standard_normal(n) * 2.0),
            "ub": bf16(r.standard_normal(n) * 2.0),
            "cons": np.array([4.0, 25.0], dtype=np.float32)}


def ref_situ_pair(inp):
    ga = inp["ga"].astype(np.float64)
    ub = inp["ub"].astype(np.float64)
    cs = inp["cons"].astype(np.float32).astype(np.float64)
    out = np.zeros(ga.shape[0])
    for i in range(ga.shape[0]):
        out[i] = _situ_ref(ga[i], ub[i], cs[0], cs[1])
    return [out.reshape(inp["ga"].shape)]


BF16 = "bfloat16"
FP16 = "float16"
FP32 = "float32"
U32, I32, U8, U16 = "uint32", "int32", "uint8", "uint16"
I8 = "int8"

# The element type the upstream call site actually passes (it fixes the
# generated MSL signature). bfloat16 builders hand back float32 arrays whose
# values are already on the bf16 grid; the runner casts them to mx.bfloat16.
IN_DTYPES = {
    "bitlinear_matmul": [FP16, U8, FP16],
    "fused_double_norm_rope": [BF16] * 8,
    "fused_single_norm_rope": [BF16] * 5,
    "inkling_banded_mask": [BF16, BF16],
    "inkling_banded_mask_v2": [BF16, BF16, FP32],
    "inkling_sconv_decode": [BF16, FP32, FP32, BF16],
    "inkling_moe_route": [BF16, BF16, FP32],
    "inkling_moe_down_combine": [BF16, U32, BF16, BF16, U32, BF16],
    "mlx_vlm_llguidance_mask": [BF16, I32],
    "custom_depthwise_conv1d": [FP32, FP32, I32],
    "qk_relu_squared": [FP16, FP16, FP16],
    "mlx_audio_phonon_unpack_base5_v1": [U8, U32],
    "cbq_gather_mm": [BF16, U32, U32, U8, U16, BF16, I8],
    "cbq_gather_mm_v2": [BF16, U32, U32, U8, U16, BF16, I8],
    "cbq_gather_mm_v3": [BF16, U32, U32, U8, U16, BF16, I8],
    "cbq_gather_mm_v4": [BF16, U32, U32, U8, U16, BF16, I8],
    "cbq_gather_mm_v3_situ": [BF16, U32, U32, U8, U16, BF16, I8, BF16],
    "cbq_gather_mm_v4_situ": [BF16, U32, U32, U8, U16, BF16, I8, BF16],
    "cbq_gather_mm_glu": [BF16, U32, U32, U8, U16, BF16, U8, U16, BF16, I8],
    "cbq_grad_d": [BF16, BF16, U32, U32, U8, U16, I8],
    "cbq_grad_x": [BF16, U32, U32, U8, U16, BF16, I8],
    "kda_glue_pre": [BF16, BF16, BF16, BF16, BF16, BF16, BF16, BF16, FP32, FP32, FP32],
    "kda_glue_post": [BF16, BF16, BF16, FP32],
    "moe_route_fused": [BF16, BF16, I32, I32, I32],
    "situ_fused": [BF16, FP32],
    "situ_pair_fused": [BF16, BF16, FP32],
}

SPECS = [
    dict(name="bitlinear_matmul", builder=build_bitlinear, ref=ref_bitlinear,
         inputs=("x", "packed_weights", "weight_scale"), outputs=("out",),
         out_shapes=[(4, 2048)], out_dtypes=[FP16], tol=[REL16],
         template=[("T", FP16), ("invert_weight_scales", False),
                   ("in_features", 2048), ("out_features", 2048)],
         grid=(32, 2048, 1), threadgroup=(32, 1, 1)),
    dict(name="fused_double_norm_rope", builder=build_flux_double, ref=ref_flux_double,
         inputs=("img_qkv", "txt_qkv", "norm_q", "norm_k", "norm_added_q",
                 "norm_added_k", "cos_vals", "sin_vals"),
         outputs=("q_out", "k_out", "v_out"),
         out_shapes=[(1, 24, 96, 128)] * 3, out_dtypes=[BF16] * 3, tol=[REL] * 3,
         source_format=dict(S_IMG=64, S_TXT=32, H_DIM=24, D_DIM=128, DIM=3072,
                            EPS="1e-6", ELEMS=4),
         template=[("T", BF16)],
         grid=(1 * (64 + 32) * 24 * 32, 1, 1), threadgroup=(32, 1, 1)),
    dict(name="fused_single_norm_rope", builder=build_flux_single, ref=ref_flux_single,
         inputs=("fused", "norm_q", "norm_k", "cos_vals", "sin_vals"),
         outputs=("q_out", "k_out", "v_out"),
         out_shapes=[(1, 24, 96, 128)] * 3, out_dtypes=[BF16] * 3, tol=[REL] * 3,
         source_format=dict(S_DIM=96, H_DIM=24, D_DIM=128, FUSED_DIM=27648,
                            DIM=3072, EPS="1e-6", ELEMS=4),
         template=[("T", BF16)],
         grid=(1 * 96 * 24 * 32, 1, 1), threadgroup=(32, 1, 1)),
    dict(name="inkling_banded_mask", builder=build_mask, ref=ref_mask,
         inputs=("rel", "proj"), outputs=("out",),
         out_shapes=[(2, 4, 16, 32)], out_dtypes=[BF16], tol=[REL],
         template=[("T", BF16), ("B", 2), ("H", 4), ("LQ", 16), ("S", 32),
                   ("Q_OFF", 16), ("D_REL", 8), ("REL_EXTENT", 16), ("SLIDING", 32)],
         grid=(32, 16, 8), threadgroup=(8, 8, 1)),
    dict(name="inkling_banded_mask_v2", builder=build_mask, ref=ref_mask_v2,
         inputs=("rel", "proj", "kshape"), outputs=("out",),
         out_shapes=[(2, 4, 16, 32)], out_dtypes=[BF16], tol=[REL],
         ensure_row_contiguous=False,
         template=[("T", BF16), ("D_REL", 8), ("REL_EXTENT", 16), ("SLIDING", 32)],
         grid=(32, 16, 8), threadgroup=(8, 8, 1)),
    dict(name="inkling_sconv_decode", builder=build_sconv, ref=ref_sconv,
         inputs=("x", "state", "w", "res"), outputs=("out", "nstate"),
         out_shapes=[(2, 4, 512), (2, 3, 512)], out_dtypes=[BF16, FP32],
         tol=[REL, REL],
         template=[("T", BF16), ("B", 2), ("L", 4), ("C", 512), ("K", 4),
                   ("HAS_RES", True)],
         grid=(512, 2, 1), threadgroup=(32, 1, 1)),
    dict(name="inkling_moe_route", builder=build_route, ref=ref_route,
         inputs=("logits", "corr", "wscale"), outputs=("idx", "wk", "gamma"),
         out_shapes=[(2, 8), (2, 8), (2, 512)], out_dtypes=[U32, BF16, BF16],
         tol=[EXACT, REL, REL],
         template=[("T", BF16), ("N", 2), ("R", 64), ("SH", 2), ("K", 8), ("I", 256)],
         grid=(32, 2, 1), threadgroup=(32, 1, 1)),
    dict(name="inkling_moe_down_combine", builder=build_down_combine, ref=ref_down_combine,
         inputs=("xin", "wq", "sc", "bi", "idx", "wk"), outputs=("out",),
         out_shapes=[(1, 512)], out_dtypes=[BF16], tol=[REL],
         template=[("T", BF16), ("OUT", 512), ("IN", 2048), ("GROUPS", 32), ("K", 8)],
         grid=(256, 512, 1), threadgroup=(256, 1, 1)),
    dict(name="mlx_vlm_llguidance_mask", builder=build_llg, ref=ref_llg,
         inputs=("logits", "mask"), outputs=("out",),
         out_shapes=[(2, 4096)], out_dtypes=[BF16], tol=[EXACT],
         template=[("T", BF16)], grid=(4096, 2, 1), threadgroup=(256, 1, 1)),
    dict(name="custom_depthwise_conv1d", builder=build_depthwise, ref=ref_depthwise,
         inputs=("inp", "weight", "params"), outputs=("out",),
         out_shapes=[(1, 64, 32)], out_dtypes=[FP32], tol=[("rel", 1e-5, 1e-3)],
         template=[("T", FP32)], grid=(32, 64, 1), threadgroup=(1, 1, 1)),
    dict(name="qk_relu_squared", builder=build_qk, ref=ref_qk,
         inputs=("q", "k", "scale"), outputs=("out",),
         out_shapes=[(2, 4, 16, 16)], out_dtypes=[FP16], tol=[REL],
         template=[("T", FP16)], grid=(16, 16, 2), threadgroup=(16, 16, 1)),
    dict(name="mlx_audio_phonon_unpack_base5_v1", builder=build_phonon, ref=ref_phonon,
         inputs=("quint5_q", "in_features"), outputs=("base_q", "residual_q"),
         out_shapes=[(512, 128), (512, 128)], out_dtypes=[U32, U32],
         tol=[EXACT, EXACT], grid=(65536, 1, 1), threadgroup=(256, 1, 1)),
    dict(name="cbq_gather_mm", builder=build_cbq_common, ref=ref_cbq_mm,
         inputs=("x", "eidx", "tokidx", "cb_qs", "cb_qh", "cb_d", "grid"),
         outputs=("y",), out_shapes=[(8, 512)], out_dtypes=[BF16], tol=[REL],
         template=[("InT", BF16), ("DT", BF16), ("K", 4096), ("O", 512), ("R", 8)],
         grid=(512, 8, 1), threadgroup=(64, 1, 1)),
    dict(name="cbq_gather_mm_v2", builder=build_cbq_common, ref=ref_cbq_mm,
         inputs=("x", "eidx", "tokidx", "cb_qs", "cb_qh", "cb_d", "grid"),
         outputs=("y",), out_shapes=[(8, 512)], out_dtypes=[BF16], tol=[REL],
         template=[("InT", BF16), ("DT", BF16), ("K", 4096), ("O", 512), ("R", 8)],
         grid=(256, 8, 1), threadgroup=(256, 1, 1)),
    dict(name="cbq_gather_mm_v3", builder=build_cbq_common, ref=ref_cbq_mm,
         inputs=("x", "eidx", "tokidx", "cb_qs", "cb_qh", "cb_d", "grid"),
         outputs=("y",), out_shapes=[(8, 512)], out_dtypes=[BF16], tol=[REL],
         template=[("InT", BF16), ("DT", BF16), ("K", 4096), ("O", 512), ("R", 8)],
         grid=(256 * 64, 8, 1), threadgroup=(256, 1, 1)),
    dict(name="cbq_gather_mm_v4", builder=build_cbq_common, ref=ref_cbq_mm,
         inputs=("x", "eidx", "tokidx", "cb_qs", "cb_qh", "cb_d", "grid"),
         outputs=("y",), out_shapes=[(8, 512)], out_dtypes=[BF16], tol=[REL],
         template=[("InT", BF16), ("DT", BF16), ("K", 4096), ("O", 512), ("R", 8)],
         grid=(128 * 16, 8, 1), threadgroup=(128, 1, 1)),
    dict(name="cbq_gather_mm_v3_situ", builder=build_cbq_situ, ref=ref_cbq_situ,
         inputs=("x", "eidx", "tokidx", "cb_qs", "cb_qh", "cb_d", "grid", "gbuf"),
         outputs=("y",), out_shapes=[(8, 512)], out_dtypes=[BF16], tol=[REL],
         template=[("InT", BF16), ("DT", BF16), ("K", 4096), ("O", 512), ("R", 8)],
         grid=(256 * 64, 8, 1), threadgroup=(256, 1, 1)),
    dict(name="cbq_gather_mm_v4_situ", builder=build_cbq_situ, ref=ref_cbq_situ,
         inputs=("x", "eidx", "tokidx", "cb_qs", "cb_qh", "cb_d", "grid", "gbuf"),
         outputs=("y",), out_shapes=[(8, 512)], out_dtypes=[BF16], tol=[REL],
         template=[("InT", BF16), ("DT", BF16), ("K", 4096), ("O", 512), ("R", 8)],
         grid=(128 * 16, 8, 1), threadgroup=(128, 1, 1)),
    dict(name="cbq_gather_mm_glu", builder=build_cbq_glu, ref=ref_cbq_glu,
         inputs=("x", "eidx", "tokidx", "a_qs", "a_qh", "a_d",
                 "b_qs", "b_qh", "b_d", "grid"),
         outputs=("yg", "yu"), out_shapes=[(8, 512), (8, 512)],
         out_dtypes=[BF16, BF16], tol=[REL, REL],
         template=[("InT", BF16), ("DT", BF16), ("K", 4096), ("O", 512), ("R", 8)],
         grid=(256 * 128, 8, 1), threadgroup=(256, 1, 1)),
    dict(name="cbq_grad_d", builder=build_cbq_gd, ref=ref_cbq_gd,
         inputs=("x", "cot", "eidx", "tokidx", "cb_qs", "cb_qh", "grid"),
         outputs=("grad_d",), out_shapes=[(8, 512, 16)], out_dtypes=[FP32],
         tol=[("rel", 1e-4)], atomic_outputs=True, init_value=0,
         template=[("InT", BF16), ("K", 4096), ("O", 512), ("R", 8)],
         grid=(512, 8, 1), threadgroup=(64, 1, 1)),
    dict(name="cbq_grad_x", builder=build_cbq_gd, ref=ref_cbq_gx,
         inputs=("cot", "eidx", "tokidx", "cb_qs", "cb_qh", "cb_d", "grid"),
         outputs=("grad_x",), out_shapes=[(1, 4096)], out_dtypes=[FP32],
         tol=[("rel", 1e-4)], atomic_outputs=True, init_value=0,
         template=[("InT", BF16), ("DT", BF16), ("K", 4096), ("O", 512), ("R", 8)],
         grid=(512, 8, 1), threadgroup=(64, 1, 1)),
    dict(name="kda_glue_pre", builder=build_glue_pre, ref=ref_glue_pre,
         inputs=("hp", "z", "s0", "s1", "s2", "wq", "wk", "wv", "dtb", "alog", "cons"),
         outputs=("q", "k", "v", "g", "beta", "ns0", "ns1", "ns2"),
         out_shapes=[(1, 1, 96, 128)] * 4 + [(1, 1, 96)] + [(1, 3, 12288)] * 3,
         out_dtypes=[BF16, BF16, BF16, FP32, BF16, BF16, BF16, BF16],
         tol=[REL, REL, REL, ("rel", 1e-5), REL, EXACT, EXACT, EXACT],
         header="kda_glue_pre.header.msl",
         grid=(96 * 128, 1, 1), threadgroup=(128, 1, 1)),
    dict(name="kda_glue_post", builder=build_glue_post, ref=ref_glue_post,
         inputs=("inp", "hp", "w", "cons"), outputs=("y",),
         out_shapes=[(1, 1, 12288)], out_dtypes=[BF16], tol=[REL],
         header="kda_glue_post.header.msl",
         grid=(96 * 128, 1, 1), threadgroup=(128, 1, 1)),
    dict(name="moe_route_fused", builder=build_route_fused, ref=ref_route_fused,
         inputs=("gate_out", "bias", "lo_map", "hi_map", "meta"),
         outputs=("lo_idx", "w_lo", "hi_idx", "w_hi"),
         out_shapes=[(1, 1, 16), (1, 1, 16), (1, 1, 4), (1, 1, 4)],
         out_dtypes=[U32, BF16, U32, BF16], tol=[EXACT, REL, EXACT, REL],
         grid=(896, 1, 1), threadgroup=(896, 1, 1)),
    dict(name="situ_fused", builder=build_situ, ref=ref_situ,
         inputs=("gu", "cons"), outputs=("out",),
         out_shapes=[(1, 1, 512)], out_dtypes=[BF16], tol=[REL],
         grid=(512, 1, 1), threadgroup=(256, 1, 1)),
    dict(name="situ_pair_fused", builder=build_situ_pair, ref=ref_situ_pair,
         inputs=("ga", "ub", "cons"), outputs=("out",),
         out_shapes=[(256,)], out_dtypes=[BF16], tol=[REL],
         grid=(256, 1, 1), threadgroup=(256, 1, 1)),
]


def get_spec(name):
    for spec in SPECS:
        if spec["name"] == name:
            return spec
    raise KeyError(name)
