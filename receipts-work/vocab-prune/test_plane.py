import numpy as np
import mlx.core as mx

SH0 = mx.arange(0, 30, 3, dtype=mx.uint32)      # codes 0..9 of a 32-code block
SH1 = mx.arange(1, 31, 3, dtype=mx.uint32)      # codes 11..20
SH2 = mx.arange(2, 32, 3, dtype=mx.uint32)      # codes 22..31
NIB = mx.arange(0, 32, 4, dtype=mx.uint32)


def plane3(w, chunk=4096):
    """a = q >> 1 of every 4-bit code, code e of a 64-code group at bit 3e
    of its six words (the layout shaders/qmm_vec.comp QMM_VEC_GREEDY reads)."""
    parts = []
    for r in range(0, w.shape[0], chunk):
        c = w[r:r + chunk]
        a = (((c[..., None] >> NIB) & 15) >> 1).reshape(c.shape[0], -1, 32)
        w0 = (a[..., 0:10] << SH0).sum(-1) + ((a[..., 10] & 3) << 30)
        w1 = (a[..., 10] >> 2) + (a[..., 11:21] << SH1).sum(-1) + ((a[..., 21] & 1) << 31)
        w2 = (a[..., 21] >> 1) + (a[..., 22:32] << SH2).sum(-1)
        p = mx.stack([w0, w1, w2], -1).reshape(c.shape[0], -1).astype(mx.uint32)
        mx.eval(p)
        parts.append(p)
    return mx.concatenate(parts)


def shader_a(pw, e):  # GREEDY_A, verbatim arithmetic
    word, off = (e * 3) // 32, (e * 3) % 32
    if off <= 29:
        return (int(pw[word]) >> off) & 7
    return ((int(pw[word]) >> off) | ((int(pw[word + 1]) << ((32 - off) & 31)) & 0xFFFFFFFF)) & 7


rng = np.random.default_rng(0)
wq = rng.integers(0, 2**32, size=(37, 256), dtype=np.uint64).astype(np.uint32)
P = plane3(mx.array(wq), chunk=16)
print("dtype", P.dtype, "shape", P.shape)
P = np.array(P)
q = ((wq[:, :, None] >> (np.arange(8, dtype=np.uint32) * 4)) & 15).reshape(37, 2048)
bad = 0
for r in range(37):
    for g in range(32):
        pw = list(P[r, g * 6:(g + 1) * 6]) + [0]
        for e in range(64):
            bad += shader_a(pw, e) != (q[r, g * 64 + e] >> 1)
print("mismatches", bad)
assert bad == 0
