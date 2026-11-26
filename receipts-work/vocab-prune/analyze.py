import numpy as np, json

H = np.load("/tmp/vp/hidden.npy").astype(np.float32)       # [N, 2048]
toks = np.load("/tmp/vp/tokens.npy")
qw = np.load("/tmp/vp/q.npy")                                 # [V, 256] uint32
s = np.load("/tmp/vp/scales.npy").astype(np.float32)          # [V, 32]
b = np.load("/tmp/vp/biases.npy").astype(np.float32)
V = qw.shape[0]; N = H.shape[0]
shifts = np.arange(8, dtype=np.uint32) * 4
q = ((qw[:, :, None] >> shifts) & 0xF).reshape(V, 32, 64).astype(np.uint8)
print("V", V, "N", N, "s>=0", bool((s >= 0).all()), "s", s.min(), s.max(), "b", b.min(), b.max())
Hg = H.reshape(N, 32, 64)
X = Hg.sum(-1)            # [N, 32]
L1 = np.abs(Hg).sum(-1)
L2 = np.sqrt((Hg ** 2).sum(-1))

def gdot(codes):  # [V, N]: sum_g s_vg * sum_k codes[v,g,k] x[n,g,k]
    acc = np.zeros((V, N), np.float32)
    for g in range(32):
        acc += s[:, g:g+1] * (codes[:, g, :].astype(np.float32) @ Hg[:, g, :].T)
    return acc

Z = gdot(q) + b @ X.T                                   # exact-ish logits [V, N]
m = Z.max(0); am = Z.argmax(0)
C2 = 4 * gdot(q >> 2) + (1.5 * s + b) @ X.T
C1 = 8 * gdot(q >> 3) + (3.5 * s + b) @ X.T
r2n = np.sqrt((((q & 3).astype(np.float32) - 1.5) ** 2).sum(-1))
res = {"m": m.tolist(), "argmax_eq_tok": int((am == toks).sum())}
for name, C, R in (("2bit-L1", C2, 1.5 * np.abs(s) @ L1.T),
                   ("2bit-CS", C2, (np.abs(s) * r2n) @ L2.T),
                   ("1bit-L1", C1, 3.5 * np.abs(s) @ L1.T)):
    lb = (C - R).max(0)
    surv = (C + R >= lb).sum(0)
    res[name] = surv.tolist()
    print(f"{name}: survivors median {np.median(surv):.0f} p90 {np.percentile(surv, 90):.0f} max {surv.max()} "
          f"pruned frac mean {1 - surv.mean() / V:.5f}; R median {np.median(R):.2f}; (m - C) median {np.median(m - C):.2f}")
Zs = np.sort(Z, 0)
print("m min/median/max", m.min(), np.median(m), m.max(), "count m<13", int((m < 13).sum()), "of", N)
print("top1-top2 gap min/median", (Zs[-1] - Zs[-2]).min(), np.median(Zs[-1] - Zs[-2]))
print("argmax==generated token", res["argmax_eq_tok"], "of", N)
print("rows within 5 of max: median", np.median((Z >= m - 5).sum(0)), " within 10:", np.median((Z >= m - 10).sum(0)))
print("x L1/L2 per token median", np.median(np.abs(H).sum(1)), np.median(np.sqrt((H**2).sum(1))), "max|x| median", np.median(np.abs(H).max(1)))
json.dump(res, open("/tmp/vp/analysis.json", "w"))
