import numpy as np, json

H = np.load("/tmp/vp/hidden.npy").astype(np.float32)
toks = np.load("/tmp/vp/tokens.npy")
qw = np.load("/tmp/vp/q.npy")
s = np.load("/tmp/vp/scales.npy").astype(np.float32)
b = np.load("/tmp/vp/biases.npy").astype(np.float32)
V = qw.shape[0]; N = H.shape[0]
shifts = np.arange(8, dtype=np.uint32) * 4
q = ((qw[:, :, None] >> shifts) & 0xF).reshape(V, 32, 64).astype(np.uint8)
Hg = H.reshape(N, 32, 64)
X = Hg.sum(-1); L1 = np.abs(Hg).sum(-1)
sa = np.abs(s)

def gdot(codes):
    acc = np.zeros((V, N), np.float32)
    for g in range(32):
        acc += s[:, g:g+1] * (codes[:, g, :].astype(np.float32) @ Hg[:, g, :].T)
    return acc

Z = gdot(q) + b @ X.T
m = Z.max(0)
Wf = (s[:, :, None] * q + b[:, :, None]).reshape(V, 2048)
wn = np.sqrt((Wf.astype(np.float64) ** 2).sum(1)).astype(np.float32)
xn = np.sqrt((H.astype(np.float64) ** 2).sum(1)).astype(np.float32)
cs = wn[:, None] * xn[None, :]
print("pure CS: rows with |w||x| >= m: median frac", np.median((cs >= m).mean(0)), " median ratio (||w|| ||x||)/m for argmax row:",
      np.median(cs[Z.argmax(0), np.arange(N)] / m))
res = {}
for keep in (1, 2, 3):
    drop = 4 - keep
    half = ((1 << drop) - 1) / 2.0
    hi = (q >> drop).astype(np.uint8)
    C = (1 << drop) * gdot(hi) + (half * s + b) @ X.T
    R = half * sa @ L1.T
    for tname, thr in (("self", (C - R).max(0)), ("oracle-m", m)):
        surv = (C + R >= thr).sum(0)
        frac = surv / V
        res[f"{keep}bit-{tname}"] = surv.tolist()
        print(f"{keep}-bit MSB sketch, threshold {tname}: survivors median {np.median(surv):.0f} "
              f"p10 {np.percentile(surv,10):.0f} p90 {np.percentile(surv,90):.0f}; pruned median {1-np.median(frac):.4f}; R median {np.median(R):.2f}")
print("logit spread: median over tokens of (m - median_j z_j):", np.median(m - np.median(Z, 0)), " std_j z median:", np.median(Z.std(0)))
json.dump(res, open("/tmp/vp/analysis2.json", "w"))
