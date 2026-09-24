import numpy as np, json

H = np.load("/tmp/vp/hidden.npy").astype(np.float32)
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

def bf16(a):
    a = np.ascontiguousarray(a, np.float32)
    u = a.view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return r.astype(np.uint32).view(np.float32)

Z = bf16(gdot(q) + b @ X.T)
m = Z.max(0)
C = 2 * gdot(q >> 1) + (0.5 * s + b) @ X.T
A = (16 * sa + np.abs(b)) @ L1.T
UB = C + 0.5 * sa @ L1.T + A / 4096
block = 128
nb = (V + block - 1) // block
cand = np.array([[blk * block + int(np.argmax(C[blk * block:(blk + 1) * block, t])) for blk in range(nb)] for t in range(N)])
LBs = np.array([Z[cand[t], t].max() for t in range(N)])
hit = (LBs == m).mean()
M = np.maximum(np.abs(LBs), np.abs(UB.max(0)))
G = 0.13 + M / 12800
surv = ((LBs[None] - UB) <= G[None] + np.abs(UB) / 128).sum(0)
print(f"LB* == m (argmax row is its block's top-C candidate): {hit:.3f}")
print(f"survivors median {np.median(surv):.0f} p90 {np.percentile(surv,90):.0f} max {surv.max()}  pruned mean {1 - surv.mean()/V:.5f}")
print("bytes/token pass1 MB", V * (768 + 128) / 1e6, "cand MB", nb * 1152 / 1e6, "surv MB median", np.median(surv) * 1152 / 1e6)
json.dump({"survivors": surv.tolist(), "lb_hit": float(hit)}, open("/tmp/vp/analysis3.json", "w"))
