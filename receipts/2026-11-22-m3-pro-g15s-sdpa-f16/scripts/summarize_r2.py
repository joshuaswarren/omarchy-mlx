"""Round 2 summary: base wheel default vs head wheel default (per call, end to end), and the CPU-disabled same-wheel A/B.
usage: summarize_r2.py RUN_DIR"""
import glob, json, statistics, sys
import numpy as np
d = sys.argv[1]
def load(pat):
    out = []
    for p in sorted(glob.glob(f"{d}/{pat}")):
        out.append((json.load(open(p)), np.load(p[:-5] + ".npz")))
    return out
f16 = lambda b: b.astype(np.uint16).view(np.float16).astype(np.float64)
def percall(a_runs, b_runs, la, lb):
    print("provenance:", sorted({j["provenance"] for j, _ in a_runs + b_runs}))
    print(f"heads kv width keys | dispatches {la}/{lb} | independent us {la} -> {lb} (x) | chain us (x) | max|{lb}-{la}| | bits stable {la}/{lb}")
    for i, r in enumerate(a_runs[0][0]["rows"]):
        k = f"c{i}"
        ea = [j["rows"][i] for j, _ in a_runs]; eb = [j["rows"][i] for j, _ in b_runs]
        ia = statistics.median(x["ind_us_median"] for x in ea); ib = statistics.median(x["ind_us_median"] for x in eb)
        ca = statistics.median(x["chain_us_median"] for x in ea); cb = statistics.median(x["chain_us_median"] for x in eb)
        diff = max(float(np.abs(f16(za[k]) - f16(zb[k])).max()) for _, za in a_runs for _, zb in b_runs)
        sa = all(np.array_equal(z[k], a_runs[0][1][k]) for _, z in a_runs); sb = all(np.array_equal(z[k], b_runs[0][1][k]) for _, z in b_runs)
        print(f"{r['heads']:2d} {r['kv_heads']} {r['width']:3d} {r['keys']:4d} | {sorted({x['vk_compute_dispatches'] for x in ea})}/{sorted({x['vk_compute_dispatches'] for x in eb})} | "
              f"{ia:7.1f} -> {ib:6.1f} ({ia/ib:4.2f}) | {ca:7.1f} -> {cb:6.1f} ({ca/cb:4.2f}) | {diff:.2e} | {sa}/{sb}")
def e2e(runs, key):
    rows = {}
    for j, _ in runs:
        side = key(j); bw = j["env_MLX_OMARCHY_BATCH_WORK"] or "unset(5000)"
        for r in j["rows"]:
            rows.setdefault((bw, r["ctx"]), {}).setdefault(side, []).append(r)
    print("provenance:", sorted({j["provenance"] for j, _ in runs}))
    for (bw, ctx), a in sorted(rows.items()):
        (s0, v0), (s1, v1) = sorted(a.items())
        m0 = [x["decode_tps_median"] for x in v0]; m1 = [x["decode_tps_median"] for x in v1]
        print(f"{bw:11s} ctx {ctx:5d} | {s0} {statistics.mean(m0):6.2f} [{min(m0):.2f}-{max(m0):.2f}] | {s1} {statistics.mean(m1):6.2f} [{min(m1):.2f}-{max(m1):.2f}] | "
              f"{s1}/{s0} {statistics.mean(m1)/statistics.mean(m0):.3f} | ids {sorted({x['ids_sha'] for x in v0})}/{sorted({x['ids_sha'] for x in v1})} | "
              f"disp/token {sorted({x['vk_compute_dispatches_per_token'] for x in v0})} -> {sorted({x['vk_compute_dispatches_per_token'] for x in v1})}")
    # log-prob comparison, first process of each side, per budget
    by = {}
    for j, z in runs: by.setdefault((j["env_MLX_OMARCHY_BATCH_WORK"], key(j)), []).append(z)
    for bw in sorted({b for b, _ in by}, key=str):
        sides = sorted(s for b, s in by if b == bw)
        za, zb = by[(bw, sides[0])], by[(bw, sides[1])]
        for c in za[0].files:
            a = za[0][c].astype(np.float64); b = zb[0][c].astype(np.float64)
            kl = (np.exp(a) * (a - b)).sum(-1)
            same = {sides[0]: all(np.array_equal(z[c], za[0][c]) for z in za), sides[1]: all(np.array_equal(z[c], zb[0][c]) for z in zb)}
            print(f"logprobs bw={bw} {c}: {sides[1]} vs {sides[0]} bit-identical={np.array_equal(a, b)} max|d|={np.abs(a-b).max():.3g} KLmax={kl.max():.3g} top1-equal={int((a.argmax(-1)==b.argmax(-1)).sum())}/{a.shape[0]} within-side-identical={same}")
side_of = lambda j: "head" if "+95126c59f" in j["provenance"] else "base"
print("== per call, base wheel default vs head wheel default (3 processes each)")
percall(load("R2-B*-base.json"), load("R2-B*-head.json"), "base", "head")
print("\n== end to end, base wheel default vs head wheel default (3 processes each per budget)")
e2e(load("R2-E*.json"), side_of)
print("\n== CPU-disabled head wheel, same wheel, switch 0 vs unset")
percall(load("R2-N11-off.json"), load("R2-N12-on.json"), "off", "on")
e2e(load("R2-N2*.json"), lambda j: "on" if j["env_MLX_OMARCHY_SDPA_F16_NO_COOPMAT"] is None else "off")
