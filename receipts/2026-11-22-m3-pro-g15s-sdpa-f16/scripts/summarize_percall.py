"""Summarize the per-call A/B: B0*-percall-{off,on}.json/.npz.

usage: summarize_percall.py RUN_DIR
Per shape: compute dispatches per call of each arm, median over processes of
the per-process median independent and chain microseconds, the ratio, the
largest |fused - composition| over every element and process pair (gate:
0.01, the require_close tolerance of test_sdpa_decode_fused.cpp), the
largest error of each arm against the float64 reference, and whether every
process of one arm stored the same bits.
"""
import glob
import json
import statistics
import sys

import numpy as np

d = sys.argv[1]
runs = {"off": [], "on": []}
for p in sorted(glob.glob(f"{d}/B*-percall-*.json")):
    j = json.load(open(p))
    arm = "on" if j["env_MLX_OMARCHY_SDPA_F16_NO_COOPMAT"] == "1" else "off"
    runs[arm].append((j, np.load(p[:-5] + ".npz")))
print("processes:", {k: len(v) for k, v in runs.items()})
print("provenance:", sorted({j["provenance"] for v in runs.values() for j, _ in v}))
print("MLX_OMARCHY_BATCH_WORK:", sorted({str(j["env_MLX_OMARCHY_BATCH_WORK"]) for v in runs.values() for j, _ in v}))


def f16(bits):
    return bits.astype(np.uint16).view(np.float16).astype(np.float64)


hdr = ("heads kv width keys | disp off/on | ind us off -> on (x) | chain us off -> on (x) | "
       "max|on-off| | err64 off / on | bits stable off/on")
print(hdr)
ok = True
rows0 = runs["off"][0][0]["rows"]
for i, r in enumerate(rows0):
    key = f"c{i}"
    ent = {}
    for arm in ("off", "on"):
        rs = [j["rows"][i] for j, _ in runs[arm]]
        ent[arm] = {
            "disp": sorted({x["vk_compute_dispatches"] for x in rs}),
            "ind": statistics.median(x["ind_us_median"] for x in rs),
            "chain": statistics.median(x["chain_us_median"] for x in rs),
            "err": max(x["max_abs_err_vs_f64"] for x in rs),
            "stable": all(np.array_equal(z[key], runs[arm][0][1][key]) for _, z in runs[arm]),
        }
    diff = max(float(np.abs(f16(a[key]) - f16(b[key])).max()) for _, a in runs["on"] for _, b in runs["off"])
    ok &= diff <= 0.01
    e0, e1 = ent["off"], ent["on"]
    print(f"{r['heads']:2d} {r['kv_heads']} {r['width']:3d} {r['keys']:4d} | {e0['disp']}/{e1['disp']} | "
          f"{e0['ind']:7.1f} -> {e1['ind']:6.1f} ({e0['ind'] / e1['ind']:4.2f}) | "
          f"{e0['chain']:7.1f} -> {e1['chain']:6.1f} ({e0['chain'] / e1['chain']:4.2f}) | "
          f"{diff:.2e} | {e0['err']:.2e} / {e1['err']:.2e} | {e0['stable']}/{e1['stable']}")
print("numerical gate |on-off| <= 0.01 on every element:", "PASS" if ok else "FAIL")
