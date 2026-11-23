"""Default route of main against head per call at head layouts outside the calibration shapes: usage summarize_layout.py RUN_DIR.
LY1..LY4 = main head head main; median of the two processes per build."""
import glob, json, statistics, sys
rows = {}
for p in sorted(glob.glob(f"{sys.argv[1]}/LY*.json")):
    _, side, lay = p.rsplit("/", 1)[1][:-5].split("-")
    j = json.load(open(p))
    for r in j["rows"]:
        rows.setdefault((lay, j["head_dim"], j["q_heads"], j["kv_heads"]), {}).setdefault(r["kv"], {}).setdefault(side, []).append(r)
for (lay, d, hq, hkv), by_kv in rows.items():
    print(f"== width {d}, {hq} q / {hkv} kv: main -> head (independent us | chain us | dispatches)")
    for kv in sorted(by_kv):
        m, h = by_kv[kv]["main"], by_kv[kv]["head"]
        mi, hi = (statistics.median(x["independent_us"] for x in s) for s in (m, h))
        mc, hc = (statistics.median(x["chain_us"] for x in s) for s in (m, h))
        print(f"{kv:>6}: {mi:8.1f} -> {hi:7.1f} ({mi / hi:5.2f}x) | {mc:8.1f} -> {hc:7.1f} ({mc / hc:5.2f}x) | {m[0]['dispatches_per_call']:g} -> {h[0]['dispatches_per_call']:g}")
