"""Default-route per-call table, main against head: usage summarize_df.py RUN_DIR. Median of the two processes per build (DF1-DF4)."""
import glob, json, statistics, sys
d = sys.argv[1]
for w, lab in (("a", "width 128, 32 q / 8 kv (Qwen3-4B)"), ("b", "width 256, 8 q / 2 kv (Qwen3.8-2B)")):
    rows = {}
    for p in sorted(glob.glob(f"{d}/DF*-{w}.json")):
        side = p.rsplit("/", 1)[1].split("-")[1]
        for r in json.load(open(p))["rows"]:
            rows.setdefault(r["kv"], {}).setdefault(side, []).append(r)
    print(f"== {lab}: main -> head, defaults, median of 2 processes (independent us | chain us | dispatches | max |err| vs f64)")
    for kv in sorted(rows):
        m, h = rows[kv]["main"], rows[kv]["head"]
        mi, hi = (statistics.median(x["independent_us"] for x in s) for s in (m, h))
        mc, hc = (statistics.median(x["chain_us"] for x in s) for s in (m, h))
        print(f"{kv:>6}: {mi:8.1f} -> {hi:7.1f} ({mi / hi:5.2f}x) | {mc:8.1f} -> {hc:7.1f} ({mc / hc:5.2f}x) | {m[0]['dispatches_per_call']:g} -> {h[0]['dispatches_per_call']:g}"
              f" | err {max(x['max_abs_err_vs_f64'] for x in m):.2g} -> {max(x['max_abs_err_vs_f64'] for x in h):.2g}")
