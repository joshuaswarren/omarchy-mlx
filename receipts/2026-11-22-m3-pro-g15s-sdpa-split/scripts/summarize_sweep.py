"""Table of the variant sweep: usage summarize_sweep.py RUN_DIR SIDE [TAG].

Per head width and key count: independent and chain microseconds per call for every variant (the two runs, lower first), dispatches per call,
the largest |error| against the float64 reference over both runs, and which variants beat the default in both shapes in both runs.
"""
import glob
import json
import sys

d, side = sys.argv[1], sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else "SW"
runs = {}
for p in sorted(glob.glob(f"{d}/{tag}-{side}-*.json")):
    _, _, w, r, v = p.rsplit("/", 1)[1][:-5].split("-")
    runs.setdefault(w, {}).setdefault(v, {})[r] = {row["kv"]: row for row in json.load(open(p))["rows"]}
for w, by_v in sorted(runs.items()):
    vs = [v for v in ("default", "split", "pv", "bk64") if v in by_v]
    print(f"== width {'128 (32 q / 8 kv)' if w == 'a' else '256 (8 q / 2 kv)'}")
    print("kv     " + "  ".join(f"{v + ' ind':>15} {v + ' chain':>15}" for v in vs) + "  wins(both shapes, both runs)  max|err|")
    for kv in sorted(by_v["default"]["1"]):
        cells, wins, err = [], [], 0.0
        for v in vs:
            rr = [by_v[v][r][kv] for r in sorted(by_v[v])]
            ind = sorted(x["independent_us"] for x in rr)
            ch = sorted(x["chain_us"] for x in rr)
            err = max(err, *(x["max_abs_err_vs_f64"] for x in rr))
            cells.append(f"{ind[0]:7.1f}-{ind[-1]:<7.1f} {ch[0]:7.1f}-{ch[-1]:<7.1f}")
            if v != "default":
                dd = [by_v["default"][r][kv] for r in sorted(by_v["default"])]
                if max(x["independent_us"] for x in rr) < min(x["independent_us"] for x in dd) and \
                        max(x["chain_us"] for x in rr) < min(x["chain_us"] for x in dd):
                    wins.append(f"{v}({rr[0]['dispatches_per_call']:g}d)")
        print(f"{kv:<6} " + "  ".join(cells) + f"  {','.join(wins) or '-':<28}  {err:.3g}")
