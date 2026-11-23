"""A/B table: usage summarize_ab.py RUN_DIR [NOCPU_DIR]. Per model and context: decode tok/s of every process (main, head), median of the
process medians, change, prefill medians, dispatches per token and the token-id sha of each process."""
import glob, json, statistics, sys
d = sys.argv[1]
runs = {}
for p in sorted(glob.glob(f"{d}/AB*.json")):
    tag, side, model = p.rsplit("/", 1)[1][:-5].split("-", 2)
    for r in json.load(open(p))["rows"]:
        runs.setdefault((model, r["ctx"]), {}).setdefault(side, []).append((tag, r))
noc = {}
if len(sys.argv) > 2:
    for p in glob.glob(f"{sys.argv[2]}/N2-*.json"):
        j = json.load(open(p))
        for r in j["rows"]:
            noc[(j["model"], r["ctx"])] = r
for (model, ctx), sides in sorted(runs.items()):
    med = {s: statistics.median(r["decode_tps_median"] for _, r in v) for s, v in sides.items()}
    pf = {s: statistics.median(statistics.median(r["prefill_tps"]) for _, r in v) for s, v in sides.items()}
    line = (f"{model} ctx {ctx}: decode main {[r['decode_tps_median'] for _, r in sides['main']]} head {[r['decode_tps_median'] for _, r in sides['head']]}"
            f" -> {med['main']:.2f} -> {med['head']:.2f} ({(med['head'] / med['main'] - 1) * 100:+.1f}%); prefill {pf['main']:.1f} -> {pf['head']:.1f}"
            f" ({(pf['head'] / pf['main'] - 1) * 100:+.2f}%); dispatches/token {sides['main'][0][1]['vk_compute_dispatches_per_token']} -> {sides['head'][0][1]['vk_compute_dispatches_per_token']};"
            f" sha main {sorted({r['ids_sha'] for _, r in sides['main']})} head {sorted({r['ids_sha'] for _, r in sides['head']})}")
    if (model, ctx) in noc:
        n = noc[(model, ctx)]
        line += f"; nocpu {n['decode_tps_median']} sha {n['ids_sha']}"
    print(line)
