"""Compact lines for the PR body from one per-call JSON and one e2e JSON of the same build.
usage: pr_lines.py PERCALL.json E2E.json"""
import json, sys
p, e = (json.load(open(x)) for x in sys.argv[1:3])
stamp = p["provenance"].split()[2].split("+")[-1]
rows = [r for r in p["rows"] if (r["heads"], r["width"]) == (16, 128) and r["keys"] in (512, 1023, 1024)]
print(f"{stamp} sdpa f16 16q/8kv w128: " + " | ".join(f"keys={r['keys']} disp={r['vk_compute_dispatches']} {r['ind_us_median']}us" for r in rows))
print(f"{stamp} decode tok/s: " + " | ".join(f"ctx={r['ctx']} {r['decode_tps_median']:.2f}" for r in e["rows"]) + "  ids " + " ".join(r["ids_sha"][:8] for r in e["rows"]))
