"""Short before/after lines for the PR: usage pr_lines.py DF_A.json DF_B.json AB_4B.json AB_2B.json (one build)."""
import json, sys
dfa, dfb, ab4, ab2 = (json.load(open(p)) for p in sys.argv[1:5])
c = dfa["provenance"]["dist_version"].rsplit("+", 1)[1]
for df, lab, keys in ((dfa, "w128 32q/8kv", (448, 1024, 4096)), (dfb, "w256 8q/2kv", (1408, 2048, 4096))):
    rows = {r["kv"]: r for r in df["rows"]}
    print(f"{c} sdpa bf16 {lab} chain: " + " | ".join(f"keys={k} disp={rows[k]['dispatches_per_call']:g} {rows[k]['chain_us']}us" for k in keys))
for ab, lab in ((ab4, "Qwen3-4B"), (ab2, "Qwen3.8-2B")):
    print(f"{c} {lab} decode tok/s: " + " | ".join(f"ctx={r['ctx']} {r['decode_tps_median']:.2f}" for r in ab["rows"]))
