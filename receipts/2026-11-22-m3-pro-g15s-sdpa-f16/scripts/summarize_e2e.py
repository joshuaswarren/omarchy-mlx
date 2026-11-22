"""Summarize E*.json: per budget, per context, decode tok/s off vs on (mean of process medians, min-max), ids, dispatches/token."""
import glob, json, statistics, sys
d = sys.argv[1]
rows = {}
prov = set()
for p in sorted(glob.glob(f"{d}/E*.json")):
    j = json.load(open(p)); prov.add(j["provenance"])
    arm = "on" if j["env_MLX_OMARCHY_SDPA_F16_NO_COOPMAT"] == "1" else "off"
    bw = j["env_MLX_OMARCHY_BATCH_WORK"] or "unset(5000)"
    for r in j["rows"]:
        rows.setdefault((bw, r["ctx"]), {}).setdefault(arm, []).append(r)
print("provenance:", sorted(prov))
print("budget ctx | off tok/s mean [min-max] | on tok/s mean [min-max] | on/off | ids off/on | dispatches/token off -> on")
for (bw, ctx), a in sorted(rows.items()):
    m = {k: [x["decode_tps_median"] for x in v] for k, v in a.items()}
    ids = {k: sorted({x["ids_sha"] for x in v}) for k, v in a.items()}
    dp = {k: sorted({x["vk_compute_dispatches_per_token"] for x in v}) for k, v in a.items()}
    mo, mn = statistics.mean(m["off"]), statistics.mean(m["on"])
    print(f"{bw:11s} {ctx:5d} | {mo:6.2f} [{min(m['off']):.2f}-{max(m['off']):.2f}] | {mn:6.2f} [{min(m['on']):.2f}-{max(m['on']):.2f}] | "
          f"{mn/mo:5.3f} (+{(mn/mo-1)*100:.1f}%) | {ids['off']}/{ids['on']} | {dp['off']} -> {dp['on']}")
