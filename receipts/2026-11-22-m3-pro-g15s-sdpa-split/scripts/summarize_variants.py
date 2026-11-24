"""Tables of the forced-variant decode (variant_decode.py) and float32 reference parity (parity_ref.py) runs: usage summarize_variants.py VARIANTS_DIR PARITY_DIR."""
import glob, json, statistics, sys
print("== forced variant against the default, same process (decode tok/s medians, change, tokens, teacher-forced KL(default || variant), argmax agreement of 32 steps)")
for p in sorted(glob.glob(f"{sys.argv[1]}/VD-*.json")):
    j = json.load(open(p)); v = j["variant"]
    for r in j["rows"]:
        print(f"{j['model']} {v} ctx {r['ctx']}: {statistics.median(r['default']['tps']):.2f} -> {statistics.median(r[v]['tps']):.2f} ({r['speedup_pct']:+.1f}%);"
              f" prefill {statistics.median(r['default']['prefill_tps']):.1f} -> {statistics.median(r[v]['prefill_tps']):.1f}; dispatches/token {r['default']['dispatches_per_token']} -> {r[v]['dispatches_per_token']};"
              f" tokens {'identical' if r['tokens_identical'] else 'differ from token %d' % r['first_differing_token']}; KL mean {r['kl_mean']:.2e} max {r['kl_max']:.2e}; argmax {r['argmax_equal_steps']}/32")
print("== against the float32 model (KL(float32 || arm) mean/max, argmax agreement of 32 steps)")
for p in sorted(glob.glob(f"{sys.argv[2]}/PR-*.json")):
    j = json.load(open(p))
    for r in j["rows"]:
        print(f"{j['model']} ctx {r['ctx']}: " + "; ".join(f"{a} {x['kl_mean']:.2e}/{x['kl_max']:.2e} {x['argmax_equal']}/32" for a, x in r["vs_float32"].items()))
