"""Summarize ab.py JSONL logs: per cell and arm, the median over processes
of each process's median, the per-process values, the B/A ratio, token-id
or argmax agreement, temperature range and peak memory.

It reads the JSON result lines of decode_bench.py (decode_median,
prompt_median, ids_sha) and prefill_bench.py (median_tok_s, last_argmax);
records from probes that print no JSON result (gemv_sweep.py, rope_bits.py,
sdpa_bits.py, bf16_linear.py, op_rate.py) are skipped, and a result with
other keys is reported, not guessed at.

usage: summarize.py <log.jsonl>...
"""
import json
import statistics
import sys
from collections import OrderedDict

cells = OrderedDict()
for path in sys.argv[1:]:
    for line in open(path):
        r = json.loads(line)
        if not r["result"]:
            continue
        key = " ".join(str(c) for c in r["cmd"]) + (f" [{r['env']}]" if r.get("env") else "")
        cells.setdefault(key, {}).setdefault(r["side"], []).append(r)

for key, sides in cells.items():
    print(f"## {key}")
    meds = {}
    for side in ("A", "B"):
        rs = sides.get(side, [])
        if not rs:
            continue
        res = [x["result"][0] for x in rs]
        if "decode_median" in res[0]:  # decode_bench.py
            vals = [x["decode_median"] for x in res]
            extra = f"prompt_med {statistics.median(x['prompt_median'] for x in res)} ids {sorted(set(x['ids_sha'] for x in res))}"
        elif "median_tok_s" in res[0]:  # prefill_bench.py
            vals = [x["median_tok_s"] for x in res]
            extra = f"argmax {sorted(set(x['last_argmax'] for x in res))}"
        else:
            print(f"  {side}: result keys {sorted(res[0])} come from no known probe; not summarized")
            continue
        mem = max(x["peak_gpu_gib"] or 0 for x in res)
        t = [v for x in rs for st in (x["before"], x["after"]) for v in st["temps_c"].values()]
        meds[side] = statistics.median(vals)
        print(f"  {side} {rs[0]['arm']}+{rs[0]['stamp']} n={len(vals)} median={meds[side]} values={vals} "
              f"{extra} peak_gpu_gib={mem} temps_c={min(t)}-{max(t)} load1_before<={max(x['before']['load1'] for x in rs)}")
    if len(meds) == 2:
        print(f"  B/A = {meds['B'] / meds['A']:.3f}")
