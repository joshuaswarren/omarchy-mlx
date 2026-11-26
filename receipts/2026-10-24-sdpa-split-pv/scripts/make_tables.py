"""make_tables.py : builds tables/*.tsv from run-logs/*.log (run from the receipt directory)."""
import glob
import os
import re
import statistics

CHIPS = {"apple-m2-max": "Apple M2 Max", "apple-m1-max": "Apple M1 Max", "apple-m1": "Apple M1"}


def header(path):
    boot = kernel = ""
    for line in open(path):
        m = re.search(r"kernel=(\S+)", line)
        if m and not kernel:
            kernel = m.group(1)
    return kernel


rows = ["chip\tkernel\tkeys\tfused_chain_us\tsplit_chain_us\tsplit_pv_chain_us\tsplit_pv_minus_split_us"]
for key, name in CHIPS.items():
    path = f"run-logs/microbench-{key}.log"
    data, arm = {}, None
    for line in open(path):
        m = re.match(r"SDPA-RUN arm (\w+)", line)
        if m:
            arm = m.group(1)
        m = re.match(r"SDPA kv (\d+): .* chain ([\d.]+) us per call", line)
        if m and arm:
            data.setdefault(int(m.group(1)), {})[arm] = float(m.group(2))
    kern = header(path)
    for kv in sorted(data):
        d = data[kv]
        rows.append(f"{name}\t{kern}\t{kv}\t{d['fused']:.1f}\t{d['split']:.1f}\t{d['splitpv']:.1f}\t{d['splitpv'] - d['split']:+.1f}")
os.makedirs("tables", exist_ok=True)
open("tables/microbench-chain-us.tsv", "w").write("\n".join(rows) + "\n")

rows = ["chip\tkernel\tprompt_tokens\told_pass1_tok_s\tsplit_pv_tok_s\tchange_percent\trounds\tdigest_equal\tdigest"]
for key, name in CHIPS.items():
    for pt in (477, 1888):
        path = f"run-logs/decode-{key}-{pt}.log"
        res = {}
        for line in open(path):
            m = re.search(r"round \d+ (\w+) rc=0 decode ([\d.]+) prefill ([\d.]+).* sha (\w+)", line)
            if m:
                res.setdefault(m.group(1), []).append((float(m.group(2)), m.group(4)))
        b = statistics.median(x[0] for x in res["base"])
        p = statistics.median(x[0] for x in res["pv"])
        digs = {x[1] for v in res.values() for x in v}
        rows.append(f"{name}\t{header(path)}\t{pt}\t{b:.2f}\t{p:.2f}\t{100 * (p / b - 1):+.1f}\t{len(res['base'])}\t{'yes' if len(digs) == 1 else 'no'}\t{sorted(digs)[0]}")
open("tables/decode-qwen3-4b.tsv", "w").write("\n".join(rows) + "\n")
