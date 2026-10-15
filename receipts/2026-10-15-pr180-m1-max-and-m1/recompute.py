"""recompute.py: recompute the PR 180 pair figures from the row JSON files in this directory. Read-only, standard library only.

usage: python3 recompute.py

Each row JSON holds 10 timed decode samples and 5 prefill samples. The arm figure is the median of the row medians.
A pair is one O row and the P row of the same index (and the same part for the DeepSeek pair).
"""
import glob
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))


def load(pattern):
    rows = {}
    for path in sorted(glob.glob(os.path.join(HERE, pattern))):
        name = os.path.basename(path)[: -len(".json")]
        with open(path) as handle:
            rows[name] = json.load(handle)
    return rows


def report(title, rows, pair_key):
    print("==", title)
    arms = {"O": [], "P": []}
    for name, row in rows.items():
        arms[pair_key(name)[0]].append((pair_key(name)[1], row))
    digests = {row["decode_digest"][:16] for row in rows.values()}
    print("digests", sorted(digests), "prefill tokens", sorted({row["pf_tokens"] for row in rows.values()}))
    for metric, label in (("decode_tok_s", "decode"), ("prefill_tok_s", "prefill")):
        o = [row[metric]["median"] for _, row in sorted(arms["O"], key=lambda item: item[0])]
        p = [row[metric]["median"] for _, row in sorted(arms["P"], key=lambda item: item[0])]
        print(label, "O", round(st.median(o), 4), "P", round(st.median(p), 4), "P/O", round(st.median(p) / st.median(o), 6))
        print("  row medians O", o, "P", p)
        print("  pair ratios", [round(b / a, 5) for a, b in zip(o, p)])
        print("  spread O %.3f%% P %.3f%%" % ((max(o) - min(o)) / st.median(o) * 100, (max(p) - min(p)) / st.median(p) * 100))


report("gemma-4-26B-A4B, 16-inch M1 Max", load("rows-26b-m1-max/*.json"), lambda n: (n[0], int(n[1:])))
report(
    "DeepSeek-Coder-V2-Lite-Instruct 4-bit, 13-inch M1",
    load("rows-deepseek-m1/*.json"),
    lambda n: (n.split("-")[1][0], (n.split("-")[0], int(n.split("-")[1][1:]))),
)
