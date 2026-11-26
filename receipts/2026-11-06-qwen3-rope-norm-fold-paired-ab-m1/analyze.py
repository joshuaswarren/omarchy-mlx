#!/usr/bin/env python3
"""Per-pair and pooled ON/OFF ratios from pairs.tsv (stdlib only).

Each pair is one run of 3 rounds per arm; the pair value is median(ON) / median(OFF).
Usage: python3 analyze.py [pairs.tsv]     (default: the pairs.tsv beside this file)
"""
import csv
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def ratios(rows, col):
    """{pair: (order, median ON / median OFF)} for one tok/s column."""
    by = {}
    for r in rows:
        by.setdefault((int(r["pair"]), r["order"]), {"on": [], "off": []})[r["arm"]].append(float(r[col]))
    return {p: (o, st.median(a["on"]) / st.median(a["off"])) for (p, o), a in sorted(by.items())}


def main(path):
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    out = {}
    for col in ("decode_tok_s", "prefill_tok_s"):
        r = ratios(rows, col)
        v = [x for _, x in r.values()]
        out[col] = v
        print(f"{col}: pooled median {st.median(v):.5f}  min {min(v):.5f}  max {max(v):.5f}  n {len(v)}")
        for order in ("on-first", "off-first"):
            o = [x for oo, x in r.values() if oo == order]
            print(f"  {order}: median {st.median(o):.5f}  n {len(o)}")
        if col == "decode_tok_s":
            for p, (o, x) in r.items():
                print(f"  pair {p} ({o}): {x:.5f}")
    return out


def selftest():
    rows = [{"pair": "1", "order": "on-first", "arm": a, "round": str(i), "decode_tok_s": str(v), "prefill_tok_s": "1"}
            for a, vs in (("on", (101, 102, 103)), ("off", (100, 100, 100))) for i, v in enumerate(vs)]
    assert abs(ratios(rows, "decode_tok_s")[1][1] - 1.02) < 1e-12


if __name__ == "__main__":
    selftest()
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pairs.tsv"))
