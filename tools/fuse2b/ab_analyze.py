#!/usr/bin/env python3
"""DispatchFuse A/B analyzer: paired deltas, ranges, greedy-token overlap
(numerics gate), digest flags from a run_ab results dir.
usage: ab_analyze.py <results-dir> [results-dir ...]
"""
import glob
import hashlib
import json
import re
import statistics
import sys

LABEL = re.compile(r"^(?:\w+-)?d(\d+)-r(\d+)-(ctl|on|neu)(\d\w?)$")


def digests(path):
    r = json.load(open(path))
    return r.get("ordered_records_sha256", "")


def tokens(path):
    r = json.load(open(path))
    return [rec.get("output_ids") or [] for rec in r.get("per_prompt", [])]


def main():
    rows = {}  # (depth, kind) -> list of (dir, label, rate, dig, toks)
    for d in sys.argv[1:]:
        for p in sorted(glob.glob(f"{d}/*.json")):
            lab = p.split("/")[-1][:-5]
            m = LABEL.match(lab)
            if not m:
                continue
            depth, rnd, kind, model = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)
            r = json.load(open(p))
            rate = r["decode_tok_rate"]["median"]
            rows.setdefault((model, depth, kind), []).append(
                (d, lab, rate, digests(p), tokens(p)))
    for (model, depth) in sorted({(k[0], k[1]) for k in rows}):
        ctl = rows.get((model, depth, "ctl"), [])
        on = rows.get((model, depth, "on"), [])
        neu = rows.get((model, depth, "neu"), [])
        if not ctl and not on:
            continue
        cr = [x[2] for x in ctl]
        orr = [x[2] for x in on]
        print(f"\n== {model} d{depth}: n_ctl={len(cr)} n_on={len(orr)}")
        deltas = []
        overlaps = []
        for i, (od, olab, orate, odig, otok) in enumerate(on):
            if i < len(ctl):
                _, clab, crate, cdig, ctok = ctl[i]
                deltas.append(orate - crate)
                same_dig = (odig == cdig)
                ov = 0.0
                if otok and ctok and len(otok) == len(ctok):
                    tot = same = 0
                    for a, b in zip(otok, ctok):
                        tot += min(len(a), len(b))
                        same += sum(1 for x, y in zip(a, b) if x == y)
                    ov = 100.0 * same / tot if tot else 100.0
                overlaps.append((ov, same_dig))
                print(f"  pair{i+1}: on={orate:.2f} ctl={crate:.2f} "
                      f"delta={100*(orate-crate)/crate:+.2f}% "
                      f"greedy_ident={ov:.2f}% digests_{'SAME' if same_dig else 'DIFF'}")
        if deltas:
            med = statistics.median(deltas)
            pct = statistics.median([100 * d / c for d, (_, c) in zip(deltas, [(0, ctl[i][2]) for i in range(len(deltas))])])
            print(f"  CTL range [{min(cr):.2f},{max(cr):.2f}] med={statistics.median(cr):.2f} | "
                  f"ON range [{min(orr):.2f},{max(orr):.2f}] med={statistics.median(orr):.2f} | "
                  f"paired delta med={med:+.3f} tok/s = {pct:+.2f}%")
            disjoint = min(orr) > max(cr)
            print(f"  ranges_disjoint={disjoint} "
                  f"greedy_min_ident={min(o[0] for o in overlaps):.2f}% "
                  f"all_digests_same={all(o[1] for o in overlaps)}")
        if neu:
            print(f"  NEU med={statistics.median([x[2] for x in neu]):.2f} "
                  f"n={len(neu)}")


if __name__ == "__main__":
    main()
