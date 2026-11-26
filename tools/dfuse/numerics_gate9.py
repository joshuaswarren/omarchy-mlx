#!/usr/bin/env python3
"""Numerics gate for the 9B GDU raw route at d512: greedy token overlap +
first-divergence position between ctl (composed) and on (fused) runs.
usage: numerics_gate9.py <w9D-dir> [w9A-dir w9B-dir w9C-dir]
"""
import glob
import json
import sys

dirs = sys.argv[1:]
for d in dirs:
    ctl = sorted(glob.glob(f"{d}/*ctl*.json"))
    on = sorted(glob.glob(f"{d}/*-on*.json"))
    if not ctl or not on:
        continue
    r_c = json.load(open(ctl[0]))["per_prompt"][0]["output_ids"]
    r_o = json.load(open(on[0]))["on"] if False else json.load(open(on[0]))["per_prompt"][0]["output_ids"]
    same = sum(1 for a, b in zip(r_c, r_o) if a == b)
    first = next((i for i, (a, b) in enumerate(zip(r_c, r_o)) if a != b), None)
    ident = 100.0 * same / max(len(r_c), 1)
    print(f"{d.split('/')[-1]}: len={len(r_c)} identical={same} ({ident:.2f}%) "
          f"first_divergence={first}")
