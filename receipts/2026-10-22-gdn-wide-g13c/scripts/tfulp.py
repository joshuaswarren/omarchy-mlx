#!/usr/bin/env python3
"""Classifies the top-1 flips of GPU arms against a reference run made with tfref_lse.py.
usage: tfulp.py REFDIR ARM=DIR [ARM=DIR ...]
The reference top logit is top_v[:, 0] + lse. One bf16 ULP at magnitude |x| is 2^(floor(log2|x|) - 7).
A flip counts as a near tie when the reference top-2 margin (the same in logit and log-prob units) is at most 2 ULP at the top logit, boundary inclusive."""
import glob
import math
import os
import sys

import numpy as np


def load(d):
    return {int(os.path.basename(f)[3:-4]): np.load(f) for f in sorted(glob.glob(os.path.join(d, "seg*.npz")))}


ref = load(sys.argv[1])
for spec in sys.argv[2:]:
    name, d = spec.split("=", 1)
    arm = load(d)
    flips = near = 0
    rows = []
    for s in sorted(set(ref) & set(arm)):
        n = min(len(ref[s]["target_lp"]), len(arm[s]["target_lp"]))
        for j in np.nonzero(ref[s]["top_i"][:n, 0] != arm[s]["top_i"][:n, 0])[0]:
            top = float(ref[s]["top_v"][j, 0] + ref[s]["lse"][j])
            gap = float(ref[s]["top_v"][j, 0] - ref[s]["top_v"][j, 1])
            ulp = 2.0 ** (math.floor(math.log2(abs(top))) - 7)
            ok = gap <= 2 * ulp
            flips += 1
            near += int(ok)
            rows.append((s, int(j), top, gap, ulp, ok))
    print(f"TFULP {name}: {flips} flips, {near} near ties (margin <= 2 ULP), {flips - near} beyond")
    for s, j, top, gap, ulp, ok in rows:
        print(f"  seg {s} step {j}: ref top logit {top:.3f} margin {gap:.4f} ulp {ulp:.4f} {'near tie' if ok else 'BEYOND 2 ULP'}")
