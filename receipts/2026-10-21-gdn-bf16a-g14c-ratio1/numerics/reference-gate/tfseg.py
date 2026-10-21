#!/usr/bin/env python3
"""Per-segment mean |dlogprob| of two arms against a reference, and a bootstrap of the paired difference over the 4 segments (all 16 resamples of 4 segments with replacement are enumerated, no sampling).
usage: tfseg.py REFDIR A=DIR B=DIR"""
import glob
import itertools
import os
import sys

import numpy as np


def load(d):
    return {int(os.path.basename(f)[3:-4]): np.load(f) for f in sorted(glob.glob(os.path.join(d, "seg*.npz")))}


ref = load(sys.argv[1])
(na, da), (nb, db) = (s.split("=", 1) for s in sys.argv[2:4])
a, b = load(da), load(db)
segs = sorted(set(ref) & set(a) & set(b))
diff = []
for s in segs:
    n = min(len(ref[s]["target_lp"]), len(a[s]["target_lp"]), len(b[s]["target_lp"]))
    x = np.abs(a[s]["target_lp"][:n] - ref[s]["target_lp"][:n]).mean()
    y = np.abs(b[s]["target_lp"][:n] - ref[s]["target_lp"][:n]).mean()
    diff.append(x - y)
    print(f"TFSEG segment {s}: {na} {x:.5f} {nb} {y:.5f} diff {x - y:+.5f}")
means = sorted(np.mean([diff[i] for i in idx]) for idx in itertools.product(range(len(diff)), repeat=len(diff)))
print(f"TFSEG paired {na}-{nb} over segments {np.mean(diff):+.5f}; enumerated resamples (n={len(means)}) 2.5th {means[int(0.025 * len(means))]:+.5f} 97.5th {means[int(0.975 * len(means)) - 1]:+.5f}")
