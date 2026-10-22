#!/usr/bin/env python3
"""Paired comparison of two GPU arms against a reference on the steps all three share.
usage: tfpair.py REFDIR A=DIR B=DIR
Prints the mean |dlogprob| of the next token for each arm, the paired difference A - B with a 95 percent bootstrap interval
(10000 draws over steps, fixed seed), and the number of steps on which each arm is closer to the reference."""
import glob
import os
import sys

import numpy as np


def load(d):
    return {int(os.path.basename(f)[3:-4]): np.load(f) for f in sorted(glob.glob(os.path.join(d, "seg*.npz")))}


ref = load(sys.argv[1])
(na, da), (nb, db) = (s.split("=", 1) for s in sys.argv[2:4])
a, b = load(da), load(db)
segs = sorted(set(ref) & set(a) & set(b))
x, y = [], []
for s in segs:
    n = min(len(ref[s]["target_lp"]), len(a[s]["target_lp"]), len(b[s]["target_lp"]))
    x.append(np.abs(a[s]["target_lp"][:n] - ref[s]["target_lp"][:n]))
    y.append(np.abs(b[s]["target_lp"][:n] - ref[s]["target_lp"][:n]))
x, y = np.concatenate(x), np.concatenate(y)
d = x - y
rng = np.random.default_rng(0)
boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"TFPAIR steps {len(d)} mean|dlp| {na} {x.mean():.5f} {nb} {y.mean():.5f}; paired {na}-{nb} {d.mean():+.5f} 95% [{lo:+.5f}, {hi:+.5f}]; "
      f"{na} closer on {(x < y).sum()}, {nb} closer on {(y < x).sum()}, equal on {(x == y).sum()}")
