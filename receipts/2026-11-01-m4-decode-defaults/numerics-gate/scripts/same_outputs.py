#!/usr/bin/env python3
"""Release wheel vs profiling wheel per-op outputs: the profiling build's kernel list
describes the release run only if every output is bit-identical.
usage: same_outputs.py main.npz dmain.npz p1.npz dp1.npz"""
import sys

import numpy as np

pairs = [(sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])]
ok = True
for a, b in pairs:
    x, y = np.load(a), np.load(b)
    same = sum(np.array_equal(x[k], y[k]) for k in x.files)
    print(f"{a.rsplit('/', 1)[-1]} vs {b.rsplit('/', 1)[-1]}: {same}/{len(x.files)} outputs bit-identical")
    ok &= same == len(x.files) and set(x.files) == set(y.files)
print("release and profiling outputs identical" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
