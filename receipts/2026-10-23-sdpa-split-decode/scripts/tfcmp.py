#!/usr/bin/env python3
"""Compare tfref.py outputs against a reference run.
usage: tfcmp.py REFDIR ARM=DIR [ARM=DIR ...]
Per arm against the reference: top-1 agreement, mean NLL and its change against the reference NLL,
mean |d log-prob| on the next token, mean |d log-prob| over the reference top 20, and every top-1 flip with the reference top-2 margin.
Then arm against arm: top-1 agreement and mean |d log-prob| on the next token."""
import glob
import os
import sys

import numpy as np


def load(d):
    return {int(os.path.basename(f)[3:-4]): np.load(f) for f in sorted(glob.glob(os.path.join(d, "seg*.npz")))}


ref = load(sys.argv[1])
arms = {}
for spec in sys.argv[2:]:
    name, d = spec.split("=", 1)
    arms[name] = load(d)
segs = sorted(set(ref).intersection(*[set(a) for a in arms.values()]))
steps = {s: min(len(ref[s]["target_lp"]), *[len(a[s]["target_lp"]) for a in arms.values()]) for s in segs}
total = sum(steps.values())
print(f"TFCMP segments {segs} steps per segment {steps} total {total}")
ref_nll = -np.concatenate([ref[s]["target_lp"][: steps[s]] for s in segs]).mean()
print(f"TFCMP reference mean NLL {ref_nll:.5f}")
print("arm | top1 agree | flips | mean NLL | NLL vs ref % | mean|dlp| next | mean|dlp| ref top20 | max flip ref gap")
for name, a in arms.items():
    agree = flips = 0
    tl, dl, d20, gaps = [], [], [], []
    for s in segs:
        n = steps[s]
        rt, at = ref[s]["top_i"][:n], a[s]["top_i"][:n]
        rv, av = ref[s]["top_v"][:n], a[s]["top_v"][:n]
        eq = rt[:, 0] == at[:, 0]
        agree += int(eq.sum())
        for j in np.nonzero(~eq)[0]:
            gaps.append(float(rv[j, 0] - rv[j, 1]))
        tl.append(a[s]["target_lp"][:n])
        dl.append(np.abs(a[s]["target_lp"][:n] - ref[s]["target_lp"][:n]))
        for j in range(n):
            amap = dict(zip(at[j].tolist(), av[j].tolist()))
            d20 += [abs(amap[t] - v) for t, v in zip(rt[j].tolist(), rv[j].tolist()) if t in amap]
    nll = -np.concatenate(tl).mean()
    flips = total - agree
    print(f"{name} | {agree}/{total} = {100 * agree / total:.2f}% | {flips} | {nll:.5f} | {100 * (nll / ref_nll - 1):+.4f} | "
          f"{np.concatenate(dl).mean():.5f} | {np.mean(d20):.5f} | {max(gaps) if gaps else 0.0:.4f}")
    if gaps:
        print(f"  {name} flip ref top1-top2 gaps: {sorted(round(g, 4) for g in gaps)}")
names = list(arms)
for i, x in enumerate(names):
    for y in names[i + 1 :]:
        same = sum(int((arms[x][s]["top_i"][: steps[s], 0] == arms[y][s]["top_i"][: steps[s], 0]).sum()) for s in segs)
        dd = np.concatenate([np.abs(arms[x][s]["target_lp"][: steps[s]] - arms[y][s]["target_lp"][: steps[s]]) for s in segs])
        print(f"TFCMP {x} vs {y}: top1 equal {same}/{total}, mean|dlp| next {dd.mean():.5f}, max {dd.max():.5f}, bit-equal steps {int((dd == 0).sum())}/{total}")
