#!/usr/bin/env python3
"""usage: tailchk.py REFDIR ARM=DIR [ARM=DIR ...]
Per arm against the reference capture of tfref-tail.py: steps, ArgMax index equal, ArgMin index equal, ArgMax index equal to the arm's own top-1 index (ties aside),
the largest |d log-prob| of the next token, and for the model-dtype row (amb, anb, lseb) the ArgMax and ArgMin index equality and the largest |d LogSumExp|. Exit 1 when any ArgMax or ArgMin index differs from the reference."""
import glob
import os
import sys

import numpy as np


def load(d):
    return {int(os.path.basename(f)[3:-4]): np.load(f) for f in sorted(glob.glob(os.path.join(d, "seg*.npz")))}


ref = load(sys.argv[1])
bad = 0
for spec in sys.argv[2:]:
    name, d = spec.split("=", 1)
    a = load(d)
    segs = sorted(set(ref) & set(a))
    steps = am = an = top = amb = anb = 0
    worst = worst_lse = 0.0
    bit = 0
    h_a = h_n = 0
    e_b = []
    e_f = []
    for s in segs:
        n = min(len(ref[s]["am"]), len(a[s]["am"]))
        steps += n
        am += int((ref[s]["am"][:n] == a[s]["am"][:n]).sum())
        an += int((ref[s]["an"][:n] == a[s]["an"][:n]).sum())
        top += int((a[s]["am"][:n] == a[s]["top_i"][:n, 0]).sum())
        amb += int((ref[s]["amb"][:n] == a[s]["amb"][:n]).sum())
        anb += int((ref[s]["anb"][:n] == a[s]["anb"][:n]).sum())
        dl = np.abs(ref[s]["lseb"][:n] - a[s]["lseb"][:n])
        worst_lse = max(worst_lse, float(dl.max()))
        bit += int((dl == 0).sum())
        h_a += int((a[s]["amb"][:n] == a[s]["am64"][:n]).sum())
        h_n += int((a[s]["anb"][:n] == a[s]["an64"][:n]).sum())
        e_b.append(np.abs(a[s]["lseb"][:n].astype(np.float64) - a[s]["lse64"][:n]))
        e_f.append(np.abs(a[s]["lsef"][:n].astype(np.float64) - a[s]["lse64"][:n]))
        worst = max(worst, float(np.abs(ref[s]["target_lp"][:n] - a[s]["target_lp"][:n]).max()))
    print(f"TAILCHK {name}: steps {steps}, argmax index equal {am}/{steps}, argmin index equal {an}/{steps}, argmax = own top-1 {top}/{steps}, max |d log-prob| {worst:.6f}; model-dtype row: argmax index equal {amb}/{steps}, argmin index equal {anb}/{steps}, LogSumExp bit-equal {bit}/{steps}, max |d LogSumExp| {worst_lse:.6f}")
    eb, ef = np.concatenate(e_b), np.concatenate(e_f)
    print(f"TAILCHK {name} against float64 on the captured row: bf16 LogSumExp error mean {eb.mean():.6f} max {eb.max():.6f}; float32 LogSumExp error mean {ef.mean():.3e} max {ef.max():.3e}; model-dtype row index equals the host float64 numpy index: argmax {h_a}/{steps}, argmin {h_n}/{steps}")
    bad += (am != steps) + (an != steps) + (amb != steps) + (anb != steps)
sys.exit(1 if bad else 0)
