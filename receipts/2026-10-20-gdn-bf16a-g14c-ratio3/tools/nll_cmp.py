"""nll_cmp.py A.npz[,A2.npz...] B.npz[,B2.npz...] [--label "A minus B"] [--bound 0.02]: paired comparison of two sets of nll_pair.py outputs on the same prefixes.
Several files per side (comma-separated) must cover contiguous windows; they are concatenated in window order (used when the window count is extended).

Rule (fixed in the pre-registration): d = nll_A - nll_B per position. The unit of replication is the PREFIX (positions inside a prefix share one
context): dbar_p = mean of d over prefix p; SE = std(dbar_p, ddof=1) / sqrt(P); interval = mean(dbar_p) +- 2 SE. ACCEPT iff the whole interval lies inside
(-bound, +bound). If it does not and the interval is wider than the bound, the verdict is WIDER (extend prefixes; the bound is not widened).
Also printed: the position-level SE (for comparison with the earlier H90/H91 numbers), the per-prefix mean differences, argmax disagreements
(count over common positions, first differing (prefix, step)), the top1 digests of both runs. Only prefixes present in both files are used.
"""
import argparse
import hashlib
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--label", default="A minus B")
ap.add_argument("--bound", type=float, default=0.02)
args = ap.parse_args()


def load(spec):
    parts = sorted((np.load(f) for f in spec.split(",")), key=lambda z: int(z["first"]))
    for left, right in zip(parts, parts[1:]):
        assert int(right["first"]) == int(left["first"]) + left["nll"].shape[0], "windows are not contiguous"
    return {
        "nll": np.concatenate([z["nll"] for z in parts]),
        "top1": np.concatenate([z["top1"] for z in parts]),
        "first": int(parts[0]["first"]),
    }


A = load(args.a)
B = load(args.b)
fa, fb = int(A["first"]), int(B["first"])
na, nb = A["nll"].shape[0], B["nll"].shape[0]
lo, hi = max(fa, fb), min(fa + na, fb + nb)
if hi <= lo:
    print("NO COMMON PREFIXES")
    sys.exit(2)
sa = slice(lo - fa, hi - fa)
sb = slice(lo - fb, hi - fb)
d = A["nll"][sa] - B["nll"][sb]
P, S = d.shape
pm = d.mean(axis=1)
mean = float(pm.mean())
se = float(pm.std(ddof=1) / np.sqrt(P)) if P > 1 else float("nan")
se_pos = float(d.reshape(-1).std(ddof=1) / np.sqrt(d.size))
lo_ci, hi_ci = mean - 2 * se, mean + 2 * se
inside = (-args.bound < lo_ci) and (hi_ci < args.bound)
wider = (hi_ci - lo_ci) > 2 * args.bound
ta, tb = A["top1"][sa], B["top1"][sb]
diff = ta != tb
nd = int(diff.sum())
first = None
if nd:
    idx = np.argwhere(diff)[0]
    first = (int(idx[0]) + lo, int(idx[1]))
print(f"{args.label}: prefixes {P} (windows {lo}..{hi - 1}), positions {d.size}")
print(f"  mean NLL A {A['nll'][sa].mean():.5f}  B {B['nll'][sb].mean():.5f}")
print(f"  paired mean {mean:+.5f}  cluster SE {se:.5f}  interval [{lo_ci:+.5f}, {hi_ci:+.5f}]  position-level SE {se_pos:.5f}")
print(f"  bound +-{args.bound}: {'ACCEPT (interval inside)' if inside else ('WIDER (interval wider than the bound: extend prefixes)' if wider else 'REJECT (interval reaches outside the bound)')}")
nz = int((d != 0).sum())
print(f"  VACUITY GUARD: positions with nll A != nll B {nz} of {d.size}; windows with any difference {int((d != 0).any(axis=1).sum())} of {P}"
      f"{'  (ZERO: the route switch changed nothing; the interval is VOID)' if nz == 0 else ''}")
print(f"  argmax disagreements {nd} of {diff.size} positions; first at (prefix, step) {first}")
print(f"  top1 digest A {hashlib.sha256(ta.tobytes()).hexdigest()[:16]}  B {hashlib.sha256(tb.tobytes()).hexdigest()[:16]}")
print("  per-prefix mean differences:", " ".join(f"{v:+.3f}" for v in pm))
