"""Paired comparison of every 7600-token capture in numerics/apple-m2-max-long/captures.
usage: paired-controls.py CAPTURE_DIR
Prints, for each bf16 arm, the mean NLL and its gap to the float32 reference; every paired difference between two bf16 arms with a 95 percent bootstrap interval
over steps (seed 0, 2000 resamples; steps are correlated, so an interval is a floor); the same differences per position bin of 950 steps for split minus old route;
and the per-step difference between the two float32 references. Arms: base (old route), split (split kernel), blk, perm, rev; references: ref and ref32rev."""
import itertools
import sys

import numpy as np

cap = sys.argv[1]


def nll(name):
    return -np.load(f"{cap}/{name}/seg0.npz")["target_lp"].astype(np.float64)


arms = ["base", "split", "blk", "perm", "rev"]
data = {a: nll(a) for a in arms}
ref = nll("ref")
ref2 = nll("ref32rev")
n = min(len(x) for x in [*data.values(), ref, ref2])
data = {a: x[:n] for a, x in data.items()}
ref = ref[:n]
ref2 = ref2[:n]
rng = np.random.default_rng(0)
idx = rng.integers(0, n, (2000, n))


def boot(x):
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


print(f"steps {n}; reference NLL {ref.mean():.5f}; second reference NLL {ref2.mean():.5f}")
for a in arms:
    g = data[a] - ref
    g2 = data[a] - ref2
    print(f"{a}: NLL {data[a].mean():.5f} gap to reference {g.mean():+.5f} ({g.mean() / ref.mean() * 100:+.4f} percent), to second reference {g2.mean():+.5f} ({g2.mean() / ref2.mean() * 100:+.4f} percent)")
print("paired differences (second arm minus first arm):")
for a, b in itertools.combinations(arms, 2):
    d = data[b] - data[a]
    lo, hi = boot(d)
    print(f"{b} minus {a}: {d.mean():+.5f} [{lo:+.5f}, {hi:+.5f}]")
print("split minus base by position bin of 950 steps:")
d = data["split"] - data["base"]
for lo in range(0, n, 950):
    print(f"positions {lo}-{min(n, lo + 950)}: {d[lo:lo + 950].mean():+.5f}")
r = np.abs(ref - ref2)
print(f"float32 references, per-step difference of the next-token log-probability: maximum {r.max():.2e}, mean {r.mean():.2e}, bit-equal steps {int((r == 0).sum())} of {n}")
