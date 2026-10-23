"""Paired analysis of the 7600-token teacher-forced captures: analyze-long.py BASE_NPZ SPLIT_NPZ REF_NPZ. Mean negative log-likelihood per arm, paired differences with a 95 percent bootstrap interval over steps (seed 0, 2000 resamples; steps are correlated, so the interval is a floor), and the same per 950-step position bin."""
import sys

import numpy as np

b, s, r = (np.load(p) for p in sys.argv[1:4])
n = min(len(x["target_lp"]) for x in (b, s, r))
nb, ns, nr = (-x["target_lp"][:n].astype(np.float64) for x in (b, s, r))
rng = np.random.default_rng(0)


def boot(x, reps=2000):
    idx = rng.integers(0, len(x), (reps, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5])


print("steps", n, "mean NLL base %.5f split %.5f ref %.5f" % (nb.mean(), ns.mean(), nr.mean()))
for name, x in (("split - base", ns - nb), ("split - ref", ns - nr), ("base - ref", nb - nr)):
    lo, hi = boot(x)
    print("%s: mean %+.5f nats, 95%% interval [%+.5f, %+.5f]" % (name, x.mean(), lo, hi))
for lo in range(0, n, 950):
    hi = min(n, lo + 950)
    print("positions %d-%d: base-ref %+.5f split-ref %+.5f split-base %+.5f" % (lo, hi, (nb[lo:hi] - nr[lo:hi]).mean(), (ns[lo:hi] - nr[lo:hi]).mean(), (ns[lo:hi] - nb[lo:hi]).mean()))
