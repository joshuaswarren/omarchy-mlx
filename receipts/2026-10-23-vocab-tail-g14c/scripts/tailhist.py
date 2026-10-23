#!/usr/bin/env python3
"""usage: tailhist.py ARM=DIR [ARM=DIR ...]   (captures of tfref-tail.py)
Per arm: the error of the bf16 LogSumExp (lseb) against the float64 LogSumExp of the same captured row (lse64), in units of one bf16 step of the result
(step of a value v in [2^k, 2^(k+1)) is 2^(k-7)), as a histogram of steps by error in half-steps: at most 0.5 half-steps (the correctly rounded value), at most 1, at most 1 half-step + 1e-5 absolute, above;
then the float32 LogSumExp error (lsef against lse64): mean and max. Prints counts only; the rule is read from them."""
import glob
import os
import sys

import numpy as np


def load(d):
    return [np.load(f) for f in sorted(glob.glob(os.path.join(d, "seg*.npz")))]


for spec in sys.argv[1:]:
    name, d = spec.split("=", 1)
    eb, step, ef = [], [], []
    for z in load(d):
        v = z["lse64"]
        eb.append(np.abs(z["lseb"].astype(np.float64) - v))
        step.append(2.0 ** (np.floor(np.log2(np.abs(v))) - 7))
        ef.append(np.abs(z["lsef"].astype(np.float64) - v))
    eb, step, ef = np.concatenate(eb), np.concatenate(step), np.concatenate(ef)
    half = 0.5 * step
    n = len(eb)
    within = int((eb <= half + 1e-5).sum())
    exact = int((eb <= 1e-9).sum())
    print(
        f"TAILHIST {name}: steps {n}; bf16 LogSumExp error against float64 at most half a bf16 step + 1e-5: {within}/{n}; "
        f"error 0: {exact}/{n}; error in half-steps mean {np.mean(eb / half):.3f} max {np.max(eb / half):.3f}; "
        f"float32 LogSumExp error mean {ef.mean():.3e} max {ef.max():.3e}"
    )
