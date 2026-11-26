"""gdn_numcmp.py RUNDIR_X RUNDIR_Y: compare two gdn_logits.py outputs (same prompt, same pinned greedy decode)."""
import sys
from pathlib import Path

import numpy as np


def load(d):
    z = np.load(Path(d) / "top.npz")
    return z["ids"], z["top_i"], z["top_v"], np.load(Path(d) / "full.npy")


def compare(dx, dy):
    ix, tix, tvx, fx = load(dx)
    iy, tiy, tvy, fy = load(dy)
    n = len(ix)
    same = ix == iy
    first = int(np.argmin(same)) if not same.all() else None
    m = n if first is None else first
    out = {"n": n, "ids_equal": bool(same.all()), "first_divergence": first}
    k = min(len(fx), m)
    d = np.abs(fx[:k].astype(np.float64) - fy[:k].astype(np.float64))
    out["full_steps_compared"] = k
    out["max_abs_dlogprob_full"] = float(d.max()) if k else None
    out["n_elements_differ_full"] = int((d > 0).sum()) if k else None
    px = np.exp(fx[:k].astype(np.float64))
    out["max_kl_x_to_y"] = float((px * (fx[:k].astype(np.float64) - fy[:k].astype(np.float64))).sum(axis=1).max()) if k else None
    top1 = tix[:m, 0] == tiy[:m, 0]
    out["top1_equal_steps"] = f"{int(top1.sum())}/{m}"
    top5 = sum(set(tix[s, :5]) == set(tiy[s, :5]) for s in range(m))
    out["top5_set_equal_steps"] = f"{top5}/{m}"
    dmax = 0.0
    for s in range(m):
        yx = dict(zip(tiy[s].tolist(), tvy[s].tolist()))
        for c in range(20):
            t = int(tix[s, c])
            if t in yx:
                dmax = max(dmax, abs(float(tvx[s, c]) - yx[t]))
    out["max_abs_dlogprob_top20_all_steps"] = dmax
    gap = tvx[:m, 0] - tvx[:m, 1]
    out["min_top1_top2_gap_x"] = float(gap.min())
    out["median_top1_top2_gap_x"] = float(np.median(gap))
    out["steps_with_gap_below_2x_maxdiff"] = int((gap < 2 * dmax).sum())
    return out


if __name__ == "__main__":
    for k, v in compare(sys.argv[1], sys.argv[2]).items():
        print(f"  {k}: {v}")
