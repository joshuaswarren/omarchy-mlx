"""Cross-build teacher-forced log-probs: usage cmp_ab.py MAIN.npz HEAD.npz.
Per context: KL(main || head) mean and max over the forced steps, argmax agreement, largest |dlogprob| on main's top token."""
import json
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in a.files:
    x, y = a[k].astype(np.float64), b[k].astype(np.float64)
    kl = (np.exp(x) * (x - y)).sum(-1)
    top = x.argmax(-1)
    print(json.dumps({"ctx": k, "steps": len(top), "bit_identical": bool(np.array_equal(a[k], b[k])), "kl_mean": float(kl.mean()),
                      "kl_max": float(kl.max()), "argmax_equal": int((top == y.argmax(-1)).sum()),
                      "max_abs_dlogprob_top1": float(np.abs(x - y)[np.arange(len(top)), top].max())}))
