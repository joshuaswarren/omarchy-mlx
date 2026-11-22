"""Compare the teacher-forced decode log-softmax of the e2e runs.

usage: cmp_logits.py RUN_DIR
For each context length: within each arm, whether every process stored the
same log-probs (bit for bit); across arms (first off process against first
on process): largest |dlogprob| over the full vocabulary and all forced
steps, largest |dlogprob| on the forced (text) token, mean and max
KL(off || on) per step, and argmax agreement.
"""
import glob
import json
import sys

import numpy as np

d = sys.argv[1]
arms = {"off": [], "on": []}
for p in sorted(glob.glob(f"{d}/E*.json")):
    j = json.load(open(p))
    arm = "on" if j["env_MLX_OMARCHY_SDPA_F16_NO_COOPMAT"] == "1" else "off"
    arms[arm].append((p, np.load(p[:-5] + ".npz")))
print("processes:", {k: len(v) for k, v in arms.items()})
for key in arms["off"][0][1].files:
    a = arms["off"][0][1][key].astype(np.float64)
    b = arms["on"][0][1][key].astype(np.float64)
    same = {k: all(np.array_equal(z[key], v[0][1][key]) for _, z in v) for k, v in arms.items()}
    pa, pb = np.exp(a), np.exp(b)
    kl = (pa * (a - b)).sum(-1)
    print(json.dumps({
        "ctx": key, "steps": a.shape[0], "vocab": a.shape[1],
        "within_arm_bit_identical": same,
        "max_abs_dlogprob_full_vocab": float(np.abs(a - b).max()),
        "max_abs_dlogprob_top1_off": float(np.abs(a - b)[np.arange(a.shape[0]), a.argmax(-1)].max()),
        "kl_off_on_mean": float(kl.mean()), "kl_off_on_max": float(kl.max()),
        "argmax_equal_steps": int((a.argmax(-1) == b.argmax(-1)).sum()),
    }))
