"""docs/numerics-gate.md table from numerics_gate.py captures: usage gate_compare.py REF.npz OLD.npz NEW.npz [BANDS].

REF is the float32 reference, OLD the deployed default (main), NEW the candidate (head). For each arm against the reference: top-1
agreement (count and Wilson 95% interval), flips and how many are near ties (the reference's top-2 margin at most 2 bf16 ULP of its
top logit, inclusive; tools/dfuse/bf16_ulp.py and receipts/2026-11-01-m4-decode-defaults/numerics-gate), perplexity and |log(PPL / PPL_ref)|, mean |dlogprob| of the text token. The gate: NEW agreement at or above the
lower Wilson bound of OLD's agreement (near-tie flips counted as accepted; the raw exact-match counts are held to the same bound
separately), and NEW's PPL deviation no larger than OLD's (the captures are deterministic: the same build and text reproduce bit
for bit, so the run-to-run allowance is 0), with a paired block-bootstrap interval for the log-perplexity change. Also
OLD-vs-NEW top-1 agreement for information, and the same table per key band (BANDS: comma-separated band edges in keys).
"""
import json
import math
import sys

import numpy as np

ref, old, new = (np.load(p) for p in sys.argv[1:4])
edges = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else []


def ulp_bf16(x):
    x = np.maximum(np.abs(x), np.finfo(np.float32).tiny)
    return np.exp2(np.floor(np.log2(x)) - 7)


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def stats(a, m):
    n = int(m.sum())
    eq = (a["argmax"] == ref["argmax"]) & m
    flips = (~(a["argmax"] == ref["argmax"])) & m
    near = flips & ((ref["top1"] - ref["top2"]) <= 2 * ulp_bf16(ref["top1"]))
    ppl = math.exp(-float(a["target_lp"][m].mean()))
    ppl_ref = math.exp(-float(ref["target_lp"][m].mean()))
    lo, hi = wilson(int(eq.sum()), n)
    return {"n": n, "agree": int(eq.sum()), "agree_pct": round(100 * eq.sum() / n, 3), "wilson95": [round(100 * lo, 3), round(100 * hi, 3)],
            "flips": int(flips.sum()), "near_tie_flips": int(near.sum()), "accepted": int(eq.sum() + near.sum()),
            "ppl": round(ppl, 5), "ppl_ref": round(ppl_ref, 5), "ppl_dev": abs(math.log(ppl / ppl_ref)),
            "mean_abs_dlogprob": float(np.abs(a["target_lp"][m] - ref["target_lp"][m]).mean())}


def table(m, label):
    o, c = stats(old, m), stats(new, m)
    lo = wilson(o["accepted"], o["n"])[0] * 100
    agree_ok = bool(100 * c["accepted"] / c["n"] >= lo)
    lo_raw = wilson(o["agree"], o["n"])[0] * 100
    raw_ok = bool(c["agree_pct"] >= lo_raw)
    ppl_ok = bool(c["ppl_dev"] <= o["ppl_dev"])
    on = int(((old["argmax"] == new["argmax"]) & m).sum())
    # Change in log-perplexity, new - old. For the whole sample (a multiple of 128 positions) also a paired block bootstrap: blocks of 128
    # consecutive positions, 10,000 resamples, seed 0. Bands are not a multiple of 128, so they report the mean only.
    d = (old["target_lp"] - new["target_lp"])[m]
    dlog = {"mean": float(d.mean())}
    if label == "all":
        assert len(d) % 128 == 0, len(d)
        blocks = d.reshape(-1, 128).mean(1)
        bs = blocks[np.random.default_rng(0).integers(0, len(blocks), (10000, len(blocks)))].mean(1)
        dlog.update({"ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "blocks": len(blocks)})
    print(json.dumps({"band": label, "old": o, "new": c, "old_vs_new_agree": on, "dlogppl_new_minus_old": dlog,
                      "bit_identical": bool(np.array_equal(old["target_lp"][m], new["target_lp"][m])),
                      "gate_agreement": agree_ok, "gate_agreement_raw": raw_ok, "old_raw_wilson_lower_pct": round(lo_raw, 3), "gate_ppl": ppl_ok, "old_accepted_wilson_lower_pct": round(lo, 3)}))


# numerics_gate.py stored keys = PREFIX + 1 + arange(STEPS), but step i is scored from the logits produced before token i is fed:
# step 0 from the 128-token prefill, step i >= 1 from the decode call that attends PREFIX + i keys. The cache length of step i is PREFIX + i.
for a in (old, new):
    assert np.array_equal(a["target"], ref["target"]) and np.array_equal(a["keys"], ref["keys"]), "captures are not aligned"
keys = ref["keys"] - 1
table(np.ones_like(keys, bool), "all")
b = [int(keys.min())] + edges + [int(keys.max()) + 1]
for lo, hi in zip(b[:-1], b[1:]):
    table((keys >= lo) & (keys < hi), f"keys {lo}-{hi - 1}")
