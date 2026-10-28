#!/usr/bin/env python3
"""docs/numerics-gate.md checks 2 and 3 for the short-pass route against the fp32 CPU reference.

  analyze_gate.py RESULTS_DIR OMARCHY_MLX_CHECKOUT
Reads ref.json, base-c4-r{1..5}.json, cand-c4-r{1..5}.json (and the c6 pair if present).
Near ties use tools/dfuse/bf16_ulp.py: a flip whose reference top-2 margin is at most 2 bf16
ULP at the reference top logit does not count. Agreement band: two-sided 95% Wilson interval on
the baseline; PPL noise: max - min of the five baseline deviations (parity-contract.md 4.2).
"""
import glob
import json
import math
import os
import statistics
import sys

res, repo = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(repo, "tools", "dfuse"))
from bf16_ulp import bf16_ulp  # noqa: E402

ref = json.load(open(os.path.join(res, "ref.json")))


def flat(d, key):
    return [v for p in d["prompts"] for v in p[key]]


r_pred, r_gap, r_mag, r_lp = (flat(ref, k) for k in ("predicted", "top2_gaps", "top2_magnitudes", "chosen_logprobs"))
n = len(r_pred)
ppl_ref = math.exp(-sum(r_lp) / n)


def score(path):
    d = json.load(open(path))
    assert d["prompt_sha256"] == ref["prompt_sha256"], path
    pred, lp = flat(d, "predicted"), flat(d, "chosen_logprobs")
    assert len(pred) == n
    match = sum(a == b for a, b in zip(pred, r_pred))
    flips = [i for i in range(n) if pred[i] != r_pred[i]]
    ties = sum(r_gap[i] <= 2 * bf16_ulp(r_mag[i]) for i in flips)
    ppl = math.exp(-sum(lp) / n)
    return {"arm": os.path.basename(path)[:-5], "small_t": d["small_t"], "chunk": d["chunk"],
            "agree": match, "agree_pct": 100 * match / n, "flips": len(flips), "near_ties": ties,
            "agree_with_ties_pct": 100 * (match + ties) / n, "ppl": ppl,
            "ppl_dev": abs(math.log(ppl / ppl_ref)),
            "mean_abs_dlogprob": statistics.mean(abs(a - b) for a, b in zip(lp, r_lp))}


def wilson_low(k, total, z=1.96):
    p = k / total
    return (p + z * z / (2 * total) - z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / (1 + z * z / total)


rows = [score(p) for p in sorted(glob.glob(os.path.join(res, "*-c[46]-r*.json")))]
base = [r for r in rows if r["arm"].startswith("base-c4")]
cand = [r for r in rows if r["arm"].startswith("cand-c4")]
noise = max(r["ppl_dev"] for r in base) - min(r["ppl_dev"] for r in base)
band = 100 * wilson_low(statistics.median(r["agree"] + r["near_ties"] for r in base), n)
gate = {
    "positions": n, "ppl_reference": ppl_ref,
    "baseline_agree_with_ties_wilson_low_pct": band,
    "candidate_agree_with_ties_median_pct": statistics.median(r["agree_with_ties_pct"] for r in cand),
    "check2_token_choices": statistics.median(r["agree_with_ties_pct"] for r in cand) >= band,
    "baseline_ppl_dev_median": statistics.median(r["ppl_dev"] for r in base),
    "baseline_ppl_noise": noise,
    "candidate_ppl_dev_median": statistics.median(r["ppl_dev"] for r in cand),
    "check3_perplexity": statistics.median(r["ppl_dev"] for r in cand)
    <= statistics.median(r["ppl_dev"] for r in base) + noise,
}
print(f"{'arm':<14}{'T max':>6}{'chunk':>6}{'agree %':>9}{'flips':>7}{'ties':>6}{'PPL':>10}{'|dlogp|':>10}")
for r in rows:
    print(f"{r['arm']:<14}{r['small_t'] or 'off':>6}{r['chunk']:>6}{r['agree_pct']:>9.3f}{r['flips']:>7}"
          f"{r['near_ties']:>6}{r['ppl']:>10.5f}{r['mean_abs_dlogprob']:>10.5f}")
print(f"{'ref (cpu fp32)':<14}{'':>6}{ref['chunk']:>6}{'':>9}{'':>7}{'':>6}{ppl_ref:>10.5f}")
print(json.dumps(gate, indent=1))
json.dump({"rows": rows, "gate": gate}, open(os.path.join(res, "gate-summary.json"), "w"), indent=1)
