#!/usr/bin/env python3
"""docs/numerics-gate.md checks 2 and 3 against the stock-MLX CPU fp32 reference.

  analyze_gate.py REF.json BASE_GLOB CAND_GLOB BF16_ULP_DIR OUT.json
Built on receipts/2026-10-28-qmm-small-t-g13s/analyze_gate.py, with the rule fixed there
and in docs/parity-contract.md section 4.2:
- near tie: a flip whose reference top-2 margin is at most 2 bf16 ULP at the reference top
  logit (tools/dfuse/bf16_ulp.py, inclusive); raw counts are kept;
- check 2: median candidate agreement (with near ties) >= lower bound of the two-sided 95%
  Wilson interval on the median baseline agreement (with near ties);
- check 3: median candidate deviation abs(log(PPL / PPL_ref)) <= median baseline deviation
  + noise, noise = max - min of the baseline deviations.
Also prints per-segment results and old-vs-new agreement (information only).
"""
import glob
import json
import math
import statistics
import sys

ref_path, base_glob, cand_glob, ulp_dir, out = sys.argv[1:6]
sys.path.insert(0, ulp_dir)
from bf16_ulp import bf16_ulp  # noqa: E402

ref = json.load(open(ref_path))


def flat(d, key):
    return [v for p in d["prompts"] for v in p[key]]


r_pred, r_gap, r_mag, r_lp = (flat(ref, k) for k in ("predicted", "top2_gaps", "top2_magnitudes", "chosen_logprobs"))
n = len(r_pred)
ppl_ref = math.exp(-sum(r_lp) / n)
seg = [len(p["predicted"]) for p in ref["prompts"]]


def score(path):
    d = json.load(open(path))
    assert d["prompt_sha256"] == ref["prompt_sha256"], path
    pred, lp = flat(d, "predicted"), flat(d, "chosen_logprobs")
    assert len(pred) == n
    flips = [i for i in range(n) if pred[i] != r_pred[i]]
    ties = [i for i in flips if r_gap[i] <= 2 * bf16_ulp(r_mag[i])]
    match = n - len(flips)
    ppl = math.exp(-sum(lp) / n)
    per_seg, i0 = [], 0
    for k, m in enumerate(seg):
        idx = range(i0, i0 + m)
        f = [i for i in flips if i in idx]
        per_seg.append({"segment": k, "agree": m - len(f), "flips": len(f),
                        "near_ties": sum(1 for i in f if i in ties),
                        "ppl": math.exp(-sum(lp[i] for i in idx) / m)})
        i0 += m
    return {"file": path.rsplit("/", 1)[-1], "commit": d.get("commit"), "chunk": d["chunk"], "env": d.get("env", {}),
            "agree": match, "agree_pct": 100 * match / n, "flips": len(flips), "near_ties": len(ties),
            "agree_with_ties": match + len(ties), "agree_with_ties_pct": 100 * (match + len(ties)) / n,
            "flip_margins_bf16_ulp": [round(r_gap[i] / bf16_ulp(r_mag[i]), 3) for i in flips],
            "ppl": ppl, "ppl_dev": abs(math.log(ppl / ppl_ref)),
            "mean_abs_dlogprob": statistics.mean(abs(a - b) for a, b in zip(lp, r_lp)),
            "per_segment": per_seg, "_pred": pred}


def wilson_low(k, total, z=1.96):
    p = k / total
    return (p + z * z / (2 * total) - z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / (1 + z * z / total)


# The first five runs of each arm in time order (file names carry time_ns); a resumed
# run can leave a sixth on one side, which is not read.
base_files, cand_files = sorted(glob.glob(base_glob))[:5], sorted(glob.glob(cand_glob))[:5]
assert len(base_files) == 5 and len(cand_files) == 5, (len(base_files), len(cand_files))
base = [score(p) for p in base_files]
cand = [score(p) for p in cand_files]
noise = max(r["ppl_dev"] for r in base) - min(r["ppl_dev"] for r in base)
band = 100 * wilson_low(statistics.median(r["agree_with_ties"] for r in base), n)
b_dev = statistics.median(r["ppl_dev"] for r in base)
c_dev = statistics.median(r["ppl_dev"] for r in cand)
c_agree = statistics.median(r["agree_with_ties_pct"] for r in cand)
old_new = sum(a == b for a, b in zip(base[0]["_pred"], cand[0]["_pred"]))
gate = {"positions": n, "reference_ppl": ppl_ref, "reference": {k: ref.get(k) for k in ("reference", "mlx", "mlx_cpu", "mlx_lm", "chunk")},
        "baseline_agree_with_ties_median_pct": statistics.median(r["agree_with_ties_pct"] for r in base),
        "baseline_agree_with_ties_wilson_low_pct": band,
        "candidate_agree_with_ties_median_pct": c_agree,
        "check2_token_choices": c_agree >= band,
        "baseline_ppl_dev_median": b_dev, "baseline_ppl_noise": noise, "candidate_ppl_dev_median": c_dev,
        "check3_perplexity": c_dev <= b_dev + noise,
        "old_vs_new_top1_same": old_new, "old_vs_new_top1_pct": 100 * old_new / n}
for r in base + cand:
    r.pop("_pred")
print(f"{'run':<44}{'agree %':>9}{'flips':>7}{'ties':>6}{'PPL':>11}{'dev':>11}{'|dlogp|':>10}")
for r in base + cand:
    print(f"{r['file'][:43]:<44}{r['agree_pct']:>9.3f}{r['flips']:>7}{r['near_ties']:>6}{r['ppl']:>11.5f}"
          f"{r['ppl_dev']:>11.6f}{r['mean_abs_dlogprob']:>10.5f}")
print(f"{'reference (stock MLX CPU fp32)':<44}{'':>9}{'':>7}{'':>6}{ppl_ref:>11.5f}")
print("per segment (first baseline run / first candidate run): agree, flips, near ties")
for sb, sc in zip(base[0]["per_segment"], cand[0]["per_segment"]):
    print(f"  segment {sb['segment']}: base {sb['agree']}/{sb['flips']}/{sb['near_ties']}  cand {sc['agree']}/{sc['flips']}/{sc['near_ties']}")
print(json.dumps(gate, indent=1))
json.dump({"gate": gate, "baseline_runs": base, "candidate_runs": cand}, open(out, "w"), indent=1)
