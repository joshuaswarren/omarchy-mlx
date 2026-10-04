#!/usr/bin/env python3
import json
import math
import statistics
import sys


def bf16_ulp(magnitude):
    if magnitude == 0:
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(abs(magnitude))) - 7)


def main():
    baseline = json.load(open(sys.argv[1], encoding="utf-8"))
    candidate = json.load(open(sys.argv[2], encoding="utf-8"))
    if baseline["prompt_sha256"] != candidate["prompt_sha256"]:
        raise ValueError("prompt sets differ")
    if len(baseline["prompts"]) != 10 or len(candidate["prompts"]) != 10:
        raise ValueError("expected ten prompts per arm")
    disagreements = []
    matched = total = 0
    logprob_deltas = []
    for base, cand in zip(baseline["prompts"], candidate["prompts"]):
        if base["prompt_index"] != cand["prompt_index"] or base["tokens"] != cand["tokens"]:
            raise ValueError("teacher-forced token prefixes differ")
        keys = ("predicted", "top2_gaps", "top2_magnitudes", "chosen_logprobs")
        if any(len(row[key]) != 512 for row in (base, cand) for key in keys):
            raise ValueError("each prompt must have exactly 512 measurements")
        for step, (teacher_token, pred_base, pred_cand) in enumerate(zip(base["tokens"], base["predicted"], cand["predicted"])):
            total += 1
            matched += pred_base == pred_cand
            logprob_deltas.append(abs(base["chosen_logprobs"][step] - cand["chosen_logprobs"][step]))
            if pred_base != pred_cand:
                gap = base["top2_gaps"][step]
                magnitude = base["top2_magnitudes"][step]
                ulp = bf16_ulp(magnitude)
                disagreements.append({"prompt_index": base["prompt_index"], "position": step,
                                      "teacher_token": teacher_token, "deployed_argmax": pred_base,
                                      "candidate_argmax": pred_cand, "deployed_gap": gap,
                                      "logit_magnitude": magnitude, "bf16_ulp": ulp,
                                      "gap_in_bf16_ulps": gap / ulp if ulp else math.inf,
                                      "within_tolerance": gap <= ulp or gap < 0.05})
    agreement = matched / total
    max_ulp_ratio = max((row["gap_in_bf16_ulps"] for row in disagreements), default=0.0)
    outside_gap = [row for row in disagreements if not row["within_tolerance"]]
    result = {"positions": total, "matched_top1": matched, "top1_agreement": agreement,
              "disagreements": len(disagreements), "disagreement_gap_bf16_ulp": {
                  "max": max_ulp_ratio,
                  "mean": statistics.mean(row["gap_in_bf16_ulps"] for row in disagreements) if disagreements else 0.0,
                  "all": disagreements},
              "disagreements_over_allowed_gap": len(outside_gap),
              "chosen_token_logprob_abs_delta_mean": statistics.mean(logprob_deltas),
              "chosen_token_logprob_abs_delta_max": max(logprob_deltas),
              "passed": agreement >= 0.95 and not outside_gap}
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
