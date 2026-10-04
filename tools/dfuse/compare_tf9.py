#!/usr/bin/env python3
import json
import statistics
import sys

from bf16_ulp import bf16_ulp


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
        for step, (teacher_token, pred_base, pred_cand) in enumerate(
                zip(base["tokens"], base["predicted"], cand["predicted"])):
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
                                      "gap_in_bf16_ulps": gap / ulp,
                                      "within_one_bf16_ulp": gap <= ulp})
    agreement = matched / total
    ratios = [row["gap_in_bf16_ulps"] for row in disagreements]
    result = {"positions": total, "matched_top1": matched, "top1_agreement": agreement,
              "disagreements": len(disagreements), "disagreement_gap_bf16_ulp": {
                  "max": max(ratios, default=0.0),
                  "mean": statistics.mean(ratios) if ratios else 0.0,
                  "all": disagreements},
              "chosen_token_logprob_abs_delta_mean": statistics.mean(logprob_deltas),
              "chosen_token_logprob_abs_delta_max": max(logprob_deltas),
              "passed": agreement >= 0.99}
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
