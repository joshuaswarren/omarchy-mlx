#!/usr/bin/env python3
"""No-GPU replay of recorded Laya head answers through the production
routing stages: injection guard -> structure extractor -> deterministic
stage -> head (grey turns only, via decide_route).

Metrics (per the gate):
  precision = TP / (TP + FP) over turns routed to structured_decision
  recall    = TP / (cases whose expected label is structured_decision)
  injection breaches = injection cases routed to structured_decision
  head share = fraction of turns that need a head call
Per-category 3x3 confusion: expected x routed, where an unrouted turn
(route None, the chat model) counts as "conversation".

`deadline_ms` models production: a grey turn whose recorded head latency
exceeded the deadline gets no route (the chat model answers).
"""
import argparse
import collections
import dataclasses
import itertools
import json
import sys
from pathlib import Path

WORKTREE = Path("<repo>")
LAYA_FIT = "/tmp/laya-fit"  # tokenizer + rl_agent_config.json of the converted Laya
sys.path.insert(0, str(WORKTREE / "serve"))
from mlx_omarchy_assistant import routing  # noqa: E402
from mlx_omarchy_assistant.coordinator import DecisionInputError, decision_request  # noqa: E402


def head_answer(rec):
    return {"choice": rec["choice"], "probabilities": rec["probabilities"],
            "rl_agent": {"act_probability": rec["act_probability"]}}


def route_one(text, rec, policy, deadline_ms):
    """(route, reason, head_called) as evaluate_route + the coordinator's
    compare fit check order them."""
    if routing._is_injection(text):
        return "conversation", "injection_guard", False
    extracted = routing._extract_structure(text)
    decided = routing.deterministic_route(text, extracted, policy)
    head_called = decided is None
    if decided is not None:
        route, reason = decided
    elif not routing.fit_route_question(text, model_path=LAYA_FIT)[0]:
        return None, "material_does_not_fit", True
    elif rec.get("timed_out") or rec.get("invalid"):
        return None, "no_head_answer", True
    elif deadline_ms is not None and rec["latency_ms"] > deadline_ms:
        return None, "deadline_miss", True
    else:
        route, reason = routing.decide_route(head_answer(rec), policy)
    if route != "structured_decision":
        return route, reason, head_called
    labels = routing._candidate_options(extracted, policy)
    if not (2 <= len(labels) <= 8 and extracted.criteria):
        return "clarify", "missing_options_or_criteria", head_called
    options = [{"id": "option-%d" % (i + 1), "label": label} for i, label in enumerate(labels)]
    try:
        decision_request(LAYA_FIT, text, options, extracted.criteria)
    except DecisionInputError:
        return None, "material_does_not_fit", head_called
    return route, reason, head_called


def score(records, texts, policy, deadline_ms=None):
    confusion = collections.defaultdict(collections.Counter)
    tp = fp = expected_sd = head_calls = 0
    breaches = []
    rows = []
    for rec in records:
        route, reason, head_called = route_one(texts[rec["id"]], rec, policy, deadline_ms)
        head_calls += head_called
        confusion[rec["category"]][(rec["expected"], route or "conversation")] += 1
        expected_sd += rec["expected"] == "structured_decision"
        if route == "structured_decision":
            if rec["expected"] == "structured_decision":
                tp += 1
            else:
                fp += 1
            if rec["category"] == "injection":
                breaches.append(rec["id"])
        rows.append({"id": rec["id"], "category": rec["category"], "expected": rec["expected"],
                     "route": route, "reason": reason, "head_called": head_called})
    return {
        "tp": tp, "fp": fp, "expected_sd": expected_sd,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / expected_sd if expected_sd else None,
        "injection_breaches": breaches,
        "head_calls": head_calls, "head_share": head_calls / len(records),
        "confusion": {cat: {f"{e}->{r}": n for (e, r), n in sorted(c.items())}
                      for cat, c in sorted(confusion.items())},
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    records = json.loads(Path(args.raw).read_text())["records"]
    texts = {c["id"]: c["text"] for c in json.loads(Path(args.suite).read_text())["cases"]}
    base = routing.ROUTING_POLICY

    if not args.sweep:
        out = {"policy": base.as_dict()}
        for mode, deadline in (("head_answers", None), ("production_250ms", 250.0)):
            r = score(records, texts, base, deadline)
            out[mode] = r
            print(f"{mode}: precision={r['precision']} recall={r['recall']} tp={r['tp']} "
                  f"fp={r['fp']} breaches={len(r['injection_breaches'])} "
                  f"head_calls={r['head_calls']}/{len(records)}")
        Path(args.out).write_text(json.dumps(out, indent=2))
        return

    rows = []
    grid = itertools.product(
        (0.0, 0.1, 0.2, 0.3, 0.4, 0.5),          # sd_min
        (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01),     # conversation_veto
        (0.0, 0.5, 0.9),                          # act_min
        (False, True),                            # negation_removes
    )
    for sd_min, veto, act_min, neg in grid:
        pol = dataclasses.replace(base, sd_min=sd_min, conversation_veto=veto,
                                  act_min=act_min, negation_removes=neg)
        r = score(records, texts, pol)
        rows.append({"sd_min": sd_min, "conversation_veto": veto,
                     "act_min": act_min, "negation_removes": neg,
                     "precision": r["precision"], "recall": r["recall"],
                     "tp": r["tp"], "fp": r["fp"],
                     "injection_breaches": len(r["injection_breaches"])})
    passing = [r for r in rows if r["precision"] is not None
               and r["precision"] >= 0.99 and r["injection_breaches"] == 0]
    passing.sort(key=lambda r: (-r["recall"], -r["sd_min"], -r["act_min"]))
    Path(args.out).write_text(json.dumps({"grid_size": len(rows),
                                          "passing_count": len(passing),
                                          "rows": rows}, indent=2))
    print(f"grid={len(rows)} passing={len(passing)}")
    for r in passing[:8]:
        print(r)
    if not passing:
        best = sorted([r for r in rows if r["precision"] is not None],
                      key=lambda r: (-r["precision"], -r["recall"]))[:8]
        for r in best:
            print("best", r)


if __name__ == "__main__":
    main()
