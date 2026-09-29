#!/usr/bin/env python3
"""Evaluate the held-out routing suite once.

Runs on the M2 under flock /tmp/m2-gpu.lock with a live Laya worker.
Single-shot: the suite is spent after this script.

Outputs a JSON receipt to stdout that downstream tooling can persist.
"""
import argparse
import collections
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

SUITE = Path("<repo>/tests/fixtures/routing_held_out.json")
FROZEN_SHA256 = "09a37b602336e308df6ece8ae9135d7771c9f9407826c89ab92b51b3198906f0"
LOG_DIR = Path("<notebook>/RoutingGate/routing-gate")
HOST = "127.0.0.1"
DECISION_URL = None  # filled by --decision-url
ROUTES = ("conversation", "structured_decision", "clarify")
POLICY = {
    "version": "1",
    "p_min": 0.55,
    "margin_min": 0.20,
    "act_min": 0.55,
    "question_text": (
        "Classify this user turn as exactly one of: conversation (the "
        "user wants an explanation or chat, no bounded choice), "
        "structured_decision (the user supplies explicit alternatives to "
        "pick from, even when phrased as a question), or clarify (the "
        "user's request is missing options, criteria, or scope and the "
        "assistant needs to ask before deciding)."
    ),
}


def build_payload(text):
    return {
        "state": text,
        "questions": {
            "route": {
                "type": "choice",
                "instructions": POLICY["question_text"],
                "criteria": {label: None for label in ROUTES},
            }
        },
    }


def call_decision(url, payload, timeout_seconds):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    deadline = time.monotonic() + timeout_seconds
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds + 5.0) as resp:
            raw = resp.read()
        elapsed = deadline - time.monotonic()
        return json.loads(raw), max(elapsed, 0.0), False
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        return {"error": str(exc), "timed_out": True}, 0.0, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-url", required=True)
    ap.add_argument("--deadline-seconds", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    assert hashlib.sha256(SUITE.read_bytes()).hexdigest() == FROZEN_SHA256, (
        "Held-out suite sha256 mismatch; aborting. Suite is single-use."
    )
    doc = json.loads(SUITE.read_text())
    cases = doc["cases"]
    print(f"Suite: {len(cases)} cases; version {doc['version']}; status {doc['status']}")

    per_category = collections.defaultdict(lambda: {"n": 0, "tp": 0, "fp": 0,
                                                    "fn": 0, "tn": 0, "decisions_in": 0,
                                                    "decisions_out": 0, "abstain": 0})
    routed_decision_total = 0
    routed_decision_correct = 0
    latencies = []
    skipped = collections.Counter()
    raw_records = []

    for c in cases:
        text = c["text"]
        expected = c["expected"]
        category = c["category"]
        payload = build_payload(text)
        start = time.monotonic()
        resp, _, timed_out = call_decision(args.decision_url, payload, args.deadline_seconds)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        ans = (resp.get("answers") or {}).get("route") if not timed_out and "error" not in resp else None
        if timed_out or not isinstance(ans, dict):
            skipped[category] += 1
            per_category[category]["n"] += 1
            raw_records.append({"id": c["id"], "category": category,
                                "expected": expected, "route": None,
                                "timed_out": timed_out, "error": resp.get("error"),
                                "latency_ms": elapsed_ms})
            continue

        probs = ans.get("probabilities") or {}
        choice = ans.get("choice")
        act = (ans.get("rl_agent") or {}).get("act_probability")
        sorted_probs = sorted(probs.values(), reverse=True)
        margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) >= 2 else 0.0
        p_selected = probs.get(choice, 0.0) if choice in probs else 0.0

        # Threshold logic from routing.ROUTING_POLICY
        if p_selected >= POLICY["p_min"] and margin >= POLICY["margin_min"] and act >= POLICY["act_min"]:
            routed = choice
        else:
            routed = "conversation"  # skip -> chat

        latencies.append(elapsed_ms)
        abstained = bool(act is not None and act < 0.5)
        per_category[category]["n"] += 1
        if routed == "structured_decision":
            per_category[category]["decisions_out"] += 1
            if expected == "structured_decision":
                per_category[category]["tp"] += 1
                routed_decision_correct += 1
            else:
                per_category[category]["fp"] += 1
        else:
            per_category[category]["decisions_in"] += 1
            if expected == "structured_decision":
                per_category[category]["fn"] += 1
            else:
                per_category[category]["tn"] += 1

        if expected == "structured_decision":
            routed_decision_total += 1

        if abstained:
            per_category[category]["abstain"] += 1

        raw_records.append({"id": c["id"], "category": category,
                            "expected": expected, "route": choice,
                            "routed": routed,
                            "p_selected": p_selected, "margin": margin,
                            "act_probability": act,
                            "probabilities": probs,
                            "abstained": abstained,
                            "latency_ms": elapsed_ms})

    precision = (routed_decision_correct / routed_decision_total
                 if routed_decision_total > 0 else None)
    coverage = (sum(per_category[k]["tp"] + per_category[k]["fn"]
                    for k in per_category if k == "decisions")
                and (per_category["decisions"]["tp"] + per_category["decisions"]["fn"])
                and per_category["decisions"]["tp"]
                / max(per_category["decisions"]["tp"] + per_category["decisions"]["fn"], 1))
    abstention_rate = (sum(per_category[k]["abstain"] for k in per_category)
                       / max(sum(per_category[k]["n"] for k in per_category), 1))

    sorted_lat = sorted(latencies)
    p50 = sorted_lat[len(sorted_lat) // 2] if sorted_lat else None
    p95 = sorted_lat[int(len(sorted_lat) * 0.95)] if sorted_lat else None

    summary = {
        "policy_version": POLICY["version"],
        "policy_thresholds": POLICY,
        "suite_sha256": FROZEN_SHA256,
        "decision_url": args.decision_url,
        "deadline_seconds": args.deadline_seconds,
        "cases": len(cases),
        "routed_decision_total": routed_decision_total,
        "routed_decision_correct": routed_decision_correct,
        "precision_on_routed_decision": precision,
        "coverage_of_decisions": coverage,
        "abstention_rate": abstention_rate,
        "latency_ms": {
            "p50": p50,
            "p95": p95,
            "max": max(latencies) if latencies else None,
            "n": len(latencies),
        },
        "per_category": {k: dict(v) for k, v in per_category.items()},
        "skipped_by_category": dict(skipped),
        "results": raw_records,
    }

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"precision={precision:.4f} coverage={coverage:.4f} abstention={abstention_rate:.4f}")
    print(f"latency p50={p50:.1f}ms p95={p95:.1f}ms (n={len(latencies)})")
    print(f"results written to {args.out}")


if __name__ == "__main__":
    main()