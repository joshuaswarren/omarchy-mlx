#!/usr/bin/env python3
"""Score a labelled case set against a live Laya worker, resumably.

Design fix for the injection-fooled choice head: the user turn is
presented to the Laya choice head as inert quoted data (JSON-encoded
inside a delimiter block), with the classification instruction OUTSIDE
the quote. This stops the head from parsing `Options:` / `Criteria:`
syntax inside injection text as decision structure.

Per-case raw outputs are written atomically; a kill or reboot resumes
from the last completed id. `sync` is the orchestrator's job after the
phase ends.
"""
import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROUTES = ("conversation", "structured_decision", "clarify")

QUESTION_DEFAULT = (
    "Classify this user turn as exactly one of: conversation (the "
    "user wants an explanation or chat, no bounded choice), "
    "structured_decision (the user supplies explicit alternatives to "
    "pick from, even when phrased as a question), or clarify (the "
    "user's request is missing options, criteria, or scope and the "
    "assistant needs to ask before deciding)."
)

QUESTION_VERSION = "2"

# The deterministic injection guard runs IN FRONT of the head (in
# routing._is_injection). When it fires, the turn is diverted to
# conversation before any Laya call. The head only classifies benign
# text with the iter-1 wording above.

# Build payload is iter-1: send the raw user text as state.
def build_payload(text, question_text):
    return {
        "state": text,
        "questions": {
            "route": {
                "type": "choice",
                "instructions": question_text,
                "criteria": {label: None for label in ROUTES},
            }
        },
    }


def neutralise_quoted_text(text):
    """No-op kept for backward-compat with the iter-2 caller; the runner
    no longer uses the inert-quote design (iter-3 failure proved the
    head reads JSON-quoted text too)."""
    return text


def call_decision(url, payload, timeout_seconds):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds + 5.0) as resp:
            raw = resp.read()
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return json.loads(raw), elapsed_ms, False
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        return {"error": str(exc), "timed_out": True}, elapsed_ms, True


def load_existing(path):
    if not path.exists():
        return {"done_ids": set(), "records": [], "meta": {}}
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {"done_ids": set(), "records": [], "meta": {}}
    return {"done_ids": {r["id"] for r in doc.get("records", [])},
            "records": doc.get("records", []),
            "meta": doc.get("meta", {})}


def persist_atomic(path, doc):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)
    os.sync()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decision-url", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--question", default=QUESTION_DEFAULT)
    ap.add_argument("--deadline-seconds", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state = load_existing(out)
    records = list(state["records"])
    done_ids = set(state["done_ids"])

    suite_doc = json.loads(Path(args.suite).read_text())
    cases = suite_doc["cases"]
    todo = [c for c in cases if c["id"] not in done_ids]

    print(f"[dev-sweep] question_version={QUESTION_VERSION} suite={suite_doc.get('version')} "
          f"total={len(cases)} already_done={len(done_ids)} todo={len(todo)}",
          file=sys.stderr, flush=True)

    meta = dict(state["meta"])
    meta.update({
        "suite_version": suite_doc.get("version"),
        "suite_status": suite_doc.get("status"),
        "question_text": args.question,
        "question_version": QUESTION_VERSION,
        "deadline_seconds": args.deadline_seconds,
        "total": len(cases),
        "started_at": meta.get("started_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })

    persist_atomic(out, {"meta": meta, "records": records})

    skipped = collections.Counter()
    for i, c in enumerate(todo):
        payload = build_payload(c["text"], args.question)
        resp, ms, timed_out = call_decision(args.decision_url, payload, args.deadline_seconds)
        rec = {"id": c["id"], "category": c["category"], "expected": c["expected"],
               "latency_ms": ms}
        if timed_out:
            rec.update({"timed_out": True, "error": resp.get("error")})
            skipped[c["category"]] += 1
        else:
            ans = (resp.get("answers") or {}).get("route")
            if not isinstance(ans, dict):
                rec.update({"invalid": True, "resp": resp})
                skipped[c["category"]] += 1
            else:
                probs = ans.get("probabilities") or {}
                choice = ans.get("choice")
                act = (ans.get("rl_agent") or {}).get("act_probability")
                nums = [v for v in probs.values() if isinstance(v, (int, float))]
                sorted_nums = sorted(nums, reverse=True)
                margin = (sorted_nums[0] - sorted_nums[1]) if len(sorted_nums) >= 2 else 0.0
                p_selected = probs.get(choice, 0.0) if choice in probs else 0.0
                rec.update({
                    "choice": choice,
                    "probabilities": probs,
                    "p_selected": p_selected,
                    "margin": margin,
                    "act_probability": act,
                    "abstained": bool(act is not None and act < 0.5),
                })
        records.append(rec)
        meta["done"] = len(records)
        meta["skipped"] = dict(skipped)
        meta["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        persist_atomic(out, {"meta": meta, "records": records})
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"[dev-sweep] {len(records)}/{len(cases)} done; "
                  f"skipped={dict(skipped)} last={rec['id']}", file=sys.stderr, flush=True)

    print(f"[dev-sweep] COMPLETE total={len(cases)} skipped={dict(skipped)}",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()