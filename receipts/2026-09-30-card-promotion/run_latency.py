#!/usr/bin/env python3
"""First-text latency probe on the 2B (warm turns).

Drives 30 ordinary chat turns through the live assistant on the M2 and
records the first-text delta wall-clock per turn.  Used to verify the
acceptance gate ``first-text p95 <= 2.0 s`` with the new markdown-promotion
default (no full schema on ordinary chat).

Usage:
    flock /tmp/m2-gpu.lock PYTHONPATH=serve python3 \\
        receipts/2026-09-30-card-promotion/run_latency.py --model qwen3.8-2b-4bit
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

sys.path.insert(0, os.path.join(REPO, "serve"))
from run_held_out import call, home_for, runtime_for, PAIR_FOR_MODEL, PYTHON  # noqa


def first_text_for(turn_url, runtime, turn, timeout_s=120):
    """Stream events for ``turn``; return wall-clock to first text chunk."""
    rt = json.load(open(runtime))
    base = f"http://127.0.0.1:{rt['port']}"
    headers = {"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
               "Origin": base}
    started = time.monotonic()
    deadline = started + timeout_s
    # Long-poll on the events endpoint, breaking when a text event lands.
    cursor = 0
    seen_first_text = False
    while time.monotonic() < deadline:
        url = f"{base}{turn_url}?after={cursor}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                body = json.loads(resp.read().decode())
        except Exception:
            time.sleep(0.1)
            continue
        for event in body.get("events") or []:
            cursor = event.get("seq", cursor) + 1
            if event.get("type") == "text":
                if not seen_first_text:
                    return time.monotonic() - started, event
                seen_first_text = True
            if event.get("type") == "done":
                return None, None
        time.sleep(0.05)
    return None, None


def run(model, n_turns=30, warmup=3):
    runtime = runtime_for(model)
    first_text_ms = []
    for i in range(-warmup, n_turns):
        cid = call("POST", "/api/conversations", {"save": False},
                   runtime=runtime)["id"]
        text = f"Tell me a one-line fact about number {i + 100}."
        started = time.monotonic()
        turn = call("POST", f"/api/conversations/{cid}/turns",
                    {"text": text, "mode": "chat", "max_tokens": 256},
                    runtime=runtime)["turn"]
        elapsed, event = first_text_for(
            f"/api/conversations/{cid}/events", runtime, turn)
        if i >= 0 and elapsed is not None:
            first_text_ms.append(round(elapsed * 1000, 1))
            print(f"turn {i + 1:2d}: {elapsed * 1000:.0f} ms",
                  flush=True)
    first_text_ms.sort()
    p50 = first_text_ms[int(0.50 * len(first_text_ms))]
    p95 = first_text_ms[int(0.95 * len(first_text_ms))]
    print(f"\nFIRST_TEXT_MS p50={p50} p95={p95} gate=2000")
    summary = {
        "model": model,
        "n": len(first_text_ms),
        "samples_ms": first_text_ms,
        "p50_ms": p50,
        "p95_ms": p95,
        "gate_ms": 2000,
        "gate_pass": p95 <= 2000,
    }
    out = os.path.join(HERE, f"latency_{model.replace('/', '_')}.json")
    with open(out, "w") as fp:
        json.dump(summary, fp, indent=2)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.8-2b-4bit")
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    run(args.model, args.n)


if __name__ == "__main__":
    main()
