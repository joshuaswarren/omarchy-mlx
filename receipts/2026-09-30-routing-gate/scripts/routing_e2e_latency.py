#!/usr/bin/env python3
"""Per-turn latency of the head-free routing path as the product runs it:
injection guard + extractor + deterministic decision, and for decision
routes the coordinator's comparison fit check (decision_request).
Usage: routing_e2e_latency.py <laya_model_dir> <suite.json> <passes>"""
import json
import sys
import time
from pathlib import Path

from mlx_omarchy_assistant import routing
from mlx_omarchy_assistant.coordinator import DecisionInputError, decision_request

model_dir, suite, passes = sys.argv[1], sys.argv[2], int(sys.argv[3])
texts = [c["text"] for c in json.loads(Path(suite).read_text())["cases"]]
policy = routing.ROUTING_POLICY
by_route = {}
for _ in range(passes):
    for text in texts:
        t0 = time.perf_counter()
        route = "conversation"
        if not routing._is_injection(text):
            extracted = routing._extract_structure(text)
            decided = routing.deterministic_route(text, extracted, policy)
            route = decided[0] if decided else "head"
            if route == "structured_decision":
                labels = routing._candidate_options(extracted, policy)
                options = [{"id": "option-%d" % (i + 1), "label": l} for i, l in enumerate(labels)]
                try:
                    decision_request(model_dir, text, options, extracted.criteria)
                except DecisionInputError:
                    route = "does_not_fit"
        by_route.setdefault(route or "chat", []).append((time.perf_counter() - t0) * 1e3)
out = {"loadavg": Path("/proc/loadavg").read_text().strip(), "python": sys.version.split()[0]}
everything = sorted(ms for v in by_route.values() for ms in v)
for name, v in list(by_route.items()) + [("all_turns", everything)]:
    v = sorted(v)
    out[name] = {"n": len(v), "p50_ms": v[len(v) // 2], "p95_ms": v[int(len(v) * 0.95) - 1],
                 "max_ms": v[-1]}
print(json.dumps(out, indent=1))
