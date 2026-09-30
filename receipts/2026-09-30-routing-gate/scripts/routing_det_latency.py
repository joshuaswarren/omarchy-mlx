#!/usr/bin/env python3
"""Time the head-free routing stages (guard + extractor + deterministic
decision) over every turn of a suite, 20 passes, per-turn microseconds."""
import importlib.util
import json
import sys
import time
from pathlib import Path

routing_py, suite = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location("routing", routing_py)
routing = importlib.util.module_from_spec(spec)
sys.modules["routing"] = routing
spec.loader.exec_module(routing)

texts = [c["text"] for c in json.loads(Path(suite).read_text())["cases"]]
policy = routing.ROUTING_POLICY
samples = []
for _ in range(20):
    for text in texts:
        t0 = time.perf_counter()
        if not routing._is_injection(text):
            routing.deterministic_route(text, routing._extract_structure(text), policy)
        samples.append((time.perf_counter() - t0) * 1e6)
samples.sort()
print(json.dumps({"n": len(samples), "p50_us": samples[len(samples) // 2],
                  "p95_us": samples[int(len(samples) * 0.95) - 1], "max_us": samples[-1],
                  "loadavg": Path("/proc/loadavg").read_text().strip(),
                  "python": sys.version.split()[0]}))
