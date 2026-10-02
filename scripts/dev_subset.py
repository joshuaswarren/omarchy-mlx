#!/usr/bin/env python3
"""Deterministic dev subset for lever iteration (never the held-out set).

One prompt per (category, kind) cell, first by id order, so lever runs stay
short while covering every failure class the gates score.
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
doc = json.load(open(src))
seen = {}
for prompt in doc["prompts"]:
    key = (prompt["category"], prompt.get("kind") or "")
    if key not in seen:
        seen[key] = prompt
subset = {"sha256": doc.get("sha256", ""), "note": "dev subset for lever iteration",
          "prompts": list(seen.values())}
json.dump(subset, open(dst, "w"), indent=2)
print(f"{len(subset['prompts'])} prompts -> {dst}")
