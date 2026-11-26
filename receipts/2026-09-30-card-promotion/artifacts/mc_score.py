#!/usr/bin/env python3
"""Score HELD-OUT runs."""
import json, sys
for model in ["qwen3.8-2b-4bit", "qwen3.8-27b-4bit"]:
    path = f"<home>/agents/MarkdownCards/results/held_out_{model}.json"
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        print(f"{model}: no results")
        continue
    prompts = d.get("prompts", [])
    print(f"\n=== {model}: {len(prompts)}/24 prompts recorded ===")
    card_worthy = [p for p in prompts if p["category"] == "card-worthy"]
    plain = [p for p in prompts if p["category"] == "plain"]
    near_miss = [p for p in prompts if p["category"] == "near-miss"]
    valid = sum(1 for p in card_worthy if p["pass"])
    spurious_plain = sum(1 for p in plain if p["components"])
    spurious_nm = sum(1 for p in near_miss if p["components"])
    print(f"  card-worthy {len(card_worthy)}/12 (valid: {valid})")
    print(f"  plain       {len(plain)}/6 (spurious: {spurious_plain})")
    print(f"  near-miss   {len(near_miss)}/6 (spurious: {spurious_nm})")
    print(f"  threshold >= 10/12: {valid >= 10}")
    print(f"  threshold 0 spurious on plain/near-miss: {spurious_plain == 0 and spurious_nm == 0}")
    for p in prompts:
        print(f"    {p['id']} {p['category']:12s} {p['kind']:14s} expect={p['expect']:5s} components={p['components']} pass={p['pass']}")
