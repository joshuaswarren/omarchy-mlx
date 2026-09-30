#!/usr/bin/env python3
"""Per-model detail for the bench receipt: card outcomes and cap hits by phase."""
import json
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    print(f"## {d['label']}")
    cards = d.get("cards", {})
    for idx in sorted(cards, key=int):
        c = cards[idx]
        extra = c.get("types") or c.get("error") or c.get("promotable") or ""
        print(f"card {idx:>2} {c['expect']:5} {c['category']:15} full={int(c['full_schema'])} "
              f"{c['outcome']:12} {extra!s:40.40} tokens={c['tokens']} cap={int(c['truncated'])}")
    for phase in ("gsm", "ife"):
        part = d.get(phase, {})
        if part:
            caps = [k for k, r in part.items() if r["truncated"]]
            fails = [k for k, r in part.items() if not r["pass"]]
            print(f"{phase}: pass {len(part) - len(fails)}/{len(part)} fails={fails} cap_hits={caps}")
