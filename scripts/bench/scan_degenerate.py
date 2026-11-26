#!/usr/bin/env python3
"""List replies that degenerate into a run of '!' (token id 0), a sign of non-finite logits."""
import json
import sys

for path in sys.argv[1:]:
    d = json.load(open(path))
    bad = []
    for phase in ("cards", "gsm", "ife"):
        for idx, r in d.get(phase, {}).items():
            text = r.get("text", "")
            if "!!!!!!!!!!!!!!!!" in text:
                bad.append(f"{phase}{idx}@{text.find('!!!!!!!!!!!!!!!!')}")
    total = sum(len(d.get(p, {})) for p in ("cards", "gsm", "ife"))
    print(f"{d['label']}: degenerate {len(bad)}/{total} {' '.join(bad)}")
