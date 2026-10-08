#!/usr/bin/env python3
"""CI guard: the docs table must have exactly 26 rows and matching counts."""
import re, sys
from pathlib import Path
doc = Path(__file__).resolve().parent.parent.parent / "docs" / "custom-kernel-inventory.md"
t = doc.read_text()
rows = re.findall(r"^\| `(\w+)` \| (PASS|WRONG|COMPILE-FAIL|REFUSED) \|", t, re.MULTILINE)
if len(rows) != 26:
    print(f"FAIL: expected 26 rows, got {len(rows)}"); sys.exit(1)
counts = {"PASS": 0, "WRONG": 0, "COMPILE-FAIL": 0, "REFUSED": 0}
for _, status in rows: counts[status] += 1
m = re.search(r"\*\*Counts: (\d+) PASS / (\d+) WRONG / (\d+) COMPILE-FAIL / (\d+) REFUSED = (\d+)\*\*", t)
if not m: print("FAIL: counts line not found"); sys.exit(1)
doc_c = [int(m.group(i)) for i in range(1, 5)]
actual = [counts["PASS"], counts["WRONG"], counts["COMPILE-FAIL"], counts["REFUSED"]]
if doc_c != actual: print(f"FAIL: {doc_c} != {actual}"); sys.exit(1)
if sum(doc_c) != 26: print(f"FAIL: sum {sum(doc_c)}"); sys.exit(1)
names = [r[0] for r in rows]
if len(names) != len(set(names)): print("FAIL: duplicates"); sys.exit(1)
print(f"OK: 26 rows, {actual}, no duplicates")
