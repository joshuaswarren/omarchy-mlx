#!/usr/bin/env python3
"""CI guard: the GPU status table in docs/custom-kernel-inventory.md must
have exactly 26 rows and a counts line that matches the row tallies.

Guards against the 2026-10-08 drift class: overlapping status buckets,
kernels listed twice, and a counts line that no longer sums to the rows.

Checks:
  1. exactly 26 table rows of the form  | `name` | PASS|WRONG|COMPILE-FAIL|REFUSED | ...
  2. no duplicate kernel names
  3. a counts line **Counts: N PASS / N WRONG / N COMPILE-FAIL / N REFUSED = 26**
     whose four numbers equal the per-status row tallies and sum to 26

Usage: check_doc_table.py [doc_path]     (default: the repo's docs file)
Exit:  0 ok, 1 failed.
"""
import re
import sys
from collections import Counter
from pathlib import Path

EXPECTED = 26
ORDER = ("PASS", "WRONG", "COMPILE-FAIL", "REFUSED")
ROW = re.compile(r"^\| `(\w+)` \| (PASS|WRONG|COMPILE-FAIL|REFUSED) \|", re.MULTILINE)
COUNTS = re.compile(
    r"\*\*Counts: (\d+) PASS / (\d+) WRONG / (\d+) COMPILE-FAIL / (\d+) REFUSED = (\d+)\*\*")


def main() -> int:
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        path = (Path(__file__).resolve().parents[2]
                / "docs" / "custom-kernel-inventory.md")
    try:
        text = path.read_text()
    except OSError as error:
        print(f"FAIL: cannot read {path}: {error}")
        return 1

    failures = []
    rows = ROW.findall(text)
    if len(rows) != EXPECTED:
        failures.append(f"expected {EXPECTED} rows, got {len(rows)}")
    names = [name for name, _ in rows]
    dupes = sorted(name for name, count in Counter(names).items() if count > 1)
    if dupes:
        failures.append("duplicate kernel names: " + ", ".join(dupes))
    tallies = Counter(status for _, status in rows)

    match = COUNTS.search(text)
    if not match:
        failures.append("counts line '**Counts: ... = 26**' not found")
    else:
        claimed = [int(match.group(i)) for i in range(1, 5)]
        actual = [tallies[status] for status in ORDER]
        if claimed != actual:
            failures.append(f"counts line {claimed} != row tallies {actual}")
        if sum(claimed) != EXPECTED:
            failures.append(f"counts line sums to {sum(claimed)}, expected {EXPECTED}")

    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        return 1
    counts = ", ".join(f"{tallies[s]} {s}" for s in ORDER)
    print(f"OK: {EXPECTED} rows, {counts}, counts line matches")
    return 0


if __name__ == "__main__":
    sys.exit(main())
