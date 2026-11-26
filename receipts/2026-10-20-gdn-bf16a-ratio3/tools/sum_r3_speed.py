import re
import statistics as st
import sys
from pathlib import Path

"""sum_r3_speed.py DIR: read run-speed*.log of verify-r3.sh and print per-prompt medians of O and N (tok/s), the change of the medians, wins and the per-round change."""
rows = {}
for f in sorted(Path(sys.argv[1]).glob("run-speed*.log")):
    for line in f.read_text().splitlines():
        m = re.search(r"time (P\d) round (\d) ([NO]) rc=0 decode ([\d.]+) tok/s.*verified=match", line)
        if m:
            rows[(m[1], int(m[2]), m[3])] = float(m[4])
print(f"rows with rc=0 and verified=match: {len(rows)} of 20")
for p in ("P1", "P2"):
    o = [rows[(p, i, "O")] for i in range(1, 6)]
    n = [rows[(p, i, "N")] for i in range(1, 6)]
    print(p, f"O median {st.median(o):.3f} N median {st.median(n):.3f} change {(st.median(n) / st.median(o) - 1) * 100:+.2f}%",
          f"N faster {sum(a > b for a, b in zip(n, o))}/5", "per round", " ".join(f"{(a / b - 1) * 100:+.2f}%" for a, b in zip(n, o)))
