"""mem_analyze.py <run-dir>: MemAvailable at each phase marker and the minimum inside each phase (MiB)."""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
samples = []
for line in (D / "memavail.txt").read_text().splitlines():
    t, kb, foreign = line.split()
    samples.append((float(t), int(kb) / 1024, int(foreign)))
marks = [(float(t), name) for t, name in (l.split() for l in (D / "marks.txt").read_text().splitlines())]
at = {name: t for t, name in marks}


def window(a, b):
    return [s for s in samples if at[a] <= s[0] <= at[b]]


def mean(a, b):
    w = window(a, b)
    return sum(s[1] for s in w) / len(w)


baseline = mean("baseline_start", "baseline_end")
alone_min = min(s[1] for s in window("alone_start", "alone_end"))
settled = mean("alone_end", "settle_end")
pair = mean("setup_end", "pair_settled")
with_rec_min = min(s[1] for s in window("pair_settled", "transcribes_end"))
with_rec = mean("transcribes_end", "recognizer_settled")
after = [s[1] for s in samples if s[0] >= at["after_stop"] - 1]
out = {
    "baseline_mib": round(baseline), "foreign_procs_max": max(s[2] for s in samples),
    "recognizer_alone_peak_drop_mib": round(baseline - alone_min),
    "after_alone_mib": round(settled),
    "pair_resident_drop_mib": round(settled - pair),
    "recognizer_with_pair_peak_drop_mib": round(pair - with_rec_min),
    "recognizer_with_pair_resident_drop_mib": round(pair - with_rec),
    "all_resident_peak_drop_from_baseline_mib": round(baseline - with_rec_min),
    "after_stop_mib": round(after[-1]) if after else None,
}
(D / "mem_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
