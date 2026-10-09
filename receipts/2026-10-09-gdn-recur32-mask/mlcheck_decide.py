"""Scores mlcheck-ticket.sh logs (rule: the docstring of mlcheck.py, fixed before the wheel was built). Arg: the log with lines
'@@ml <arm> MLCHECK {...}' in the order qgpr qrm qrm qgpr."""
import json
import re
import statistics
import sys

runs = {"qgpr": [], "qrm": []}
for line in open(sys.argv[1]):
    m = re.match(r"@@ml (\w+) MLCHECK (\{.*\})", line)
    if m:
        runs[m[1]].append(json.loads(m[2])["cases"])
if any(len(v) != 2 for v in runs.values()):
    print("DECISION: VOID (need two runs per arm)", {a: len(v) for a, v in runs.items()})
    sys.exit(1)
ok = True
for case in runs["qgpr"][0]:
    hashes = {a: {(r[case]["y"], r[case]["state"]) for r in v} for a, v in runs.items()}
    ctl = [r[case]["median_ms"] for r in runs["qgpr"]]
    chg = [r[case]["median_ms"] for r in runs["qrm"]]
    mc, mr = statistics.median(ctl), statistics.median(chg)
    tol = max(0.02, (max(ctl) - min(ctl)) / mc)
    same = len(hashes["qgpr"]) == 1 and hashes["qgpr"] == hashes["qrm"]
    rel = (mr - mc) / mc
    good = same and abs(rel) <= tol
    ok &= good
    print(f"{case}: bits {'identical' if same else 'DIFFER'} {hashes['qgpr'] == hashes['qrm']}  qgpr {ctl} ms  qrm {chg} ms  delta {rel * 100:+.1f}% tol {tol * 100:.1f}%  {'ok' if good else 'FAIL'}")
print("DECISION:", "PASS" if ok else "FAIL")
