"""Read the serve-path block of serveab.sh and print the result lines.

Usage: serveab7_analyze.py OUT_DIR ROUNDS REQS CSV LOG
Arms S (governor schedutil) and P (governor performance), same environment; ratio P/S. Each run holds REQS request pairs; a pair gives one decode rate from the wall times of a 32-token and a 128-token request.
A request counts when its JSON line has finish "length". Ratio D/O uses the arm medians over all counted requests; a spread of
either arm (largest over smallest per-run median) above 5 percent makes the block NOISY. The util_min table of each server's
threads and the shim line are quoted so the clamp state of every run can be read.
"""
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

out, rounds, reqs, csv_path, log = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


per_arm = {"S": [], "P": []}
run_med = {"S": [], "P": []}
digests = set()
ttft = {"S": [], "P": []}
bad = []
for arm in per_arm:
    for r in range(1, rounds + 1):
        got = []
        for q in range(1, reqs + 1):
            text = read(out / f"run-{arm}{r}.req{q}").strip()
            try:
                j = json.loads(text.splitlines()[-1])
            except (ValueError, IndexError):
                bad.append(f"{arm}{r}.req{q}: unreadable")
                continue
            if j.get("finish") != "length" or not j.get("decode_tps"):
                bad.append(f"{arm}{r}.req{q}: finish {j.get('finish')}")
                continue
            got.append(j["decode_tps"])
            digests.add(j["text_sha16"])
            if j.get("ttft_s"):
                ttft[arm].append(j["ttft_s"])
        if got:
            per_arm[arm].extend(got)
            run_med[arm].append(st.median(got))


def spread(v):
    return (max(v) / min(v) - 1) * 100 if v else float("nan")


med = {a: st.median(v) if v else float("nan") for a, v in per_arm.items()}
sp = {a: spread(run_med[a]) for a in run_med}
ratio = med["P"] / med["S"] if per_arm["P"] and per_arm["S"] else float("nan")
noisy = any(s > 5 for s in sp.values() if s == s)
text = read(log)
lines = []
for a, name in (("S", "S schedutil"), ("P", "P performance")):
    lines.append(f"{name} median {med[a]:.2f} tok/s over {len(per_arm[a])} requests, per-run medians {[round(x, 2) for x in run_med[a]]} spread {sp[a]:.1f}%"
                 + (f", ttft median {st.median(ttft[a]):.3f}s" if ttft[a] else ""))
print("SERVEAB7 RESULT decode: " + " | ".join(lines) + f" | P/S {ratio:.3f}" + (" NOISY(an arm spread above 5%: no verdict)" if noisy else ""))
verdict = "NO VERDICT: the digests differ" if len(digests) != 1 else ("NO VERDICT: an arm has no valid request" if ratio != ratio else ("NOISY: no verdict" if noisy else ("BUILD: P/S at or above 1.10" if ratio >= 1.10 else ("DOCS: P/S at or below 1.05" if ratio <= 1.05 else "BETWEEN 1.05 and 1.10: decide with the owner"))))
print(f"SERVEAB7 VERDICT by the rule registered before the run: {verdict}")
print("SERVEAB7 DIGESTS " + ("equal in all requests: " + next(iter(digests)) if len(digests) == 1 else f"DIFFER: {sorted(digests)}"))
if bad:
    print("SERVEAB7 BAD REQUESTS " + "; ".join(bad))
for m in re.finditer(r"round (\d+) arm ([SP]) (THREADS pid \d+ total \d+ util_min>=1024 \d+ util_min=0 \d+ unreadable \d+)", text):
    print(f"SERVEAB7 round {m.group(1)} arm {m.group(2)} {m.group(3)}")
for m in re.finditer(r"round (\d+) arm ([SP]) shim line: (.*)", text):
    print(f"SERVEAB7 round {m.group(1)} arm {m.group(2)} shim: {m.group(3).strip()}")

# P-cluster clock and power per arm over the windows of the runs (as clampab_analyze.py).
starts, wins = {}, {}
for line in text.splitlines():
    m = re.search(r"round (\d+) arm ([SP]) governors=\S* load1=\S+ port \d+ start_epoch=(\d+)", line)
    if m:
        starts[(m.group(2), m.group(1))] = int(m.group(3))
    m = re.search(r"round (\d+) arm ([SP]) end_epoch=(\d+)", line)
    if m and (m.group(2), m.group(1)) in starts:
        wins.setdefault(m.group(2), []).append((starts[(m.group(2), m.group(1))], int(m.group(3))))
try:
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
except OSError:
    rows = []
if rows:
    pols = sorted((c for c in rows[0] if c.startswith("khz_policy")), key=lambda c: int(c[len("khz_policy"):]))
    for arm, ws in sorted(wins.items()):
        p, hp = [], []
        for t0, t1 in ws:
            for r in rows:
                try:
                    t = float(r["epoch"])
                    if t0 <= t <= t1 and pols:
                        p.append(int(r[pols[-1]]) / 1000.0)
                        hp.append(float(r["heatpipe_uW"]) / 1e6)
                except (ValueError, KeyError):
                    pass
        if p:
            print(f"SERVEAB7 CLOCK arm {arm}: mean P-cluster {st.mean(p):.0f} MHz, heatpipe power {st.mean(hp):.1f} W over {len(p)} one-second samples (whole runs, load and warm-up included)")
