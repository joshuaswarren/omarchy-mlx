"""Read the block of flip_ab.sh and print the result lines.

Usage: flip_analyze.py OUT_DIR ROUNDS REQS CSV LOG

Arms: A bench_decode (thread clamp), G mlx_lm.server generator in-process, GM / GL / GS = G with ONE candidate cost removed
(GM no all-True mask array for the sliding batch cache, GL no per-token logprob read, GS new thread-local stream).
Rate: ts_tps for G-type arms (consumer-side token stamps), bench_decode's own tok/s for A; m = 1000 / tok/s is ms per token.

REGISTERED RULE (fixed before the run, never changed after seeing numbers):
  gap = m(G) - m(A). The gap is REAL when it is at least 3 percent of m(A) and every G run median is below every A run median.
  For X in GM, GL, GS: gain = m(G) - m(GX); share = gain / gap. X is REAL when gain is at least 3 percent of m(A) and every GX run median
  is above every G run median (every run faster). X is the CAUSE when it is REAL and share is at least 0.5; PART when REAL and share
  is below 0.5; NOTHING when not REAL. Several candidates can be PART or CAUSE (their gains need not add up).
  NO VERDICT for X when its ids digests differ from arm G's, or when an arm has no valid sample; NOISY for X when the spread of run
  medians (largest over smallest) of G or of GX is above 5 percent. When the gap is not REAL, the whole block says GAP NOT REPRODUCED
  and no candidate is judged. The ids digests of A and G must be equal, else the whole block is NO VERDICT.
"""
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

out, rounds, reqs, csv_path, log = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
FLIPS = ("GM", "GL", "GS")
GARMS = ("G",) + FLIPS


def read(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def jlines(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                yield json.loads(line)
            except ValueError:
                pass


ts = {a: [] for a in ("A",) + GARMS}
runs = {a: [] for a in ("A",) + GARMS}
ids = {a: set() for a in ("A",) + GARMS}
bad = []
for r in range(1, rounds + 1):
    j = [x for x in jlines(read(out / f"run-A{r}.out")) if x.get("engine") == "bench_decode"]
    if j and j[-1].get("decode_tps"):
        ts["A"].append(j[-1]["decode_tps"])
        runs["A"].append(j[-1]["decode_tps"])
        ids["A"].add(j[-1]["ids_sha256_16"])
    else:
        bad.append(f"A{r}: no bench_decode result")
    for a in GARMS:
        got = []
        for x in jlines(read(out / f"run-{a}{r}.out")):
            if x.get("arm") != a or x.get("finish") != "length" or not x.get("ts_tps"):
                bad.append(f"{a}{r}: {str(x)[:120]}")
                continue
            got.append(x["ts_tps"])
            ids[a].add(x["ids_sha16"])
        ts[a].extend(got)
        if got:
            runs[a].append(st.median(got))


def med(v):
    return st.median(v) if v else float("nan")


def spread(v):
    return (max(v) / min(v) - 1) * 100 if len(v) > 1 else float("nan")


m = {a: 1000 / med(ts[a]) for a in ts}
mA = m["A"]
gap = m["G"] - mA
thr = 0.03 * mA
have = all(runs[a] for a in ("A", "G"))
ids_ok = len(ids["A"] | ids["G"]) == 1
gap_real = have and gap >= thr and max(runs["G"]) < min(runs["A"])
print("SERVEFLIP RESULT ts decode tok/s: " + " | ".join(
    f"{a} median {med(ts[a]):.2f} (m {m[a]:.2f} ms) n {len(ts[a])} run medians {[round(x, 2) for x in runs[a]]} spread {spread(runs[a]):.1f}%" for a in ts))
print(f"SERVEFLIP GAP m(G) - m(A) = {gap:+.2f} ms ({100 * gap / mA:+.1f}% of m(A); threshold {thr:.2f} ms) " + ("REAL" if gap_real else "not real"))
lines = []
for x in FLIPS:
    if not runs[x] or not have:
        lines.append(f"{x}: NO VERDICT (no valid sample)")
        continue
    if ids[x] != ids["G"] or len(ids[x]) != 1:
        lines.append(f"{x}: NO VERDICT (ids digests {sorted(ids[x])} differ from G {sorted(ids['G'])})")
        continue
    gain = m["G"] - m[x]
    share = gain / gap if gap else float("nan")
    real = gain >= thr and min(runs[x]) > max(runs["G"])
    noisy = spread(runs["G"]) > 5 or spread(runs[x]) > 5
    kind = "NOISY, no verdict" if noisy else ("CAUSE" if real and share >= 0.5 else ("PART" if real else "NOTHING"))
    lines.append(f"{x}: gain {gain:+.2f} ms/token = {100 * share:.0f}% of the gap, {'REAL' if real else 'not real'} -> {kind}")
    print(f"SERVEFLIP STEP {x}: m {m[x]:.2f} ms, gain over G {gain:+.2f} ms, share of gap {100 * share:.0f}%, runs {[round(v, 2) for v in runs[x]]}")
if not ids_ok:
    verdict = "NO VERDICT: the ids digests of A and G differ"
elif not gap_real:
    verdict = "GAP NOT REPRODUCED: no candidate judged"
else:
    verdict = "; ".join(lines)
print(f"SERVEFLIP VERDICT by the rule registered before the run: {verdict}")
print("SERVEFLIP DIGESTS ids " + ("equal in A and G: " + next(iter(ids['A'] | ids['G'])) if ids_ok else f"DIFFER: { {a: sorted(v) for a, v in ids.items()} }")
      + "; flips: " + ", ".join(f"{x} {'equal to G' if ids[x] == ids['G'] else sorted(ids[x])}" for x in FLIPS))
if bad:
    print("SERVEFLIP BAD SAMPLES " + "; ".join(bad)[:600])
text = read(log)
for mm in re.finditer(r"^ARM (\w+) (THREADS pid \d+ total \d+ util_min>=1024 \d+ util_min=0 \d+ unreadable \d+)", text, re.MULTILINE):
    print(f"SERVEFLIP arm {mm.group(1)} {mm.group(2)}")
starts, wins = {}, {}
for line in text.splitlines():
    mm = re.search(r"round (\d+) arm (\w+) load1=\S+ start_epoch=(\d+)", line)
    if mm:
        starts[(mm.group(2), mm.group(1))] = int(mm.group(3))
    mm = re.search(r"round (\d+) arm (\w+) end_epoch=(\d+)", line)
    if mm and (mm.group(2), mm.group(1)) in starts:
        wins.setdefault(mm.group(2), []).append((starts[(mm.group(2), mm.group(1))], int(mm.group(3))))
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
            for row in rows:
                try:
                    t = float(row["epoch"])
                    if t0 <= t <= t1 and pols:
                        p.append(int(row[pols[-1]]) / 1000.0)
                        hp.append(float(row["heatpipe_uW"]) / 1e6)
                except (ValueError, KeyError):
                    pass
        if p:
            print(f"SERVEFLIP CLOCK arm {arm}: mean P-cluster {st.mean(p):.0f} MHz, heatpipe power {st.mean(hp):.1f} W over {len(p)} one-second samples (whole runs, load and warm-up included)")
