"""Read the block of flip_ab.sh and print the result lines.

Usage: flip_analyze.py OUT_DIR ROUNDS REQS CSV LOG

Arms: A bench_decode (thread clamp), G mlx_lm.server generator in-process, GM / GL = G with ONE cost removed (GM no all-True mask array
for the sliding batch cache, GL no per-token logprob read), GML = both removed, GMLT = GML with phase timers (rate not judged), GS (optional)
a new thread-local stream. Rate: ts_tps for G-type arms (consumer-side token stamps), bench_decode's own tok/s for A; m = 1000 / tok/s.

REGISTERED RULE (fixed before the run, never changed after seeing numbers). An arm X is FASTER THAN Y when every run median of X is above
every run median of Y. A gain is REAL when it is at least 3 percent of m(A) and the faster arm is FASTER THAN the slower one.
  gap = m(G) - m(A) must be REAL, else GAP NOT REPRODUCED and nothing else is judged.
  gGM = m(G) - m(GM), gGL = m(G) - m(GL), gML = m(G) - m(GML); each REAL or not.
  ADDITIVITY (judged only when gGM, gGL and gML are all REAL): d = gML - (gGM + gGL).
     ADDITIVE if |d| <= 1.0 ms; SUB-ADDITIVE if d < -1.0 ms; SUPER-ADDITIVE if d > +1.0 ms. Otherwise ADDITIVITY NOT JUDGED.
  REMAINDER = m(GML) - m(A): REAL when it is at least 3 percent of m(A) and A is FASTER THAN GML; otherwise NOT REAL (GML is within noise of A).
  NO VERDICT for the block when the ids digests of the arms present differ; NOISY (no verdict) when the spread of run medians (largest over
  smallest) of G, GM, GL, GML or A is above 5 percent. PHASES (information, not judged): from GMLT, mean per decoded token of the time in
  BatchGenerator.next(), in mx.eval and mx.async_eval inside it, the host part of next() (next - eval - async_eval), the calls per token, the
  wall per token and the part outside next() (wall - next: the server loop body, detokenizer, queue hand-off and the consumer). The timers
  perturb the rate of GMLT, which is why its rate is not judged.
"""
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

out, rounds, reqs, csv_path, log = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
GARMS = ("G", "GM", "GL", "GML", "GMLT", "GS")
ALL = ("A",) + GARMS


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


ts = {a: [] for a in ALL}
runs = {a: [] for a in ALL}
ids = {a: set() for a in ALL}
phases = []
bad = []
for r in range(1, rounds + 1):
    if (out / f"run-A{r}.out").exists():
        j = [x for x in jlines(read(out / f"run-A{r}.out")) if x.get("engine") == "bench_decode"]
        if j and j[-1].get("decode_tps"):
            ts["A"].append(j[-1]["decode_tps"])
            runs["A"].append(j[-1]["decode_tps"])
            ids["A"].add(j[-1]["ids_sha256_16"])
        else:
            bad.append(f"A{r}: no bench_decode result")
    for a in GARMS:
        if not (out / f"run-{a}{r}.out").exists():
            continue
        got = []
        for x in jlines(read(out / f"run-{a}{r}.out")):
            if x.get("arm") != a or x.get("finish") != "length" or not x.get("ts_tps"):
                bad.append(f"{a}{r}: {str(x)[:120]}")
                continue
            got.append(x["ts_tps"])
            ids[a].add(x["ids_sha16"])
            if a == "GMLT" and x.get("ph_ms_per_token"):
                phases.append((x["ph_ms_per_token"], x["wall_ms_per_token"]))
        ts[a].extend(got)
        if got:
            runs[a].append(st.median(got))
present = [a for a in ALL if runs[a]]


def med(v):
    return st.median(v) if v else float("nan")


def spread(v):
    return (max(v) / min(v) - 1) * 100 if len(v) > 1 else float("nan")


m = {a: 1000 / med(ts[a]) for a in present}
thr = 0.03 * m["A"] if "A" in m else float("nan")


def faster(x, y):
    """every run median of x above every run median of y"""
    return bool(runs[x]) and bool(runs[y]) and min(runs[x]) > max(runs[y])


def gain(slow, fast):
    g = m[slow] - m[fast]
    return g, (g >= thr and faster(fast, slow))


print("SERVEFLIP RESULT ts decode tok/s: " + " | ".join(
    f"{a} median {med(ts[a]):.2f} (m {m[a]:.2f} ms) n {len(ts[a])} run medians {[round(x, 2) for x in runs[a]]} spread {spread(runs[a]):.1f}%" for a in present))
need = ("A", "G", "GM", "GL", "GML")
missing = [a for a in need if a not in m]
ids_all = set().union(*(ids[a] for a in present)) if present else set()
ids_ok = len(ids_all) == 1
verdict = ""
if missing:
    verdict = f"NO VERDICT: no valid sample for {missing}"
else:
    gap = m["G"] - m["A"]
    gap_real = gap >= thr and faster("A", "G")
    print(f"SERVEFLIP GAP m(G) - m(A) = {gap:+.2f} ms ({100 * gap / m['A']:+.1f}% of m(A); threshold {thr:.2f} ms) " + ("REAL" if gap_real else "not real"))
    gGM, rGM = gain("G", "GM")
    gGL, rGL = gain("G", "GL")
    gML, rML = gain("G", "GML")
    print(f"SERVEFLIP GAINS over G: GM {gGM:+.2f} ms {'REAL' if rGM else 'not real'}; GL {gGL:+.2f} ms {'REAL' if rGL else 'not real'}; "
          f"GML {gML:+.2f} ms {'REAL' if rML else 'not real'}; GM + GL = {gGM + gGL:+.2f} ms")
    if rGM and rGL and rML:
        d = gML - (gGM + gGL)
        add = "ADDITIVE" if abs(d) <= 1.0 else ("SUB-ADDITIVE" if d < 0 else "SUPER-ADDITIVE")
        print(f"SERVEFLIP ADDITIVITY d = GML gain - (GM gain + GL gain) = {d:+.2f} ms -> {add}")
    else:
        add = "ADDITIVITY NOT JUDGED (a gain is not REAL)"
        print(f"SERVEFLIP ADDITIVITY {add}")
    rem = m["GML"] - m["A"]
    rem_real = rem >= thr and faster("A", "GML")
    print(f"SERVEFLIP REMAINDER m(GML) - m(A) = {rem:+.2f} ms ({100 * rem / m['A']:+.1f}% of m(A)) " + ("REAL" if rem_real else "NOT REAL (GML within noise of A)"))
    noisy = [a for a in ("A", "G", "GM", "GL", "GML") if spread(runs[a]) > 5]
    if not ids_ok:
        verdict = "NO VERDICT: the ids digests differ between arms"
    elif noisy:
        verdict = f"NOISY: spread above 5 percent in {noisy}, no verdict"
    elif not gap_real:
        verdict = "GAP NOT REPRODUCED"
    else:
        verdict = f"{add}; remainder {'REAL' if rem_real else 'NOT REAL'} ({rem:+.2f} ms)"
print(f"SERVEFLIP VERDICT by the rule registered before the run: {verdict}")
print("SERVEFLIP DIGESTS ids " + ("equal in all arms: " + next(iter(ids_all)) if ids_ok else f"DIFFER: { {a: sorted(v) for a, v in ids.items() if v} }"))
if phases:
    keys = ("next", "eval", "async_eval", "calls")
    mean = {k: st.mean(p[0][k] for p in phases) for k in keys}
    wall = st.mean(p[1] for p in phases)
    print(f"SERVEFLIP PHASES (GMLT, {len(phases)} pairs, per decoded token): wall {wall:.2f} ms; next() {mean['next']:.2f} ms of which mx.eval {mean['eval']:.2f}, "
          f"mx.async_eval {mean['async_eval']:.2f}, host rest {mean['next'] - mean['eval'] - mean['async_eval']:.2f}; outside next() {wall - mean['next']:.2f} ms; "
          f"next() calls per token {mean['calls']:.2f}")
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
