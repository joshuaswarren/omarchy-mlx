"""Read the block of fixab_ab.sh and print the result lines.

Usage: fixab_analyze.py OUT_DIR ROUNDS REQS CSV LOG

Arms: A bench_decode (tree main81, anchor); G / GF the mlx_lm.server generator in-process on tree main81 (before) / main81-fix (after);
H / HF the real server (non-streamed serve_client.py) on main81 / main81-fix. main81-fix = main81 + the rotating-cache mask patch and the
server logprob-read patch. Rates: ts_tps for A, G, GF; the wall difference of a short and a long request for H and HF. m = 1000 / tok/s.

REGISTERED RULE (fixed before the run, never changed after seeing numbers). X is FASTER THAN Y when every run median of X is above every run
median of Y. A gain is REAL when it is at least 3 percent of m(A) and the faster arm is FASTER THAN the slower one.
  IN-PROCESS FIX = G -> GF: gain_G = m(G) - m(GF); share_G = gain_G / (m(G) - m(A)) (the part of the measured generator gap it closes).
  SERVED FIX     = H -> HF: gain_H = m(H) - m(HF), also as a ratio of tok/s HF / H.
  NO VERDICT when the ids digests of A, G and GF are not all equal, or the text digests of H and HF differ (same bytes of generated text are
  required), or an arm has no valid sample. NOISY (no verdict) when the spread of run medians of A, G, GF, H or HF is above 5 percent.
  Otherwise: FIX WORKS when the SERVED FIX is REAL; FIX DOES NOT SHOW IN THE SERVER when the in-process fix is REAL and the served fix is not;
  FIX DOES NOT WORK when neither is REAL. The shares are printed, not judged.
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


def jlines(text):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                yield json.loads(line)
            except ValueError:
                pass


ALL = ("A", "G", "GF", "H", "HF")
rate = {a: [] for a in ALL}
runs = {a: [] for a in ALL}
ids = {a: set() for a in ("A", "G", "GF")}
text = {a: set() for a in ("H", "HF")}
bad = []
for r in range(1, rounds + 1):
    j = [x for x in jlines(read(out / f"run-A{r}.out")) if x.get("engine") == "bench_decode"]
    if j and j[-1].get("decode_tps"):
        rate["A"].append(j[-1]["decode_tps"])
        runs["A"].append(j[-1]["decode_tps"])
        ids["A"].add(j[-1]["ids_sha256_16"])
    else:
        bad.append(f"A{r}: no bench_decode result")
    for a in ("G", "GF"):
        got = []
        for x in jlines(read(out / f"run-{a}{r}.out")):
            if x.get("arm") != "G" or x.get("finish") != "length" or not x.get("ts_tps"):
                bad.append(f"{a}{r}: {str(x)[:100]}")
                continue
            got.append(x["ts_tps"])
            ids[a].add(x["ids_sha16"])
        rate[a].extend(got)
        if got:
            runs[a].append(st.median(got))
    for a in ("H", "HF"):
        got = []
        for q in range(1, reqs + 1):
            body = read(out / f"run-{a}{r}.req{q}").strip()
            try:
                x = json.loads(body.splitlines()[-1])
            except (ValueError, IndexError):
                bad.append(f"{a}{r}.req{q}: unreadable")
                continue
            if x.get("finish") != "length" or not x.get("decode_tps"):
                bad.append(f"{a}{r}.req{q}: finish {x.get('finish')}")
                continue
            got.append(x["decode_tps"])
            text[a].add(x["text_sha16"])
        rate[a].extend(got)
        if got:
            runs[a].append(st.median(got))


def med(v):
    return st.median(v) if v else float("nan")


def spread(v):
    return (max(v) / min(v) - 1) * 100 if len(v) > 1 else float("nan")


present = [a for a in ALL if runs[a]]
print("SERVEFIX RESULT decode tok/s: " + " | ".join(
    f"{a} median {med(rate[a]):.2f} (m {1000 / med(rate[a]):.2f} ms) n {len(rate[a])} run medians {[round(x, 2) for x in runs[a]]} spread {spread(runs[a]):.1f}%" for a in present))
verdict = ""
if len(present) < len(ALL):
    verdict = f"NO VERDICT: no valid sample for {[a for a in ALL if a not in present]}"
else:
    m = {a: 1000 / med(rate[a]) for a in ALL}
    thr = 0.03 * m["A"]

    def faster(x, y):
        return min(runs[x]) > max(runs[y])

    gG, gH = m["G"] - m["GF"], m["H"] - m["HF"]
    realG, realH = gG >= thr and faster("GF", "G"), gH >= thr and faster("HF", "H")
    gap = m["G"] - m["A"]
    print(f"SERVEFIX IN-PROCESS G -> GF: gain {gG:+.2f} ms/token ({med(rate['GF']) / med(rate['G']):.3f}x tok/s), share of the G-A gap ({gap:+.2f} ms) {100 * gG / gap if gap else float('nan'):.0f}% "
          + ("REAL" if realG else "not real"))
    print(f"SERVEFIX SERVED H -> HF: gain {gH:+.2f} ms/token ({med(rate['HF']) / med(rate['H']):.3f}x tok/s) " + ("REAL" if realH else "not real")
          + f"; HF {med(rate['HF']):.2f} tok/s against A {med(rate['A']):.2f}")
    ids_all = set().union(*ids.values())
    text_all = text["H"] | text["HF"]
    noisy = [a for a in ALL if spread(runs[a]) > 5]
    if len(ids_all) != 1 or len(text["H"]) != 1 or text["H"] != text["HF"]:
        verdict = "NO VERDICT: digests differ (ids A/G/GF or served text H/HF)"
    elif noisy:
        verdict = f"NOISY: spread above 5 percent in {noisy}, no verdict"
    elif realH:
        verdict = "FIX WORKS: the served gain is REAL, ids and served text equal"
    elif realG:
        verdict = "FIX DOES NOT SHOW IN THE SERVER: in-process gain REAL, served gain not REAL"
    else:
        verdict = "FIX DOES NOT WORK: neither gain is REAL"
print(f"SERVEFIX VERDICT by the rule registered before the run: {verdict}")
ids_all = set().union(*ids.values())
print("SERVEFIX DIGESTS ids " + ("equal in A, G, GF: " + next(iter(ids_all)) if len(ids_all) == 1 else f"DIFFER: { {a: sorted(v) for a, v in ids.items()} }")
      + "; served text " + ("equal in H and HF: " + next(iter(text['H'])) if len(text["H"]) == 1 and text["H"] == text["HF"] else f"DIFFER: { {a: sorted(v) for a, v in text.items()} }"))
if bad:
    print("SERVEFIX BAD SAMPLES " + "; ".join(bad)[:600])
txt = read(log)
starts, wins = {}, {}
for line in txt.splitlines():
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
            print(f"SERVEFIX CLOCK arm {arm}: mean P-cluster {st.mean(p):.0f} MHz, heatpipe power {st.mean(hp):.1f} W over {len(p)} one-second samples (whole runs, load and warm-up included)")
