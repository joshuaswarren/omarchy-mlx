"""Read the idle power block: idlepow_analyze.py CSV LOG SKIP

Four windows (P S S P). For each: mean of the 1 Hz samples after the first SKIP seconds: system power, heatpipe power, the highest-numbered policy's clock
(a P cluster), fan 1 and the battery hotspot temperature. Per governor: the mean of its two windows, and the gap between its two windows (the noise of
this measurement). The difference P minus S is reported in watts and percent and counts only when it is larger than both gaps and larger than 0.3 W;
otherwise the line says "no difference above the window gap".
"""
import csv
import re
import statistics as st
import sys

csv_path, log_path, skip = sys.argv[1], sys.argv[2], float(sys.argv[3])
rows = list(csv.DictReader(open(csv_path)))
text = open(log_path).read()
wins = []
for m in re.finditer(r"window (\d) arm ([PS]) governors=\S+ start_epoch=(\d+)", text):
    n, arm, s = m.group(1), m.group(2), int(m.group(3))
    e = re.search(rf"window {n} arm {arm} end_epoch=(\d+)", text)
    if e:
        wins.append((int(n), arm, s, int(e.group(1))))
pols = sorted((c for c in rows[0] if c.startswith("khz_policy")), key=lambda c: int(c[len("khz_policy"):])) if rows else []
cols = {"sys_W": ("sys_power_uW", 1e6), "heatpipe_W": ("heatpipe_uW", 1e6), "pcluster_MHz": (pols[-1] if pols else "", 1e3),
        "fan1_rpm": ("fan1_rpm", 1.0), "hotspot_C": ("batt_hotspot_mC", 1e3)}
per = {}
for n, arm, s, e in wins:
    res = {}
    for name, (col, div) in cols.items():
        v = []
        for r in rows:
            try:
                t = float(r["epoch"])
                if s + skip <= t <= e and r.get(col) not in ("", None):
                    v.append(float(r[col]) / div)
            except (ValueError, KeyError):
                pass
        res[name] = (st.mean(v), len(v)) if v else (float("nan"), 0)
    per[n] = (arm, res)
    print(f"IDLEPOW window {n} arm {arm}: " + ", ".join(f"{k} {v[0]:.2f} (n={v[1]})" for k, v in res.items()))
for name in cols:
    by = {"P": [], "S": []}
    for arm, res in per.values():
        by[arm].append(res[name][0])
    if len(by["P"]) != 2 or len(by["S"]) != 2:
        print(f"IDLEPOW {name}: windows missing")
        continue
    mp, ms = st.mean(by["P"]), st.mean(by["S"])
    gp, gs = abs(by["P"][0] - by["P"][1]), abs(by["S"][0] - by["S"][1])
    diff = mp - ms
    pct = 100.0 * diff / ms if ms else float("nan")
    floor = 0.3 if name in ("sys_W", "heatpipe_W") else 0.0
    counts = abs(diff) > max(gp, gs, floor)
    print(f"IDLEPOW {name}: P {mp:.2f} (windows {by['P'][0]:.2f}, {by['P'][1]:.2f}), S {ms:.2f} (windows {by['S'][0]:.2f}, {by['S'][1]:.2f}), P minus S {diff:+.2f} ({pct:+.1f} percent), "
          f"window gaps {gp:.2f} and {gs:.2f}: " + ("a difference above the window gap" if counts else "no difference above the window gap"))
