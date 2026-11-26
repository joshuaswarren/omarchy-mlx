"""Read the clamp placement block and print the result lines.

Usage: clampab_analyze.py OUT_DIR ROUNDS CSV LOG
Arms N (none), T (calling thread only, after the imports), B (before anything starts), A (every thread after the imports).
Control: N. A spread of N above 5 percent makes the block NOISY. Ratios T/N, B/N and A/N use the arm medians.
"""
import csv
import re
import statistics as st
import sys

DEC = re.compile(r"decode ([\d.]+) tok/s over (\d+) tokens")
DIG = re.compile(r"generated_ids sha256:([0-9a-f]+) n=(\d+)")


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def windows(log):
    starts, out = {}, {}
    for line in read(log).splitlines():
        m = re.search(r"round (\d+) arm ([NTBA]) mode \S+ governors=\S* load1=\S+ start_epoch=(\d+)", line)
        if m:
            starts[(m.group(2), m.group(1))] = int(m.group(3))
        m = re.search(r"round (\d+) arm ([NTBA]) end_epoch=(\d+)", line)
        if m and (m.group(2), m.group(1)) in starts:
            out.setdefault(m.group(2), []).append((starts[(m.group(2), m.group(1))], int(m.group(3))))
    return out


def clocks(path, wins):
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return {}
    if not rows:
        return {}
    pols = sorted((c for c in rows[0] if c.startswith("khz_policy")), key=lambda c: int(c[len("khz_policy"):]))
    if not pols:
        return {}
    res = {}
    for arm, ws in wins.items():
        p, hp = [], []
        for t0, t1 in ws:
            for r in rows:
                try:
                    t = float(r["epoch"])
                    if t0 <= t <= t1:
                        p.append(int(r[pols[-1]]) / 1000.0)
                        hp.append(float(r["heatpipe_uW"]) / 1e6)
                except (ValueError, KeyError):
                    pass
        if p:
            res[arm] = (st.mean(p), st.mean(hp) if hp else 0.0, len(p))
    return res


def main():
    out, rounds, csv_path, log = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    vals, digs = {a: [] for a in "NTBA"}, {a: set() for a in "NTBA"}
    for a in "NTBA":
        for r in range(1, rounds + 1):
            t = read(f"{out}/run-{a}{r}.out")
            m, d = DEC.search(t), DIG.search(t)
            if m:
                vals[a].append(float(m.group(1)))
            if d:
                digs[a].add(d.group(1))
    if not all(vals[a] for a in "NTBA"):
        print(f"CLAMP RESULT incomplete {vals}")
        return 1
    med = {a: st.median(vals[a]) for a in "NTBA"}
    spr = {a: (max(vals[a]) - min(vals[a])) / med[a] * 100 for a in "NTBA"}
    flag = " NOISY(control spread over 5%: no gain claimed)" if spr["N"] > 5 else ""
    names = {"N": "none", "T": "thread only", "B": "before", "A": "all threads"}
    print(
        "CLAMP RESULT decode: "
        + " | ".join(f"{a} {names[a]} median {med[a]:.2f} {[round(x, 2) for x in vals[a]]} spread {spr[a]:.1f}%" for a in "NTBA")
        + f" | T/N {med['T'] / med['N']:.3f} B/N {med['B'] / med['N']:.3f} A/N {med['A'] / med['N']:.3f}{flag}"
    )
    alld = set().union(*digs.values())
    print("CLAMP DIGESTS " + (f"equal in all arms: {next(iter(alld))}" if len(alld) == 1 else f"DIFFER {sorted(alld)}"))
    clk = clocks(csv_path, windows(log))
    for a in "NTBA":
        if a in clk:
            p, hp, n = clk[a]
            print(f"CLAMP CLOCK arm {a}: mean P-cluster {p:.0f} MHz, heatpipe power {hp:.1f} W over {n} one-second samples (whole runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
