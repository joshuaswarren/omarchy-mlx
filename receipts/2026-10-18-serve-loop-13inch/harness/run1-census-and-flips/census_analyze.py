"""census_analyze.py OUT_DIR: per-token dispatch histograms from census_ab.sh (hist-A-6/10, hist-G-24/40, hist-GM-24/40) and their differences.

per token = (hist(big) - hist(small)) / delta tokens; delta is 4 for A (tokens 6 -> 10) and 20 for G and GM (TOKENS 24 -> 40: requests of
8 + 6 + 24 and 8 + 10 + 40 tokens). A key is "kernel=<ComputeKernel id> count=<n> gx gy gz". Prints the per-token totals, the ids and keys
whose per-token count differs between G and A (largest first), and between GM and G. Only differences of at least 0.5 dispatches per token
are listed. Reading rule written before the run: a kernel id that G has per token and A does not, or has more of, is where G does extra work;
GM must remove dispatches relative to G (else the mask flip did nothing on the device).
"""
import re
import sys
from collections import Counter
from pathlib import Path

out = Path(sys.argv[1])


def load(name):
    h = Counter()
    try:
        for line in (out / name).read_text().splitlines():
            m = re.match(r"\s*(\d+) (kernel=\d+ count=\d+ gx=\d+ gy=\d+ gz=\d+)", line)
            if m:
                h[m.group(2)] += int(m.group(1))
    except OSError:
        pass
    return h


def per_token(arm, small, big, delta):
    hs, hb = load(f"hist-{arm}-{small}.txt"), load(f"hist-{arm}-{big}.txt")
    if not hs or not hb:
        return None
    return {k: (hb[k] - hs[k]) / delta for k in set(hs) | set(hb) if hb[k] != hs[k]}


def kid(key):
    return int(re.match(r"kernel=(\d+)", key).group(1))


def by_id(pt):
    c = Counter()
    for k, v in pt.items():
        c[kid(k)] += v
    return c


A, G, GM = per_token("A", 6, 10, 4), per_token("G", 24, 40, 20), per_token("GM", 24, 40, 20)
for name, pt in (("A", A), ("G", G), ("GM", GM)):
    if pt is None:
        print(f"SERVECENSUS {name}: NO DISPATCH LINES (trace unavailable or the run failed)")
        continue
    print(f"SERVECENSUS {name}: {sum(pt.values()):.1f} dispatches per decoded token over {len(by_id(pt))} kernel ids")


def diff(x, y, xn, yn):
    if x is None or y is None:
        return
    ix, iy = by_id(x), by_id(y)
    rows = sorted(((iy[i] - ix[i], i, ix[i], iy[i]) for i in set(ix) | set(iy) if abs(iy[i] - ix[i]) >= 0.5), key=lambda r: -abs(r[0]))
    print(f"SERVECENSUS {yn} minus {xn} per token by kernel id (>= 0.5): " + ("; ".join(f"id {i}: {xv:.1f} -> {yv:.1f} ({d:+.1f})" for d, i, xv, yv in rows[:20]) or "none"))
    print(f"SERVECENSUS {yn} minus {xn} total {sum(y.values()) - sum(x.values()):+.1f} dispatches per token")
    keys = sorted(set(x) | set(y), key=lambda k: -abs(y.get(k, 0) - x.get(k, 0)))
    print(f"SERVECENSUS {yn} minus {xn} by key (top 12): " + "; ".join(f"{k} {x.get(k, 0):.1f} -> {y.get(k, 0):.1f}" for k in keys[:12] if abs(y.get(k, 0) - x.get(k, 0)) >= 0.5))


diff(A, G, "A", "G")
diff(G, GM, "G", "GM")
