import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import gdn_numcmp  # noqa: E402

D = Path("./pfres")


def speed():
    data = {}
    digest = {}
    bad = []
    for f in sorted(D.glob("t-*.out")):
        m = re.match(r"t-(P\d)-([ABC])-(\d)\.out", f.name)
        pn, arm, rnd = m.groups()
        js = [json.loads(line) for line in f.read_text().splitlines() if line.startswith("{")]
        if len(js) != 1:
            bad.append(f.name)
            continue
        j = js[0]
        data.setdefault((pn, arm), {})[int(rnd)] = j["decode_tps"]
        digest.setdefault(pn, set()).add((arm, j["ids_sha256_16"], j["generated"]))
        if "verified=match" not in f.read_text():
            bad.append(f.name + " provenance")
    print("bad files:", bad)
    for pn in ("P1", "P2"):
        med = {}
        for arm in "ABC":
            v = [data[(pn, arm)][r] for r in sorted(data[(pn, arm)])]
            med[arm] = st.median(v)
            print(pn, arm, "n", len(v), "median", med[arm], "min", min(v), "max", max(v), v)
        for x, y in (("B", "A"), ("C", "A"), ("C", "B")):
            lo_x = min(data[(pn, x)].values())
            hi_y = max(data[(pn, y)].values())
            print(f"{pn} {x} vs {y}: {100 * (med[x] / med[y] - 1):+.2f} percent, every {x} above every {y}: {lo_x > hi_y}")
        print(pn, "digests", sorted(digest[pn]))
        pairs = Counter()
        for r in sorted(data[(pn, "A")]):
            pairs["C>B" if data[(pn, "C")][r] > data[(pn, "B")][r] else "C<=B"] += 1
        print(pn, "per-round C vs B", dict(pairs))


def numerics():
    for pn in ("P1", "P2"):
        for x, y in (("A", "B"), ("B", "C"), ("C", "C2"), ("A", "C"), ("A", "A")):
            if x == y:
                continue
            r = gdn_numcmp.compare(D / f"num-{pn}-{x}", D / f"num-{pn}-{y}")
            print(pn, x, y, {k: r[k] for k in r if k in ("n", "ids_equal", "first_divergence", "max_abs_full", "full_equal", "top1_equal", "top5_equal", "max_kl", "elements_differ")} or r)


def census():
    def load(arm, n):
        c = Counter()
        for line in (D / f"hist-{arm}-{n}.txt").read_text().splitlines():
            cnt, k = line.split()
            c[int(k)] = int(cnt)
        return c

    for arm in "ABC":
        h6, h10, h128 = load(arm, 6), load(arm, 10), load(arm, 128)
        per = {k: (h10[k] - h6[k]) / 4 for k in set(h6) | set(h10) if h10[k] != h6[k]}
        tot = sum(h10.values()) - sum(h6.values())
        print(arm, "dispatches per token", tot / 4, "kernels with per-token change:", {k: v for k, v in sorted(per.items()) if v >= 8})
        print(arm, "ids 636 per token", per.get(636), "514", per.get(514), "516", per.get(516), "58", per.get(58), "total128", sum(h128.values()))


speed()
numerics()
census()
