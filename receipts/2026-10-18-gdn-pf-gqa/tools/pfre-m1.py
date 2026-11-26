import json
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import gdn_numcmp

host = sys.argv[1]
D = Path(f"./pfres-{host}")
print("host", host, "dir", D)
data, digest, bad = {}, {}, []
for f in sorted(D.glob("t-*.out")):
    m = re.match(r"t-(P\d)-([ABC])-(\d)\.out", f.name)
    pn, arm, rnd = m.groups()
    text = f.read_text()
    js = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    if len(js) != 1 or "verified=match" not in text:
        bad.append(f.name)
        continue
    data.setdefault((pn, arm), {})[int(rnd)] = js[0]["decode_tps"]
    digest.setdefault(pn, set()).add((arm, js[0]["ids_sha256_16"], js[0]["generated"]))
    dev = js[0]["device"]
print("bench files", len(list(D.glob("t-*.out"))), "bad", bad, "device", dev)
for pn in ("P1", "P2"):
    med = {}
    for arm in "ABC":
        v = [data[(pn, arm)][r] for r in sorted(data[(pn, arm)])]
        med[arm] = st.median(v)
        print(pn, arm, "n", len(v), "median", med[arm], "min", min(v), "max", max(v), v)
    for x, y in (("B", "A"), ("C", "A"), ("C", "B")):
        print(f"{pn} {x} vs {y}: {100 * (med[x] / med[y] - 1):+.2f} percent, every {x} above every {y}: {min(data[(pn, x)].values()) > max(data[(pn, y)].values())}")
    print(pn, "per-round C>B", sum(data[(pn, "C")][r] > data[(pn, "B")][r] for r in data[(pn, "C")]), "of", len(data[(pn, "C")]))
    print(pn, "digests", sorted(digest[pn]))
for x, y in (("A", "B"), ("B", "C"), ("C", "C2"), ("A", "C"), ("A", "C2")):
    r = gdn_numcmp.compare(D / f"num-P1-{x}", D / f"num-P1-{y}")
    print("P1", x, y, "ids_equal", r["ids_equal"], "full_steps", r["full_steps_compared"], "max|d| full", r["max_abs_dlogprob_full"],
          "differ", r["n_elements_differ_full"], "top1", r["top1_equal_steps"], "top5", r["top5_set_equal_steps"], "KL", r["max_kl_x_to_y"])


def load(arm, n):
    c = Counter()
    for line in (D / f"hist-{arm}-{n}.txt").read_text().splitlines():
        cnt, k = line.split()
        c[int(k)] = int(cnt)
    return c


for arm in "ABC":
    h6, h10 = load(arm, 6), load(arm, 10)
    per = {k: (h10[k] - h6[k]) / 4 for k in set(h6) | set(h10) if h10[k] != h6[k]}
    print(arm, "dispatches per token", (sum(h10.values()) - sum(h6.values())) / 4, "kernel 514", per.get(514), "516", per.get(516), "636", per.get(636), "479", per.get(479), "480", per.get(480), "58", per.get(58))
