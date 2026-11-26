import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
import gdn_numcmp

D = Path("./pfres2")
for f in sorted(D.glob("t-*.out")):
    text = f.read_text()
    js = [json.loads(line) for line in text.splitlines() if line.startswith("{")]
    print(f.name, "json", len(js), "verified=match" in text, js[0]["decode_tps"], js[0]["ids_sha256_16"], js[0]["generated"], js[0]["device"])
for x, y in (("A", "B"), ("B", "C"), ("C", "C2"), ("A", "C"), ("A", "C2")):
    r = gdn_numcmp.compare(D / f"num-P1-{x}", D / f"num-P1-{y}")
    print("P1", x, y, "ids_equal", r["ids_equal"], "full_steps", r["full_steps_compared"], "max|d| full", r["max_abs_dlogprob_full"], "differ", r["n_elements_differ_full"],
          "top1", r["top1_equal_steps"], "top5", r["top5_set_equal_steps"], "KL", r["max_kl_x_to_y"])
for arm in "ABC":
    for n in (6, 10):
        out = (D / f"cen-{arm}-{n}.out").read_text()
        print(arm, n, "verified=match" in out, "ids", re.findall(r"ids_sha256_16 (\w+)", out)[:1])


def load(arm, n):
    c = Counter()
    for line in (D / f"hist-{arm}-{n}.txt").read_text().splitlines():
        cnt, k = line.split()
        c[int(k)] = int(cnt)
    return c


for arm in "ABC":
    h6, h10 = load(arm, 6), load(arm, 10)
    per = {k: (h10[k] - h6[k]) / 4 for k in set(h6) | set(h10) if h10[k] != h6[k]}
    print(arm, "dispatches per token", (sum(h10.values()) - sum(h6.values())) / 4, {k: per.get(k) for k in (514, 516, 636, 637, 58)})
