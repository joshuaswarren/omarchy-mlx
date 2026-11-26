import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gdn_numcmp

ROOT = Path(sys.argv[1])
HOSTS = {h: n for h, n in {"m2": "14-inch M2 Max", "16-inch": "16-inch M1 Max", "13-inch": "13-inch M1"}.items() if (ROOT / h / "run.log").exists()}


def tok_s(path):
    text = path.read_text(errors="replace")
    m = re.search(r"decode ([\d.]+) tok/s", text)
    return float(m.group(1)) if m else None


def devname(path):
    text = path.read_text(errors="replace")
    m = re.search(r'"device"\s*:\s*"([^"]+)"', text)
    return m.group(1) if m else None


def hist(path):
    d = {}
    for ln in path.read_text().splitlines():
        parts = ln.split()
        if len(parts) == 2:
            d[int(parts[1])] = int(parts[0])
    return d


for h, name in HOSTS.items():
    D = ROOT / h
    print("=" * 8, h, name)
    devs = {devname(p) for p in D.glob("t-*.out")}
    print("device field of the 30 timing outputs:", devs)
    med = {}
    for pn in ("P1", "P2"):
        for arm in ("D", "T", "U"):
            v = [tok_s(D / f"t-{pn}-{arm}-{i}.out") for i in range(1, 6)]
            assert None not in v, (pn, arm, v)
            med[(pn, arm)] = statistics.median(v)
            print(f"  speed {pn} {arm}: rounds {v} median {med[(pn, arm)]:.3f} spread {100 * (max(v) - min(v)) / statistics.median(v):.2f}%")
    for pn in ("P1", "P2"):
        print(f"  {pn}: D/T {med[(pn, 'D')] / med[(pn, 'T')]:.4f}  D/U {med[(pn, 'D')] / med[(pn, 'U')]:.4f}  U/T {med[(pn, 'U')] / med[(pn, 'T')]:.4f}")
    for arm in ("D", "T", "U"):
        h6, h10 = hist(D / f"hist-{arm}-6.txt"), hist(D / f"hist-{arm}-10.txt")
        tot6, tot10 = sum(h6.values()), sum(h10.values())
        per = (tot10 - tot6) / 4
        diff = {k: (h10.get(k, 0) - h6.get(k, 0)) / 4 for k in set(h6) | set(h10) if h10.get(k, 0) != h6.get(k, 0)}
        top = {k: diff[k] for k in (637, 514, 516, 58) if k in diff}
        print(f"  census {arm}: totals {tot6} {tot10} -> {per} dispatches per token; per-token by kernel {top}")
    for pn, pairs in (("P1", (("D", "T"), ("D", "U"), ("D", "D2"), ("T", "U"))), ("P2", (("D", "T"),))):
        for x, y in pairs:
            r = gdn_numcmp.compare(D / f"num-{pn}-{x}", D / f"num-{pn}-{y}")
            print(
                f"  numerics {pn} {x} vs {y}: ids_equal {r['ids_equal']} full_steps {r['full_steps_compared']} "
                f"max|d| full {r['max_abs_dlogprob_full']} elements differ {r['n_elements_differ_full']} "
                f"top1 {r['top1_equal_steps']} top5 {r['top5_set_equal_steps']} KL {r['max_kl_x_to_y']}"
            )
    log = (D / "run.log").read_text()
    print("  rc!=0 in the log:", len(re.findall(r"rc=[1-9]", log)), "| verified=match count:", log.count("verified=match"), "| whole binary:",
          re.findall(r"test whole binary rc=(\d+)", log))
