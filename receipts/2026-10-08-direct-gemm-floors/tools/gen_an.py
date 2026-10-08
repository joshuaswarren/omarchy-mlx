"""H34 analysis. usage: gen_an.py <dir> <chip g13g|g14c|g13c> [base-arm label, default C2];
reads gen-<chip>-{BASE,PR}-r{1,2,3}.jsonl.

A cell has a PR row if m >= the row floor and n >= 4096 for its (chip, dtype, orientation) and the base arm has no row
there (m below OLD); otherwise it is a control. Rules (pre-registered): hashes equal; REGRESSION = PR slower than the
base by > 3 % on at least two of three rounds; NO BENEFIT = gain < 3 % on every round; control = |diff| < 3 % on every
round; base spread > 5 % marks the cell void.
"""
import json
import sys

d, chip = sys.argv[1], sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else "C2"
FLOOR = {  # (dtype, ori) -> m floor of the PR row
    "g14c": {("bf16", "nt"): 512, ("f16", "nt"): 1024, ("bf16", "tn"): 4096},
    "g13g": {("f16", "nt"): 512, ("f16", "tn"): 1024, ("bf16", "tn"): 512, ("f32", "nt"): 512, ("f32", "nn"): 2048, ("f32", "tn"): 512},
    "g13c": {("f16", "nt"): 512, ("f16", "tn"): 4096, ("bf16", "tn"): 4096, ("bf16", "nt"): 512, ("f32", "nt"): 4096},
}[chip]
OLD = {  # floor of the row the base arm already has on that chip (a cell is a row cell only below it)
    "g14c": {},
    "g13g": {("f16", "nt"): 512, ("f16", "tn"): 4096, ("bf16", "tn"): 4096, ("f32", "nt"): 4096, ("f32", "nn"): 4096, ("f32", "tn"): 4096},
    "g13c": {("f16", "nt"): 512, ("f16", "tn"): 4096},
}[chip]
STAMP = {"C2": "27aa6738", "C6": "8235d7c6", "C7": "14ca9e39", "PR": "582c1a94", "M": "ef70b8cc"}


def check_stamps(files_by_arm):
    """Each arm's host record must carry the build stamp of the commit the arm is meant to be, and the stamps must differ."""
    seen = {}
    for arm, paths in files_by_arm.items():
        for path in paths:
            with open(path) as f:
                host = next((x for x in map(json.loads, f) if x.get("k") == "host"), None)
            stamp = host["mlx"] if host else None
            if stamp is None or not stamp.endswith("+" + STAMP[arm]):
                print(f"STAMP MISMATCH arm {arm}: {path} carries {stamp}, expected a build ending +{STAMP[arm]}")
                sys.exit(2)
            seen.setdefault(arm, set()).add(stamp)
    if len(set().union(*seen.values())) < len(seen) or any(len(v) != 1 for v in seen.values()):
        print("STAMP CHECK FAILED: arms share a build or an arm mixes builds:", {a: sorted(v) for a, v in seen.items()})
        sys.exit(2)
    print("build stamps:", {a: sorted(v)[0] for a, v in seen.items()})


check_stamps({arm: [f"{d}/gen-{chip}-{arm}-r{r}.jsonl" for r in (1, 2, 3)] for arm in (BASE, "PR")})
cells = {}
for arm in (BASE, "PR"):
    for r in (1, 2, 3):
        with open(f"{d}/gen-{chip}-{arm}-r{r}.jsonl") as f:
            for line in f:
                x = json.loads(line)
                if x["k"] == "cell":
                    cells.setdefault((x["dtype"], x["ori"], x["m"], x["n"], x["kdim"]), {})[(arm, r)] = (x["median_us"], x["sha"], x["tflops"])
n_row = n_ctl = 0
regress, nobenefit, void, badhash, ctl_out = [], [], [], [], []
for key in sorted(cells):
    dt, ori, m, n, k = key
    v = cells[key]
    if any((a, r) not in v for a in (BASE, "PR") for r in (1, 2, 3)):
        print("incomplete", key)
        continue
    g = [100 * (v[(BASE, r)][0] / v[("PR", r)][0] - 1) for r in (1, 2, 3)]
    base_t = [v[(BASE, r)][0] for r in (1, 2, 3)]
    sp = 100 * (max(base_t) / min(base_t) - 1)
    heq = all(v[(BASE, r)][1] == v[("PR", r)][1] for r in (1, 2, 3))
    fl = FLOOR.get((dt, ori))
    new_region = fl is not None and m >= fl and m < OLD.get((dt, ori), 10**9)
    if not heq:
        badhash.append(key)
    tag = "control"
    if new_region:
        tag = "row"
        n_row += 1
        if sp > 5:
            void.append(key)
            tag += " VOID"
        elif sum(1 for x in g if x < -3) >= 2:
            regress.append((key, [round(x, 1) for x in g]))
            tag += " REGRESSION"
        elif all(x < 3 for x in g):
            nobenefit.append((key, [round(x, 1) for x in g]))
            tag += " no-benefit"
    else:
        n_ctl += 1
        if sp > 5:
            void.append(key)
            tag += " VOID"
        elif not all(abs(x) < 3 for x in g):
            ctl_out.append((key, [round(x, 1) for x in g]))
            tag += " OUT"
    print(f"{dt:4s} {ori} m={m:5d} n={n:5d} k={k:5d} gain% " + " ".join(f"{x:+6.1f}" for x in g)
          + f"  {BASE} spread {sp:4.1f}%  hash_eq {heq}  {BASE} {v[(BASE, 1)][2]:.2f} TFLOP/s  {tag}")
print(f"base arm {BASE}: row cells {n_row}, control cells {n_ctl}")
print("hash mismatches:", badhash or "none")
print("REGRESSION cells:", regress or "none")
print("no-benefit row cells:", nobenefit or "none")
print("control cells outside 3 %:", ctl_out or "none")
print("void cells:", void or "none")
