"""H27 analysis (amendment 2 rules): C2 vs C6 per cell over three rounds. usage: h27_an.py <dir> <chip g13g|g13c|g14c>"""
import json
import sys

d, chip = sys.argv[1], sys.argv[2]
REGION = {  # (dtype, ori) -> measured m values where C6 has the row and C2 does not
    "g13g": {("f16", "nt"): (128, 256), ("f16", "tn"): (1024, 2048), ("bf16", "tn"): (512, 1024, 2048),
             ("f32", "nt"): (512, 1024, 2048), ("f32", "nn"): (512, 1024, 2048), ("f32", "tn"): (512, 1024, 2048)},
    "g14c": {("f16", "nt"): (1024, 2048, 4096), ("bf16", "nt"): (512, 1024, 2048, 4096), ("bf16", "tn"): (4096,)},
    "g13c": {},
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


check_stamps({arm: [f"{d}/h27-{chip}-{arm}-r{r}.jsonl" for r in (1, 2, 3)] for arm in ("C2", "C6")})
cells = {}
for arm in ("C2", "C6"):
    for r in (1, 2, 3):
        with open(f"{d}/h27-{chip}-{arm}-r{r}.jsonl") as f:
            for line in f:
                x = json.loads(line)
                if x["k"] == "cell":
                    cells.setdefault((x["dtype"], x["ori"], x["m"]), {})[(arm, r)] = (x["median_us"], x["sha"])
bad_hash, ctl_bad, rows = [], [], {}
for key, v in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
    dt, ori, m = key
    gains = [100 * (v[("C2", r)][0] / v[("C6", r)][0] - 1) for r in (1, 2, 3)]
    c2 = [v[("C2", r)][0] for r in (1, 2, 3)]
    spread = 100 * (max(c2) / min(c2) - 1)
    heq = all(v[("C2", r)][1] == v[("C6", r)][1] for r in (1, 2, 3))
    if not heq:
        bad_hash.append(key)
    in_region = m in REGION.get((dt, ori), ())
    void = spread > 5
    ok = (not void) and heq and all(g >= 5 for g in gains)
    ctl = (not in_region) and all(abs(g) < 3 for g in gains)
    if in_region:
        rows.setdefault((dt, ori), []).append((m, ok))
    elif not ctl:
        ctl_bad.append((key, [round(g, 1) for g in gains]))
    tag = "REGION " + ("PASS" if ok else "FAIL") if in_region else ("control ok" if ctl else "CONTROL OUT")
    print(f"{dt:4s} {ori} m={m:5d} gains% " + " ".join(f"{g:+6.1f}" for g in gains) + f"  C2 spread {spread:4.1f}%  hash_eq {heq}  {tag}{' VOID' if void else ''}")
print("hash mismatches:", bad_hash or "none")
print("control cells outside 3 %:", ctl_bad or "none")
for (dt, ori), res in rows.items():
    res.sort()
    floor = None
    for i in range(len(res)):
        if all(ok for _, ok in res[i:]):
            floor = res[i][0]
            break
    print(f"row {dt} {ori}: region {[m for m, _ in res]} passes {[ok for _, ok in res]} -> new floor {floor if floor is not None else 'DROP (keep old floor)'}")
