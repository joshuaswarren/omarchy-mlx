#!/usr/bin/env python3
"""PAR-99: per-kernel GPU time + achieved bandwidth for the 4B decode window
on jw16 Linux (T6001 G14X 16-inch M1 Max, LPDDR5 peak 400 GB/s).

Same attribution as PAR-97 (dispatches whose submit sequence queued inside the
marker window), extended with per-kernel bytes: each k:d record carries the
binding list; a kernel's bytes = sum of the referenced binding ranges.
Bandwidth = bytes / kernel-ns.
"""
import json
import re
import sys

prof_path, marker_path, compute_h = sys.argv[1], sys.argv[2], sys.argv[3]

names = []
inside = False
for line in open(compute_h):
    if "enum class ComputeKernel" in line:
        inside = True
        continue
    if inside:
        m = re.match(r"^\s{2}(\w+),\s*(?://.*)?$", line)
        if m:
            names.append(m.group(1))
        elif "};" in line:
            break

markers = [json.loads(l) for l in open(marker_path) if l.strip()]
ds = [m["t"] for m in markers if m.get("p") == "decode_start"]
dd = [m["t"] for m in markers if m.get("p") == "decode_done"]
if not ds or not dd:
    print("NO_MARKER decode", [m.get("p") for m in markers])
    sys.exit(3)
lo, hi = ds[0], dd[0]

queue = {}
bindings = {}  # seq -> binding list (shared by all dispatches of that submit)
agg = {}
total = 0
nd = 0
bytes_total = 0
for line in open(prof_path, errors="ignore"):
    if '"k":"q"' in line:
        try:
            r = json.loads(line)
        except Exception:
            continue
        queue[r["s"]] = r["queue_t0"]
        continue
    if '"k":"b"' in line:
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("s") is not None:
            bindings[r["s"]] = r.get("b", [])
        continue
    if '"k":"d"' not in line:
        continue
    try:
        r = json.loads(line)
    except Exception:
        continue
    q0 = queue.get(r.get("s"))
    if q0 is None:
        continue
    if not (lo < q0 <= hi):
        continue
    key = (names[r["e"]] if r["e"] < len(names) else f"kernel_{r['e']}", r.get("op", "?"))
    a = agg.setdefault(key, [0, 0, 0])  # ns, count, bytes
    ns = r["t1"] - r["t0"]
    a[0] += ns
    a[1] += 1
    b = 0
    for bd in r.get("b", []) or []:
        rng = bd[2] if len(bd) > 2 else 0
        try:
            b += int(rng)
        except (TypeError, ValueError):
            pass
    a[2] += b
    total += ns
    bytes_total += b
    nd += 1

PEAK_GBS = 400.0  # T6001 16-inch M1 Max LPDDR5
print(f"# PAR-99 per-kernel GPU time + achieved bandwidth, 4B decode window, jw16 Linux (T6001 G14X)")
print("# window_ns=(%d,%d] dispatches=%d total_kernel_ns=%d bytes_moved=%d" % (lo, hi, nd, total, bytes_total))
print(f"# kernel\top\tcount\ttotal_us\tshare_pct\tbytes\tachieved_GBs\tof_peak_pct")
for (k, op), (ns, c, byt) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:12]:
    share = 100.0 * ns / total if total else 0.0
    gbs = (byt / ns) if ns else 0.0  # bytes per ns = GB/s
    print(f"{k}\t{op}\t{c}\t{ns}\t{share:.2f}\t{byt}\t{gbs:.1f}\t{100.0 * gbs / PEAK_GBS:.1f}")
