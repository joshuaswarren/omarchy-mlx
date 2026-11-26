#!/usr/bin/env python3
"""Per-class prefill-phase attribution from gap census NDJSON files.

Reproduces profile_analyze's phase contract (dispatch belongs to the phase
its SUBMISSION was made in) and joins kernel enum names from compute.h.
Emits per-class per-phase GPU tick totals, dispatch counts, and the giant
dispatch table with binding ranges.
"""
import json
import re
import sys

HEADER = ("overlay/mlx/backend/omarchy/compute.h")


def kernel_names(header_path):
    names = []
    inside = False
    for line in open(header_path):
        if "enum class ComputeKernel" in line:
            inside = True
            continue
        if inside:
            m = re.match(r"^\s{2}(\w+),\s*$", line)
            if m:
                names.append(m.group(1))
            elif names and "}" in line:
                break
    return names


def phase_of(host_ns, markers):
    spans = dict(markers)
    if "prefill_start" in spans and "prefill_done" in spans:
        if spans["prefill_start"] <= host_ns < spans["prefill_done"]:
            return "prefill"
    if "decode_start" in spans and "decode_done" in spans:
        if spans["decode_start"] <= host_ns < spans["decode_done"]:
            return "decode"
    return "other"

def main():
    path = sys.argv[1]
    markers_path = sys.argv[2]
    names = kernel_names(HEADER)
    rows = [json.loads(l) for l in open(path)]
    markers = []
    for l in open(markers_path):
        r = json.loads(l)
        markers.append((r["p"], r["t"]))
    markers.sort(key=lambda x: x[1])

    dispatches = [r for r in rows if r.get("k") == "d"]

    # phase per dispatch via its submission's host clock
    sub_host = {r["s"]: r["t"] for r in rows if r.get("k") == "s"}
    # fall back: use dispatch order vs phase boundaries if host missing
    per_class = {}
    unmatched = 0
    for d in dispatches:
        s = d["s"]
        host = sub_host.get(s)
        if host is None:
            unmatched += 1
            continue
        ph = phase_of(host, markers)
        kn = names[d["e"]] if d["e"] < len(names) else str(d["e"])
        c = per_class.setdefault((kn, ph), [0, 0])
        c[0] += 1
        c[1] += d["t1"] - d["t0"]
    if unmatched:
        print(f"unmatched dispatches (no submit host): {unmatched}")

    print(f"== {path}")
    for ph in ("prefill", "decode"):
        cls = sorted(((v[1], v[0], k[0]) for k, v in per_class.items()
                      if k[1] == ph), reverse=True)
        tot = sum(v[1] for v in cls)
        n = sum(v[0] for v in cls)
        print(f"-- phase={ph} dispatches={n} gpu_busy={tot/1e6:.1f}ms")
        for t, c, kn in cls:
            if t / 1e6 < 0.5:
                continue
            print(f"   {kn:24s} n={c:5d} {t/1e6:9.2f}ms share={100*t/tot:5.1f}%")

    # giant dispatch table for the top-3 classes in prefill
    big = set()
    for (kn, ph), v in per_class.items():
        if ph == "prefill" and v[1] / 1e6 > 50:
            big.add(kn)
    print("-- largest individual prefill dispatches per big class:")
    rows_sel = []
    for d in dispatches:
        s = d["s"]
        host = sub_host.get(s)
        if host is None or phase_of(host, markers) != "prefill":
            continue
        kn = names[d["e"]] if d["e"] < len(names) else str(d["e"])
        if kn in big:
            rows_sel.append((d["t1"] - d["t0"], kn, d))
    rows_sel.sort(reverse=True)
    for t, kn, d in rows_sel[:40]:
        binds = " ".join(f"[{x[2]/1024:.0f}KiB]" for x in d["b"][:4])
        print(f"   {kn:24s} tick_ms={t/1e6:7.2f} gx={d['gx']:5d} "
              f"gy={d['gy']:4d} gz={d['gz']:3d} n={d['n']:9d} {binds}")


if __name__ == "__main__":
    main()
