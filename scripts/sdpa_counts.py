#!/usr/bin/env python3
"""Per-token dispatch counts for the SDPA rework acceptance.

Reads one MLX_OMARCHY_GPU_PROFILE stream plus the profile_generate
markers and prints the median decode token's counts for the kernel
classes the SDPA census tracks: CopyGeneralF16 per token and per layer
(24 layers in the pinned model), the three cast classes, and the
attention matmul/softmax classes. Same ordering rule as
scripts/chain_census.py (submission host timestamp, then file order).

Usage:
  python3 scripts/sdpa_counts.py PROFILE.jsonl --compute-h compute.h \
      --markers markers.jsonl [--layers 24]
"""

import argparse
import json
from collections import Counter, defaultdict


def kernel_names(header_path):
    names = []
    pattern = None
    inside = False
    import re
    pattern = re.compile(r"^\s*([A-Za-z0-9_]+),?\s*$")
    for line in open(header_path, encoding="utf-8"):
        if "enum class ComputeKernel" in line:
            inside = True
            continue
        if inside:
            line = line.split("//", 1)[0]
            if "}" in line:
                break
            m = pattern.match(line)
            if m and m.group(1) != "Count":
                names.append(m.group(1))
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--compute-h", required=True)
    ap.add_argument("--markers", required=True)
    ap.add_argument("--layers", type=int, default=24)
    args = ap.parse_args()

    names = kernel_names(args.compute_h)
    sub_t = {}
    for line in open(args.profile, encoding="utf-8"):
        ev = json.loads(line)
        if ev.get("k") == "s":
            sub_t[ev["s"]] = ev["t"]

    events = []
    for line in open(args.profile, encoding="utf-8"):
        ev = json.loads(line)
        if ev.get("k") == "d":
            events.append(ev)
    events.sort(key=lambda e: sub_t.get(e["s"], 0))

    starts, toks = [], []
    for line in open(args.markers, encoding="utf-8"):
        ev = json.loads(line)
        if ev["p"] == "decode_start":
            starts.append(ev["t"])
        elif ev["p"] == "tok":
            toks.append(ev["t"])
    bounds = []
    if starts:
        bounds.append((starts[0], toks[0] if toks else None))
    for i, t in enumerate(toks):
        hi = toks[i + 1] if i + 1 < len(toks) else None
        bounds.append((t, hi))

    def token_of(ev):
        t = sub_t.get(ev["s"], 0)
        for idx, (lo, hi) in enumerate(bounds):
            if t >= lo and (hi is None or t < hi):
                return idx
        return None

    per_token = defaultdict(Counter)
    for ev in events:
        t = token_of(ev)
        if t is None:
            continue
        k = names[ev["e"]] if ev["e"] < len(names) else f"K{ev['e']}"
        per_token[t][k] += 1

    if not per_token:
        raise SystemExit("no decode-token dispatches found")
    counts = sorted(per_token, key=lambda t: sum(per_token[t].values()))
    med = counts[len(counts) // 2]
    c = per_token[med]
    total = sum(c.values())
    print(f"decode tokens: {len(per_token)}; median token dispatches: {total}")
    tracked = [
        "CopyGeneralF16",
        "CastF32F16",
        "CastF16F32",
        "CastI32F32",
        "MatmulF32",
        "MatmulF16",
        "SoftmaxF32",
        "SoftmaxF16",
        "ElementwiseF32",
        "ElementwiseF16",
    ]
    for k in tracked:
        n = c.get(k, 0)
        extra = ""
        if k == "CopyGeneralF16":
            extra = f"  (= {n / args.layers:g} per layer at {args.layers} layers)"
        print(f"  {k:20s} {n:6d}{extra}")
    casts = sum(c.get(k, 0) for k in
                ("CastF32F16", "CastF16F32", "CastI32F32"))
    print(f"  {'casts total':20s} {casts:6d}")


if __name__ == "__main__":
    main()
