#!/usr/bin/env python3
"""DispatchFuse W1 census: per-token decode dispatch classes from
MLX_OMARCHY_GPU_PROFILE NDJSON streams.

Usage: analyze_w1.py <dir-with-prof-*.ndjson> <repo-compute.h> [model-tag ...]

Outputs per stream:
  - total dispatches, submissions
  - per-token dispatch count via d512-vs-d64 slope (same protocol)
  - class table for the steady tail (count/token, us/token, bar%)
  - the in-order single-token dispatch sequence (last steady token)
  - KV_DIRECT on/off dispatch delta at d64
"""
import json, re, sys, glob, hashlib
from collections import Counter, defaultdict

def kernel_names(compute_h):
    src = open(compute_h).read()
    m = re.search(r"enum class ComputeKernel[^{]*\{(.*?)\n\};", src, re.S)
    body = m.group(1)
    names = []
    for line in body.splitlines():
        line = line.split("//")[0].strip().rstrip(",")
        if not line or "=" in line:
            continue
        names.append(line)
    return names

def load(path):
    ev = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{"k":"d"'):
                continue
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # truncated tail line
    return ev

def classes(events, per):
    tail = events[-per * 12:] if per else events
    cls = defaultdict(lambda: [0, 0.0, [], 0])
    order = []
    for r in tail:
        ns = (r.get("t1", 0) - r.get("t0", 0))
        key = (r["p"], r["e"], r["gx"], r["gy"], r["gz"], len(r["b"]))
        c = cls[key]
        c[0] += 1; c[1] += ns / 1000.0; c[2].append(ns / 1000.0); c[3] += 1 if r["bar"] else 0
        order.append(key)
    return cls, order

def main():
    d, compute_h = sys.argv[1], sys.argv[2]
    names = kernel_names(compute_h)
    streams = {}
    for p in sorted(glob.glob(f"{d}/prof-*.ndjson")):
        tag = p.split("prof-")[1].replace(".ndjson", "")
        streams[tag] = load(p)
    for tag, ev in streams.items():
        print(f"\n== {tag}: dispatches={len(ev)} subs={len(set(r['s'] for r in ev))}")

    # per-token slope from matching d512/d64 pairs
    for model in ("2b", "4b", "9b"):
        a, b = streams.get(f"{model}-d64"), streams.get(f"{model}-d512")
        if a and b:
            per = (len(b) - len(a)) / (512 - 64)
            print(f"\n== {model}: per-token dispatches (slope) = {per:.2f}")
            cls, order = classes(b, int(round(per)))
            tot = sum(c[0] for c in cls.values())
            print(f"   class table (tail {int(round(per))*12} dispatches, sum/tok={tot/12:.1f}):")
            for key, (cnt, tot_us, times, nb) in sorted(cls.items(), key=lambda kv: -kv[1][1]):
                times.sort()
                kn = names[key[1]] if key[1] < len(names) else f"enum{key[1]}"
                print(f"   {cnt/12:6.1f}/tok {tot_us/12:8.1f}us/tok med={times[len(times)//2]:6.1f}us "
                      f"bar={100*nb/cnt:3.0f}%  {key[0]} | {kn} gx={key[2]},{key[3]},{key[4]} binds={key[5]}")
            # periodicity: block signatures of one token each
            pt = int(round(per))
            sigs = []
            for i in range(0, len(order) - pt + 1, pt):
                sigs.append(hashlib.md5(repr(order[i:i+pt]).encode()).hexdigest()[:8])
            print(f"   block sigs ({pt} each): {sigs}")
            # single-token in-order listing (last complete token)
            print(f"   == last-token in-order sequence (first 90 of {pt}):")
            for key in order[-pt:][:90]:
                kn = names[key[1]] if key[1] < len(names) else f"enum{key[1]}"
                print(f"     {key[0]} | {kn} gx={key[2]}")

    # KV_DIRECT delta at d64
    for model in ("2b", "4b", "9b"):
        on, off = streams.get(f"{model}-d64"), streams.get(f"{model}-d64-kvoff")
        if on and off:
            print(f"\n== {model} KV_DIRECT delta d64: on={len(on)} off={len(off)} delta={len(off)-len(on)} "
                  f"({(len(off)-len(on))/64:.2f}/decode-token incl warmup+prefill diff)")

if __name__ == "__main__":
    main()
