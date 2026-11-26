#!/usr/bin/env python3
"""Elementwise-chain census over an MLX_OMARCHY_GPU_PROFILE NDJSON stream.

Answers three questions for one decode run:
1. How many GPU dispatches does a decode token issue, split by the tape
   path (dispatches recorded inside eval_compiled_tape, event field
   "tp":1) and the eager path ("tp":0)?
2. What is the per-kernel composition of each path?
3. What elementwise chains actually occur: maximal runs of consecutive
   elementwise-class dispatches where each dispatch writes the buffer
   the next one reads, at the same element count. Reports the chain
   length histogram, per-length dispatch totals, and the distinct op
   sequences.

Ordering: per-dispatch events ("k":"d") are flushed at submission
completion and carry no host timestamp, so each is ordered by the host
timestamp of its submission ("k":"s" events), then by file order within
the submission. Token windows come from the markers file written by
scripts/profile_generate.py.

Usage:
  python3 chain_census.py PROFILE.jsonl --compute-h compute.h \
      [--markers markers.jsonl]
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict


def kernel_names(header_path):
    names = []
    pat = re.compile(r"^\s*([A-Za-z0-9_]+)\s*,?\s*$")
    inside = False
    with open(header_path, encoding="utf-8") as fh:
        for line in fh:
            if "enum class ComputeKernel" in line:
                inside = True
                continue
            if inside:
                line = line.split("//", 1)[0]
                if "}" in line:
                    break
                m = pat.match(line)
                if m and m.group(1) != "Count":
                    names.append(m.group(1))
    return names


ELEMENTWISE_CLASSES = (
    "Elementwise",
    "Cast",
    "CopyGeneral",
    "Select",
    "Compare",
    "LogicalOr",
    "Fill",
)


def is_elementwise(name):
    return name.startswith(ELEMENTWISE_CLASSES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--compute-h", required=True)
    ap.add_argument("--markers")
    ap.add_argument("--detail-tokens", type=int, default=2,
                    help="tokens to dump full dispatch sequences for")
    args = ap.parse_args()

    names = kernel_names(args.compute_h)
    if not names:
        sys.exit("no kernel names parsed from " + args.compute_h)

    # Pass 1: submission host timestamps, in submission order.
    sub_t = {}
    with open(args.profile, encoding="utf-8") as fh:
        for line in fh:
            ev = json.loads(line)
            if ev.get("k") == "s":
                sub_t[ev["s"]] = ev["t"]

    # Pass 2: dispatch events ordered by (submission time, file order).
    events = []
    with open(args.profile, encoding="utf-8") as fh:
        for line in fh:
            ev = json.loads(line)
            if ev.get("k") == "d":
                events.append(ev)
    events.sort(key=lambda e: (sub_t.get(e["s"], 0)))

    # Decode token windows from markers (host monotonic ns).
    token_bounds = []
    if args.markers:
        toks = []
        with open(args.markers, encoding="utf-8") as fh:
            for line in fh:
                ev = json.loads(line)
                if ev["p"] in ("decode_start", "tok", "decode_done"):
                    toks.append(ev)
        starts = [e["t"] for e in toks if e["p"] == "decode_start"]
        tok_ts = [e["t"] for e in toks if e["p"] == "tok"]
        if starts:
            token_bounds.append((0, starts[0], tok_ts[0] if tok_ts else None))
        for i, t in enumerate(tok_ts):
            hi = tok_ts[i + 1] if i + 1 < len(tok_ts) else None
            token_bounds.append((i + 1, t, hi))

    def token_of(ev):
        t = sub_t.get(ev["s"], 0)
        if not token_bounds:
            return None
        for idx, lo, hi in token_bounds:
            if t >= lo and (hi is None or t < hi):
                return idx
        return None

    tok_total = Counter()
    tok_tape = Counter()
    tok_kernel = defaultdict(Counter)
    tok_kernel_tape = defaultdict(Counter)
    total = Counter()
    tape_total = Counter()
    for ev in events:
        k = names[ev["e"]] if ev["e"] < len(names) else f"K{ev['e']}"
        tp = ev.get("tp", 0)
        total[k] += 1
        if tp:
            tape_total[k] += 1
        t = token_of(ev)
        if t is not None:
            tok_total[t] += 1
            tok_kernel[t][k] += 1
            if tp:
                tok_tape[t] += 1
                tok_kernel_tape[t][k] += 1

    def counter_block(title, counter, grand):
        print(f"\n{title}")
        for k, n in counter.most_common():
            print(f"  {k:28s} {n:7d}  {100.0 * n / max(grand, 1):5.1f}%")

    print(f"dispatch events parsed: {len(events)}")
    print(f"decode tokens: {len(tok_total)}")
    if tok_total:
        counts = sorted(tok_total.values())
        print(f"dispatches/token: min={counts[0]} median="
              f"{counts[len(counts) // 2]} max={counts[-1]}")
        tap = sorted(tok_tape.values())
        if tap:
            print(f"tape-path dispatches/token: min={tap[0]} median="
                  f"{tap[len(tap) // 2]} max={tap[-1]}")

    counter_block("== whole-run by kernel ==", total, sum(total.values()))
    counter_block("== whole-run TAPE-path by kernel ==", tape_total,
                  sum(total.values()))
    print("\n== median decode token: kernel composition ==")
    if tok_kernel:
        med = sorted(tok_total, key=lambda t: tok_total[t])[
            len(tok_total) // 2]
        print(f"(token {med}: {tok_total[med]} dispatches,"
              f" {tok_tape[med]} tape-path)")
        counter_block("-- all paths --", tok_kernel[med], tok_total[med])
        counter_block("-- tape path --", tok_kernel_tape[med],
                      tok_total[med])

    # Chain census within tokens: runs of consecutive elementwise
    # dispatches linked by buffer flow (writer -> reader, same count).
    def chains_for(seq):
        chains = []
        run = []
        for ev in seq:
            k = names[ev["e"]] if ev["e"] < len(names) else f"K{ev['e']}"
            if not is_elementwise(k):
                if len(run) >= 2:
                    chains.append(run)
                run = []
                continue
            linked = False
            if run:
                prev = run[-1]
                pbs = prev.get("b", [])
                cbs = ev.get("b", [])
                if prev["n"] == ev["n"] and len(pbs) >= 3 and cbs:
                    out_binding = pbs[2]
                    linked = any(c == out_binding for c in cbs[:2])
                if not linked:
                    if len(run) >= 2:
                        chains.append(run)
                    run = []
            run.append(ev)
        if len(run) >= 2:
            chains.append(run)
        return chains

    tok_chains = defaultdict(list)
    for t in sorted(tok_total):
        tok_chains[t] = chains_for(
            [ev for ev in events if token_of(ev) == t])

    hist = Counter()
    disp_in_chains = Counter()
    op_seqs = Counter()
    for chains in tok_chains.values():
        for run in chains:
            hist[len(run)] += 1
            disp_in_chains[len(run)] += len(run)
            sig = "+".join(
                f"{names[e['e']] if e['e'] < len(names) else '?'}"
                f"/op{e['op']}" for e in run)
            op_seqs[sig] += 1
    print("\n== chain census, decode tokens, all paths ==")
    print(f"chains>=2: {sum(hist.values())}, dispatches inside chains:"
          f" {sum(disp_in_chains.values())}")
    for length in sorted(hist):
        print(f"  len={length:2d}: count={hist[length]:5d} dispatches="
              f"{disp_in_chains[length]:5d}")
    for sig, n in op_seqs.most_common(15):
        print(f"  x{n:4d} {sig[:150]}")

    for t in sorted(tok_total)[: args.detail_tokens]:
        print(f"\n== token {t} dispatch sequence ==")
        for ev in [e for e in events if token_of(e) == t]:
            k = names[ev["e"]] if ev["e"] < len(names) else f"K{ev['e']}"
            print(f"  {'T' if ev.get('tp', 0) else 'e'} {k:24s} op="
                  f"{ev['op']:3d} n={ev['n']:6d}")


if __name__ == "__main__":
    main()
