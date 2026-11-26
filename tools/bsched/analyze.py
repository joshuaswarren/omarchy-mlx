#!/usr/bin/env python3
"""Analyze BarrierSched census cells and A/B pairs.

census: JSON cell files -> per-token dispatch/barrier numbers via the depth
slope (d512 - d64 over 448 tokens), wave off vs on.
ab: pairs.jsonl -> paired deltas per (depth, prefill), medians, min-max,
disjointness verdict against the pre-registered +1.5% landing bar.
"""
import glob
import json
import os
import statistics
import sys


def _rate(d, key):
    v = d.get(key)
    if isinstance(v, dict):
        return v.get("median")
    if isinstance(v, list) and v:
        vals = sorted(x.get("pure_prefill_tok_rate", 0) for x in v)
        return vals[len(vals) // 2]
    return v


def load_cells(census_dir):
    cells = []
    for path in sorted(glob.glob(os.path.join(census_dir, "*.json"))):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if "trace_delta" not in d:
            continue
        proto = d.get("protocol") or {}
        cells.append(
            {
                "label": d.get("label") or os.path.basename(path),
                "wave": d.get("wave"),
                "depth": proto.get("new_tokens"),
                "prefill": proto.get("prefill_tokens"),
                "passes": proto.get("passes", 1),
                "digest": d.get("ordered_records_sha256"),
                "tps": _rate(d, "decode_tok_rate"),
                "pf_rate": _rate(d, "pure_prefill"),
                "dispatches": d["trace_delta"]["vk_compute_dispatches"],
                "barriers": d["trace_delta"]["barriers_emitted"],
                "skipped": d["trace_delta"]["barriers_skipped"],
                "copies": d["trace_delta"]["vk_buffer_copies"],
                "fills": d["trace_delta"]["vk_buffer_fills"],
                "submissions": d["trace_delta"]["vk_submissions"],
            }
        )
    return cells


def per_token(cells, wave):
    """Slope (max depth - min depth) per-token counts among decode cells."""
    dec = [
        c
        for c in cells
        if c["wave"] == wave and (c["depth"] or 0) > 1 and c["tps"]
    ]
    if len(dec) < 2:
        return None
    dec.sort(key=lambda c: c["depth"])
    lo, hi = dec[0], dec[-1]
    n = hi["depth"] - lo["depth"]
    if n <= 0:
        return None
    out = {"wave": wave, "depth_lo": lo["depth"], "depth_hi": hi["depth"]}
    for k in ("dispatches", "barriers", "skipped", "copies", "fills", "submissions"):
        out[f"{k}_per_token"] = round((hi[k] - lo[k]) / n, 2)
    return out


def census_report(census_dir):
    cells = load_cells(census_dir)
    if not cells:
        print(f"no census cells in {census_dir}")
        return
    digests = {}
    for c in cells:
        key = (c["depth"], c["prefill"], c["passes"])
        digests.setdefault(key, set()).add(c["digest"])
    print("== cells ==")
    for c in cells:
        print(
            f"{c['label']:>12} wave={c['wave']} depth={c['depth']} "
            f"prefill={c['prefill']} tps={c['tps']} "
            f"disp={c['dispatches']} bar={c['barriers']} skip={c['skipped']} "
            f"subs={c['submissions']} digest={c['digest']}"
        )
    bad = {k: v for k, v in digests.items() if len(v) > 1}
    if bad:
        print("!! DIGEST MISMATCH across wave arms:", bad)
    else:
        print("digests: identical across wave arms for every (depth,prefill,passes)")
    for wave in (0, 1):
        pt = per_token(cells, wave)
        if pt:
            print(f"== per-token wave={wave} ==", json.dumps(pt))
    # prefill cells: absolute per-pass barriers at new-tokens=1
    pf = [c for c in cells if c["depth"] == 1 and c["passes"]]
    for c in pf:
        passes = c["passes"] or 1
        print(
            f"== prefill wave={c['wave']} p{c['prefill']} per-pass: "
            f"dispatches={c['dispatches'] / passes:.1f} "
            f"barriers={c['barriers'] / passes:.1f} "
            f"skipped={c['skipped'] / passes:.1f} =="
        )


def ab_report(pairs_path):
    import collections

    pairs = collections.defaultdict(list)
    for line in open(pairs_path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        d = json.loads(line)
        pairs[(d["depth"], d["prefill"])].append(d)
    print("== A/B paired deltas (cand vs ctl) ==")
    verdict = []
    for key in sorted(pairs):
        ds = pairs[key]
        for d in ds:
            assert d.get("ctl_digest") == d.get("cand_digest"), (
                f"digest mismatch in pair: {d}")
        deltas = [d["delta_pct"] for d in ds]
        ctl = [d["ctl_tps"] for d in ds]
        cand = [d["cand_tps"] for d in ds]
        disjoint = max(ctl) < min(cand) or max(cand) < min(ctl)
        med = statistics.median(deltas)
        print(
            f"depth={key[0]} prefill={key[1]} n={len(ds)} "
            f"delta_med={med:+.2f}% range=[{min(deltas):+.2f}, "
            f"{max(deltas):+.2f}] ctl=[{min(ctl):.2f},{max(ctl):.2f}] "
            f"cand=[{min(cand):.2f},{max(cand):.2f}] disjoint={disjoint} "
            f"digest={ds[0]['digest']}"
        )
        verdict.append((key, med, disjoint, len(ds)))
    print("== verdict vs +1.5% bar ==")
    for (depth, prefill), med, disjoint, n in verdict:
        print(
            f"depth={depth} prefill={prefill}: med={med:+.2f}% n={n} "
            f"disjoint={disjoint} -> "
            f"{'PASS-candidate' if med >= 1.5 and disjoint and n >= 5 else 'below-bar'}"
        )


if __name__ == "__main__":
    if sys.argv[1] == "census":
        census_report(sys.argv[2])
    elif sys.argv[1] == "ab":
        ab_report(sys.argv[2])
    else:
        raise SystemExit("usage: analyze.py census DIR | ab PAIRS.jsonl")
