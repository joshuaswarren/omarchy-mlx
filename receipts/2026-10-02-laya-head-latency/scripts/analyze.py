#!/usr/bin/env python3
"""Analyze LayaHead2 probe NDJSON and compare raw answer files.

pct    per-(mode, arm) percentiles of wall_ms plus per-phase splits; a call
       counts as valid only if its host record shows the expected boot,
       uptime >= 360 s, load1 < 0.5, PSI cpu some avg10 == 0, and zero other
       python/mlx processes. Invalid calls are counted, never averaged in.
cmp    route-level equality of two dev_sweep raw answer files under frozen
       policy 3 (decide_route), with a bit-level diff count as information.
"""
import argparse
import collections
import json
import sys
from pathlib import Path


def pct(values, q):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(len(s) * q) - 1 if q > 0 else 0)], 1)


def host_ok(h, boot, allow_others=False):
    if h["boot_id"] != boot:
        return "boot"
    if h["uptime_s"] < 360:
        return "uptime"
    load1 = float(h["loadavg"].split()[0])
    if load1 >= 0.5:
        return "load"
    psi = h["psi_cpu"]["some"]
    if psi and float(psi.get("avg10", 1)) != 0:
        return "psi"
    if h["other_python_mlx_procs"] != 0 and not allow_others:
        return "others"
    return None


def cmd_pct(args):
    recs = [json.loads(l) for l in Path(args.ndjson).read_text().splitlines() if l.strip()]
    groups = collections.defaultdict(list)
    for r in recs:
        groups[(r["mode"], r["arm"])].append(r)
    for key in sorted(groups):
        rows = groups[key]
        boot = collections.Counter(
            r["host"]["boot_id"] for r in rows).most_common(1)[0][0]
        bad = collections.Counter()
        ok = []
        for r in rows:
            why = host_ok(r["host"], boot, args.allow_others) if boot else "mixed_boots"
            if why:
                bad[why] += 1
            else:
                ok.append(r)
        print(f"== {key[0]} arm={key[1]}: n={len(rows)} valid={len(ok)} "
              f"invalid={dict(bad) or '{}'} boot={boot[:8]}")
        walls = [r["wall_ms"] for r in ok]
        print(f"   wall ms p50={pct(walls,0.5)} p95={pct(walls,0.95)} "
              f"min={min(walls) if walls else None} max={max(walls) if walls else None}")
        if key[0] == "phases":
            for ph in ("tokenize_ms", "collate_ms", "arrays_ms", "record_ms",
                       "eval_ms", "readback_ms", "post_ms"):
                v = [r[ph] for r in ok]
                print(f"   {ph:12s} p50={pct(v,0.5)} p95={pct(v,0.95)}")
            lens = [r["seq_len"] for r in ok]
            toks = [r["n_tokens"] for r in ok]
            print(f"   seq_len min={min(lens)} max={max(lens)}; n_tokens max={max(toks)}")
        else:
            for f in ("server_prompt_ms", "server_predicted_ms"):
                v = [r[f] for r in ok if r[f] is not None]
                print(f"   {f:20s} p50={pct(v,0.5)} p95={pct(v,0.95)}")
            res = [round(r["wall_ms"] - (r["server_prompt_ms"] or 0)
                         - (r["server_predicted_ms"] or 0), 1) for r in ok]
            print(f"   residual_ms         p50={pct(res,0.5)} p95={pct(res,0.95)}")
            fails = sum(1 for r in ok if r["failed"])
            print(f"   failed={fails} (of valid)")


def load_raw(path):
    doc = json.loads(Path(path).read_text())
    recs = doc["records"] if isinstance(doc, dict) and "records" in doc else doc
    return recs


def cmd_cmp(args):
    sys.path.insert(0, str(Path(args.serve).resolve()))
    from mlx_omarchy_assistant.routing import ROUTING_POLICY, decide_route

    a, b = load_raw(args.raw_a), load_raw(args.raw_b)
    by_id_a = {r["id"]: r for r in a}
    by_id_b = {r["id"]: r for r in b}
    assert by_id_a.keys() == by_id_b.keys(), "case id sets differ"
    diff = 0
    bitdiff = 0
    for cid in sorted(by_id_a):
        ra, rb = by_id_a[cid], by_id_b[cid]
        answer = lambda r: {"choice": r["choice"],
                            "probabilities": r["probabilities"],
                            "rl_agent": {"act_probability": r["act_probability"]}}
        route_a = decide_route(answer(ra), ROUTING_POLICY)
        route_b = decide_route(answer(rb), ROUTING_POLICY)
        if route_a != route_b:
            diff += 1
            print(f"  ROUTE DIFF id={cid}: {route_a} vs {route_b}")
        if (ra["probabilities"] != rb["probabilities"]
                or ra["choice"] != rb["choice"]
                or ra["act_probability"] != rb["act_probability"]):
            bitdiff += 1
    print(f"cases={len(by_id_a)} route_diffs={diff} bit_diffs={bitdiff}")
    sys.exit(1 if diff else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pct")
    p.add_argument("ndjson")
    p.add_argument("--allow-others", action="store_true",
                   help="screening only: count runs with foreign python/mlx "
                        "processes as valid (still reported per record)")
    c = sub.add_parser("cmp")
    c.add_argument("raw_a")
    c.add_argument("raw_b")
    c.add_argument("--serve", default=str(Path(__file__).resolve().parents[3] / "serve"))
    args = ap.parse_args()
    {"pct": cmd_pct, "cmp": cmd_cmp}[args.cmd](args)


if __name__ == "__main__":
    main()
