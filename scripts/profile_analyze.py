#!/usr/bin/env python3
"""Analyze an MLX_OMARCHY_GPU_PROFILE NDJSON stream.

Usage:
  python3 profile_analyze.py PROFILE.jsonl [--markers markers.jsonl]
      [--compute-h header-path]

Produces: GPU busy fraction, gap distribution (intra- vs inter-submission),
per-kernel totals/mean/median/share, dispatches per submission, host-side
join/submit/record costs, and the dependency proxy (fraction of consecutive
dispatch pairs with disjoint compute bindings - a lower-bound PROXY
only: transfer/fill/alias dependencies are invisible to it, so it says
nothing about which barriers are actually removable).

Kernel names come from the declaration order of the ComputeKernel enum in
overlay/mlx/backend/omarchy/compute.h, the same order the C++ harness
records. GPU ticks convert with meta.period_ns; unwrapping uses
meta.valid_bits. Phases come from an optional markers file written by
scripts/profile_generate.py (CLOCK_MONOTONIC ns, same clock as the harness
host timestamps).

Phase attribution contract: dispatch ("k":"d") records are written
DELAYED - when their ring slot is flushed at a join or slot reuse, long
after the work was submitted - so file order and the "op" field never
attribute anything. A dispatch belongs to the phase its SUBMISSION was
made in: look up the submit record ("k":"s") with the same id d["s"]
and canonicalize the last marker at that host time. Two unknowables
are kept distinct: "unknown (no matching submit record)" and "unknown
(submit outside named phases)". Neither is folded into a named phase
or reported as zero-cost work. Host markers are WINDOWS, not semantic
phases: an async pipeline can submit the next decode step before the
first token yields, so boundary work lands in the earlier window and
decode-window rates use inter-token intervals (n_tok - 1).
"""

import argparse
import json
import re
import sys


def parse_kernel_names(header_path):
    names = []
    pattern = re.compile(r"^\s{2}(\w+),\s*(?://.*)?$")
    inside = False
    with open(header_path, "r", encoding="utf-8") as f:
        for line in f:
            if "enum class ComputeKernel" in line:
                inside = True
                continue
            if inside:
                m = pattern.match(line)
                if m:
                    names.append(m.group(1))
                elif "};" in line:
                    break
    return names


def unwrap(ticks, valid_bits):
    """Unwrap device ticks into a monotonically increasing series."""
    if not ticks:
        return []
    if valid_bits >= 64 or valid_bits == 0:
        return list(ticks)
    m = 1 << valid_bits
    half = m >> 1
    out = [ticks[0]]
    delta = 0
    for prev_raw, raw in zip(ticks, ticks[1:]):
        d = (raw - prev_raw) % m
        if d > half:
            d -= m
        delta += d
        out.append(ticks[0] + delta)
    return out


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    i = min(len(sorted_vals) - 1, int(p * (len(sorted_vals) - 1) + 0.5))
    return sorted_vals[i]


def fmt_ns(v):
    if v >= 1e6:
        return f"{v / 1e6:.3f}ms"
    if v >= 1e3:
        return f"{v / 1e3:.1f}us"
    return f"{v:.0f}ns"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--markers")
    ap.add_argument("--compute-h", default=None)
    args = ap.parse_args()

    meta = None
    end_rec = None
    dispatches = []
    joins = []
    submits = []
    lineno = 0
    with open(args.profile, "r", encoding="utf-8") as f:
        for line in f:
            lineno += 1
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"malformed profile line {lineno}: {exc}",
                      file=sys.stderr)
                sys.exit(2)
            kind = rec.get("k")
            if kind == "meta":
                meta = rec
            elif kind == "d":
                dispatches.append(rec)
            elif kind == "j":
                joins.append(rec)
            elif kind == "s":
                submits.append(rec)
            elif kind == "end":
                end_rec = rec

    if meta is None:
        print("no meta line; not a profile file", file=sys.stderr)
        sys.exit(1)
    for field in ("period_ns", "valid_bits"):
        if not isinstance(meta.get(field), (int, float)):
            print(f"malformed meta: {field} missing or not numeric",
                  file=sys.stderr)
            sys.exit(2)

    names = {}
    if args.compute_h:
        for i, n in enumerate(parse_kernel_names(args.compute_h)):
            names[i] = n

    def kname(e):
        return names.get(e, f"kernel_{e}")

    period = meta["period_ns"]
    valid_bits = meta["valid_bits"]
    has_gpu = any("t0" in d for d in dispatches)

    markers = []
    if args.markers:
        with open(args.markers, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    markers.append(json.loads(line))

    submit_by_s = {}
    for s in submits:
        submit_by_s[s["s"]] = s

    # Marker names canonicalize to the phase whose work they bracket.
    # tok is the per-token decode marker; the *_done markers are phase
    # boundaries where no new work is submitted.
    PHASE_OF_MARKER = {
        "load_start": "load",
        "prefill_start": "prefill",
        "decode_start": "decode",
        "tok": "decode",
        "decode": "decode",
    }

    def phase_of_host(t):
        """Canonical phase of a host timestamp, or None if unknowable."""
        if not markers:
            return None
        cur = markers[0]["p"]
        for m in markers:
            if m["t"] > t:
                break
            cur = m["p"]
        return PHASE_OF_MARKER.get(cur)

    UNKNOWN_NOSUBMIT = "unknown (no matching submit record)"
    UNKNOWN_OUTSIDE = "unknown (submit outside named phases)"
    # Dispatch records are flushed late (join or slot reuse), so a
    # dispatch's phase is the phase of its SUBMISSION, resolved by the
    # submission id d["s"] against the submit record's host time. The
    # join's host time would attribute work to whenever the host got
    # around to waiting, not to when the work was submitted. Two
    # distinct unknowables are kept distinct: no submit record at all
    # vs a known submit time that falls outside the named phase
    # markers. Neither is guessed.
    for d in dispatches:
        s_rec = submit_by_s.get(d["s"])
        if s_rec is None:
            d["phase"] = UNKNOWN_NOSUBMIT
        else:
            ph = phase_of_host(s_rec["t"])
            d["phase"] = ph if ph else UNKNOWN_OUTSIDE

    out = []

    def say(s):
        out.append(s)

    say(f"== profile {args.profile}")
    say(f"   device={meta['device']} period={period}ns "
        f"valid_bits={valid_bits}")
    say(f"   dispatches={len(dispatches)} submissions={len(submits)} "
        f"joins={len(joins)}")
    if markers:
        phases = []
        for m in markers:
            if m["p"] not in phases:
                phases.append(m["p"])
        say(f"   phases={phases}")

    if has_gpu:
        ordered = sorted(
            (d for d in dispatches if "t0" in d), key=lambda d: d["t0"])
        ticks = unwrap([d["t0"] for d in ordered] +
                       [d["t1"] for d in ordered], valid_bits)
        # unwrap preserves lengths and total order of the concatenated
        # (t0..., t1...) list; t1 entries interleave correctly because the
        # concatenation is re-sorted below via paired indices.
        n = len(ordered)
        t0s = ticks[:n]
        t1s = ticks[n:]
        span = max(t1s) - min(t0s)
        busy = 0.0
        kernels_ns = []
        for d, u0, u1 in zip(ordered, t0s, t1s):
            dur = max(0.0, (u1 - u0)) * period
            busy += dur
            kernels_ns.append((dur, d))
        busy_frac = busy / (span * period) if span else 0.0
        say("")
        say(f"== GPU busy fraction: {busy_frac * 100:.2f}% "
            f"(busy {fmt_ns(busy)} / span {fmt_ns(span * period)})")

        # gaps between consecutive kernels on the GPU timeline
        gap_pairs = []
        ends = {id(d): u1 for d, u1 in zip(ordered, t1s)}
        starts = {id(d): u0 for d, u0 in zip(ordered, t0s)}
        for a, b in zip(ordered, ordered[1:]):
            gap = (starts[id(b)] - ends[id(a)]) * period
            gap_pairs.append((gap, 1 if a["s"] == b["s"] else 0))
        for tag, sel in (("intra-submission",
                          lambda gi: gi[1] == 1),
                         ("inter-submission",
                          lambda gi: gi[1] == 0)):
            vals = sorted(g for g, i in gap_pairs if sel(gi=(g, i)))
            if not vals:
                continue
            say(f"   gaps {tag}: n={len(vals)} total={fmt_ns(sum(vals))} "
                f"p50={fmt_ns(pct(vals, .5))} p90={fmt_ns(pct(vals, .9))} "
                f"p99={fmt_ns(pct(vals, .99))} max={fmt_ns(vals[-1])}")
        hist = {}
        for g, _ in gap_pairs:
            if g <= 0:
                key = "<=0"
            else:
                b = 1
                while g > 10 ** (b + 1):
                    b += 1
                key = f"1e{b + 1}"
            hist[key] = hist.get(key, 0) + 1
        say("   gap histogram (ns decade: count): " + str(
            dict(sorted(hist.items(), key=lambda kv: kv[0]))))

        agg = {}
        for dur, d in kernels_ns:
            a = agg.setdefault(d["e"], {"n": 0, "total": 0.0, "durs": []})
            a["n"] += 1
            a["total"] += dur
            a["durs"].append(dur)
        say("")
        say("== top kernels by cumulative GPU time")
        for e, a in sorted(agg.items(),
                           key=lambda kv: -kv[1]["total"])[:15]:
            durs = sorted(a["durs"])
            say(f"   {kname(e):<26} n={a['n']:<6} "
                f"total={fmt_ns(a['total']):<10} "
                f"share={a['total'] / busy * 100:5.1f}% "
                f"mean={fmt_ns(a['total'] / a['n']):<9} "
                f"p50={fmt_ns(pct(durs, .5))}")

        def bufs(d):
            return {b[0] for b in d.get("b", [])}

        for tag, sel in (("decode", lambda d: d.get("phase") == "decode"),
                         ("all", lambda d: True)):
            ds = [d for d in ordered if sel(d)]
            if len(ds) < 2:
                continue
            indep = sum(1 for a, b in zip(ds, ds[1:])
                        if bufs(a).isdisjoint(bufs(b)))
            say(f"   dependency proxy [{tag}]: {indep}/{len(ds) - 1} "
                f"consecutive pairs fully disjoint "
                f"({indep / (len(ds) - 1) * 100:.1f}% - compute-binding "
                f"disjointness only; transfer/fill/alias deps invisible, "
                f"not a barrier-correctness bound)")

    waits = sorted(j["wait"] for j in joins)
    invals = sorted(j["inval"] for j in joins)
    subs = sorted(s["dur"] for s in submits)
    recs = sorted(d["h"] for d in dispatches)
    say("")
    say("== host-side costs (instrumented)")
    if waits:
        say(f"   join wait    n={len(waits)} total={fmt_ns(sum(waits))} "
            f"p50={fmt_ns(pct(waits, .5))} p90={fmt_ns(pct(waits, .9))} "
            f"max={fmt_ns(waits[-1])}")
    if invals:
        say(f"   join inval   n={len(invals)} total={fmt_ns(sum(invals))} "
            f"p50={fmt_ns(pct(invals, .5))} max={fmt_ns(invals[-1])}")
    if subs:
        say(f"   submit()     n={len(subs)} total={fmt_ns(sum(subs))} "
            f"p50={fmt_ns(pct(subs, .5))} max={fmt_ns(subs[-1])}")
    if recs:
        say(f"   dispatch record (host) n={len(recs)} "
            f"total={fmt_ns(sum(recs))} p50={fmt_ns(pct(recs, .5))} "
            f"mean={fmt_ns(sum(recs) / len(recs))}")

    per_join = {}
    for d in dispatches:
        per_join[d["s"]] = per_join.get(d["s"], 0) + 1
    say("")
    say("== dispatches per submission (in submit order)")
    say("   " + str([per_join.get(s["s"], 0) for s in submits]))

    disp_bar = [d for d in dispatches if "bar" in d]
    if disp_bar:
        emitted = sum(d["bar"] for d in disp_bar)
        say("")
        say("== dependency-barrier decisions (MLX_OMARCHY_GATED_BARRIERS)")
        say(f"   dispatch barriers: emitted={emitted} "
            f"skipped={len(disp_bar) - emitted} "
            f"({(len(disp_bar) - emitted) / len(disp_bar) * 100:.1f}% "
            f"skipped)")
        if end_rec and "barriers" in end_rec:
            say(f"   all nodes incl. transfer/fill: "
                f"emitted={end_rec['barriers']} "
                f"skipped={end_rec['barriers_skipped']}")

    if markers and has_gpu:
        by_phase = {}
        for d in dispatches:
            by_phase.setdefault(d["phase"], []).append(d)
        dur_by_id = {id(d): dur for dur, d in kernels_ns}
        say("")
        say("== per host-marker window (a dispatch sits in its submission's")
        say("   window; records flush late at joins; windows are NOT")
        say("   semantic phases - an async pipeline can submit the next")
        say("   decode step before the first token yields, so boundary")
        say("   work lands in the earlier window)")
        for ph in sorted(by_phase, key=lambda p: (p.startswith("unknown"), p)):
            ds = by_phase[ph]
            dur = sum(dur_by_id.get(id(d), 0.0) for d in ds)
            joins_ph = [j for j in joins
                        if (phase_of_host(j["t"]) or UNKNOWN_OUTSIDE) == ph]
            say(f"   {ph}: dispatches={len(ds)} gpu_busy={fmt_ns(dur)} "
                f"joins={len(joins_ph)} join_wait_total="
                f"{fmt_ns(sum(j['wait'] for j in joins_ph))}")
        n_tok = sum(1 for m in markers if m["p"] == "tok")
        decode_ds = len(by_phase.get("decode", []))
        decode_subs = [s for s in submits
                       if phase_of_host(s["t"]) == "decode"]
        if n_tok >= 2:
            # Inter-token intervals, not yields: the first decode
            # submission can land inside the prefill window (async
            # pipeline), so the decode window covers n_tok - 1 steps.
            n_int = n_tok - 1
            say(f"   dispatches/decode-interval: {decode_ds / n_int:.1f} "
                f"({decode_ds} over {n_int} inter-token intervals; the "
                f"first decode submit may sit in the prefill window)")
            say(f"   submissions/decode-interval: "
                f"{len(decode_subs) / n_int:.1f} "
                f"({len(decode_subs)} over {n_int} intervals)")
            jw = sum(j["wait"] for j in joins
                     if phase_of_host(j["t"]) == "decode")
            say(f"   join wait per decode-interval: {fmt_ns(jw / n_int)}")
            if disp_bar:
                db = sum(d["bar"] for d in by_phase.get("decode", [])
                         if "bar" in d)
                say(f"   dispatch barriers/decode-interval: {db / n_int:.1f} "
                    f"({db} emitted over {n_int} intervals)")

    print("\n".join(out))


if __name__ == "__main__":
    main()
