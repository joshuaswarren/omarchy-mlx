#!/usr/bin/env python3
"""Summarize quality_ttft_probe session JSON into a per-phase table.

Usage: python3 summarize_ttft.py SESSION.json [SESSION.json ...]

Phase deltas are computed inside the coordinator process (monotonic clock);
engine lines come from the chat worker's [ttft] trace (same host clock).
Prints per-run rows and medians per phase, plus the SSE-delivery lag.
"""
import json
import statistics
import sys

ORDER = [
    ("pair_start", "submit", "pair manager start"),
    ("prompt_built", "pair_start", "history+system prompt build"),
    ("count_done", "prompt_built", "prompt token count"),
    ("admit_done", "count_done", "ensure_context admission"),
    ("generating_status_emitted", "admit_done", "manager.status + status emit"),
    ("yield_probe_done", "generating_status_emitted", "speech yield probe"),
    ("chat_sent", "yield_probe_done", "chat request dispatch"),
    ("chat_headers", "chat_sent", "engine TTFB (HTTP+queue+template+tokenize)"),
    ("first_chunk", "chat_headers", "engine prefill + first decode step"),
    ("first_text_emitted", "first_chunk", "coordinator loop + store emit"),
]


def rows(session):
    out = []
    for run in session.get("runs", []):
        phases = run.get("phases") or {}
        if not phases:
            continue
        row = {"label": run["label"], "fvt_s": run.get("fvt_s"),
               "wall_s": run.get("wall_s"), "prompt_tokens": phases.get("prompt_tokens")}
        for name, base, _human in ORDER:
            if name in phases and base in phases:
                row[name] = phases[name] - phases[base]
        if "first_text_emitted" in phases and "submit" in phases:
            row["coordinator_total"] = phases["first_text_emitted"] - phases["submit"]
        if run.get("fvt_s") is not None and "coordinator_total" in row:
            row["sse_lag"] = run["fvt_s"] - row["coordinator_total"]
        engine = {}
        for line in run.get("engine_lines") or []:
            try:
                payload = json.loads(line[len("[ttft] "):])
            except ValueError:
                continue
            engine[payload.get("event")] = payload
        if "tokenized" in engine:
            row["engine_tokenize_ms"] = engine["tokenized"].get("t")
            row["engine_prompt_tokens"] = engine["tokenized"].get("prompt")
        if "first_chunk" in engine:
            row["engine_first_chunk_ms"] = engine["first_chunk"].get("t")
        out.append(row)
    return out


def median(rows, key):
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return statistics.median(values) if values else None


def main():
    sessions = [json.load(open(path)) for path in sys.argv[1:]]
    for session in sessions:
        print(f"== {session.get('label')} ({session.get('repo_serve')}) "
              f"boot={session.get('boot_id', '')[:8]} "
              f"user_prompt_tokens={session.get('user_prompt_tokens')} "
              f"prov={'ok' if 'verified=match' in (session.get('provenance') or '') else 'CHECK'}")
        table = rows(session)
        if not table:
            print("  no measured runs with phases")
            continue
        med = {name: median(table, name) for name, _b, _h in ORDER}
        med["coordinator_total"] = median(table, "coordinator_total")
        med["sse_lag"] = median(table, "sse_lag")
        header = f"{'phase':44s} " + " ".join(f"{r['label']:>9s}" for r in table) + f" {'median':>9s}"
        print(header)
        for name, _base, human in ORDER:
            cells = " ".join(f"{r.get(name):9.3f}" if isinstance(r.get(name), (int, float)) else " " * 9
                             for r in table)
            m = med.get(name)
            print(f"{human:44s} {cells} {m:9.3f}" if m is not None else f"{human:44s} {cells}")
        for extra in ("coordinator_total", "sse_lag"):
            cells = " ".join(f"{r.get(extra):9.3f}" if isinstance(r.get(extra), (int, float)) else " " * 9
                             for r in table)
            m = med.get(extra)
            print(f"{extra:44s} {cells} {m:9.3f}" if m is not None else f"{extra:44s} {cells}")
        fvts = [r["fvt_s"] for r in table if r.get("fvt_s") is not None]
        if fvts:
            print(f"fvt_s: median={statistics.median(fvts):.3f} "
                  f"min={min(fvts):.3f} max={max(fvts):.3f} n={len(fvts)}")
        # engine-internal split (worker clock): tokenize-done -> first chunk
        eng = []
        for r in table:
            if isinstance(r.get("engine_first_chunk_ms"), (int, float)) \
                    and isinstance(r.get("engine_tokenize_ms"), (int, float)):
                eng.append(r["engine_first_chunk_ms"] - r["engine_tokenize_ms"])
        if eng:
            print(f"engine prefill+first-decode (worker clock): median={statistics.median(eng):.3f}")
        gate_bad = [r["label"] for r in session.get("runs", []) if (r.get("gate") or {}).get("gate")]
        if gate_bad:
            print(f"TIMING GATE NOT QUIET on: {gate_bad}")
        print()


if __name__ == "__main__":
    main()
