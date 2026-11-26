#!/usr/bin/env python3
"""Card-turn decomposition: where does the time to the first visible
component go, per pair?

Drives ONE card turn through the assistant HTTP API (same shape as
card_timing_boot.py) and captures every SSE event with full data and a
monotonic timestamp.  Derives: first text, text-over-time, prose/payload
split (markdown-table aware, fence aware), tokenizer counts per phase,
decode tok/s per phase, component emit time, and the final message
content (so the fence bytes are inspectable).

Usage:
  card_decompose.py --pair everyday9b --home <dir> --repo-serve <dir> \
      --out <json> [--timeout-s 600] [--prompt "custom"]
"""
import argparse
import json
import os
import signal
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from card_timing_boot import load_assistant_cls, stream_events, tok_count

CARD_PROMPT = ("Show a chart of population for: Paris 2.1M, Tokyo 13.9M, "
               "Lagos 21.0M.")


def derive(events, message, t0, prompt_tokens=None):
    """Split the reply into prose / payload phases from the event stream."""
    out = {}
    text_events = [(e["ts"], e["data"].get("text", ""))
                   for e in events if e["type"] == "text"]
    out["first_text_s"] = (round(text_events[0][0] - t0, 3)
                           if text_events else None)
    comp = [(e["ts"], e["data"].get("type")) for e in events
            if e["type"] == "component"]
    out["first_component_s"] = (round(comp[0][0] - t0, 3) if comp else None)
    out["component_types"] = [c[1] for c in comp]
    status_events = [(e["ts"], e["data"].get("state"))
                     for e in events if e["type"] == "status"]
    out["status_events"] = [(round(ts - t0, 3), st) for ts, st in status_events]

    full = "".join(t for _, t in text_events)
    out["text_len"] = len(full)
    out["text_tokens"] = tok_count(full) if full else 0

    # Reconstruct the arrival time of each character, then find the
    # prose/payload boundary: the first markdown table line ("| ...") that
    # survives to the final text, or, when the text is one burst, the whole
    # text is one phase (fence fallback) — phase split not derivable.
    char_ts = []
    for ts, chunk in text_events:
        char_ts.extend([ts] * len(chunk))
    lines = full.split("\n")
    table_start_char = None
    pos = 0
    for line in lines:
        if line.lstrip().startswith("|"):
            table_start_char = pos
            break
        pos += len(line) + 1
    table_end_char = None
    pos = 0
    last_pipe_line_end = None
    prev_was_table = False
    for line in lines:
        is_table = bool(line.lstrip().startswith("|"))
        end = pos + len(line)
        if is_table:
            last_pipe_line_end = end
            prev_was_table = True
        elif prev_was_table and not line.strip():
            table_end_char = last_pipe_line_end
            prev_was_table = False
        pos = end + 1
    if prev_was_table and last_pipe_line_end is not None:
        table_end_char = last_pipe_line_end

    def ts_at(char_idx):
        if not char_ts or char_idx is None or char_idx <= 0:
            return None
        return round(char_ts[min(char_idx, len(char_ts) - 1)] - t0, 3)

    prose = full[:table_start_char] if table_start_char is not None else full
    payload = (full[table_start_char:table_end_char]
               if table_start_char is not None else "")
    out["prose_chars"] = len(prose.rstrip())
    out["prose_tokens"] = tok_count(prose) if prose.strip() else 0
    out["payload_chars"] = len(payload)
    out["payload_tokens"] = tok_count(payload) if payload.strip() else 0
    out["table_start_s"] = ts_at(table_start_char)
    out["table_complete_s"] = ts_at(table_end_char)
    ft, ts_, tc = out["first_text_s"], out["table_start_s"], out["table_complete_s"]
    if prose.strip() and ts_ and ft is not None and ts_ > ft:
        out["prose_decode_tok_s"] = round(out["prose_tokens"] / (ts_ - ft), 2)
    if payload.strip() and tc and ts_ and tc > ts_:
        out["payload_decode_tok_s"] = round(
            out["payload_tokens"] / (tc - ts_), 2)
    if out["text_tokens"] and out["first_text_s"] is not None:
        wall = events[-1]["ts"] - t0 if events else None
        if wall and wall > out["first_text_s"]:
            out["overall_decode_tok_s"] = round(
                out["text_tokens"] / (wall - out["first_text_s"]), 2)

    # Cumulative visible characters at ~1 s resolution (streaming shape).
    shape = []
    if char_ts:
        end = int(char_ts[-1] - t0) + 1
        n = 0
        for sec in range(0, end):
            while n < len(char_ts) and char_ts[n] - t0 <= sec:
                n += 1
            shape.append(n)
    out["cum_chars_per_s"] = shape
    if message:
        content = message.get("content") or ""
        fence_at = content.find("```assistant-ui")
        out["message_has_fence"] = fence_at >= 0
        if fence_at >= 0:
            close = content.find("```", fence_at + 3)
            out["fence_bytes"] = len(content[fence_at:close + 3])
            out["fence_tokens"] = tok_count(content[fence_at:close + 3])
        out["message_status"] = message.get("status")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--repo-serve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="decompose")
    ap.add_argument("--prompt", default=CARD_PROMPT)
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()

    from pair_memory_v2 import call, heartbeat_loop
    Assistant, _, _ = load_assistant_cls()

    pair = Assistant(args.pair, args.home, args.repo_serve)
    do_setup = pair.boot_id_mismatch()
    print(f"do_setup={do_setup}", flush=True)
    pair.start(do_setup)
    setup_deadline = time.monotonic() + (1500 if do_setup else 180)
    while time.monotonic() < setup_deadline:
        state = call("GET", "/api/status",
                     runtime_path=pair.runtime).get("setup") or {}
        if state.get("state") == "complete":
            break
        if state.get("state") == "error":
            pair.kill()
            raise RuntimeError(f"setup error: {state}")
        time.sleep(1)
    print("setup complete", flush=True)

    cid = call("POST", "/api/conversations", {"save": False},
               runtime_path=pair.runtime)["id"]
    body = {"text": args.prompt, "mode": "chat", "max_tokens": 700}
    t0 = time.monotonic()
    turn = call("POST", f"/api/conversations/{cid}/turns", body,
                runtime_path=pair.runtime)["turn_id"]

    stop = threading.Event()
    beat = threading.Thread(target=heartbeat_loop,
                            args=(cid, turn, pair.runtime, stop),
                            daemon=True)
    beat.start()

    events = []

    def drain():
        try:
            for etype, data, ts in stream_events(cid, pair.runtime, stop,
                                                 args.timeout_s):
                events.append({"type": etype, "ts": ts, "data": data})
        except Exception as e:
            events.append({"type": "stream_error", "data": {"text": str(e)}})

    drain_t = threading.Thread(target=drain, daemon=True)
    drain_t.start()

    message = None
    deadline = time.monotonic() + args.timeout_s
    try:
        while time.monotonic() < deadline:
            record = call("GET", f"/api/conversations/{cid}",
                          runtime_path=pair.runtime)
            message = next((m for m in record.get("messages") or []
                            if m.get("turn_id") == turn
                            and m.get("role") == "assistant"), None)
            if message and message.get("status") in ("complete", "error",
                                                     "stopped"):
                break
            time.sleep(0.5)
    finally:
        stop.set()
        beat.join(timeout=5)
        drain_t.join(timeout=5)

    wall = time.monotonic() - t0
    derived = derive(events, message, t0)
    out = {
        "label": args.label,
        "pair_id": args.pair,
        "prompt": args.prompt,
        "wall_s": round(wall, 2),
        "derived": derived,
        "final_text": (message or {}).get("content", ""),
        "events": [{"type": e["type"], "t": round(e["ts"] - t0, 3),
                    "data": e["data"]} for e in events],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    summary = {"label": args.label, "pair": args.pair,
               "wall_s": out["wall_s"],
               **{k: v for k, v in derived.items()
                  if k != "cum_chars_per_s"}}
    print(json.dumps(summary, indent=2), flush=True)

    time.sleep(3)
    pair.kill()
    time.sleep(3)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
