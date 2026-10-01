#!/usr/bin/env python3
"""PairGates per-pair card timing on ONE boot.

For each pair (compact4b / everyday9b / quality27b): resume the saved
pair home, submit one card turn, and capture the SSE stream with
timestamps.  Records first_visible_text_s, first_visible_component_s,
wall, visible-text tokens (tokenizer-counted), and decode tok/s.

Usage:
  card_timing_boot.py --pair compact --home <dir> --repo-serve <dir> \
      --label <label> --out <json> [--timeout-s 600]
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CARD_PROMPT = ("Show a chart of population for: Paris 2.1M, Tokyo 13.9M, "
               "Lagos 21.0M.")


def load_assistant_cls():
    try:
        from pair_memory_v2 import Assistant, call, heartbeat_loop
        return Assistant, call, heartbeat_loop
    except ImportError:
        raise


def stream_events(cid, runtime, stop, timeout_s):
    import http.client
    rt = json.load(open(runtime))
    conn = http.client.HTTPConnection("127.0.0.1", rt["port"], timeout=300)
    path = f"/api/conversations/{cid}/events?after=0"
    conn.request("GET", path, headers={"Cookie": rt["cookie"]})
    r = conn.getresponse()
    buf = b""
    started = time.monotonic()
    try:
        while time.monotonic() - started < timeout_s:
            if stop.is_set():
                break
            chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                block, buf = buf.split(b"\n\n", 1)
                for line in block.split(b"\n"):
                    if line.startswith(b"data:"):
                        event = json.loads(line[5:])
                        yield (event.get("type"), event.get("data") or {},
                               time.monotonic())
    finally:
        conn.close()


_TOK = None


def tok_count(text):
    global _TOK
    if _TOK is None:
        try:
            from transformers import AutoTokenizer
            _TOK = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-0.6B", trust_remote_code=False)
        except Exception as e:
            print(f"tokenizer unavailable ({e}); char/4 heuristic")
            _TOK = False
    if _TOK is False:
        return len(text) // 4
    return len(_TOK(text)["input_ids"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--repo-serve", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()

    Assistant, call, heartbeat_loop = load_assistant_cls()

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
    body = {"text": CARD_PROMPT, "mode": "chat", "max_tokens": 700}
    t0 = time.monotonic()
    turn = call("POST", f"/api/conversations/{cid}/turns", body,
                runtime_path=pair.runtime)["turn_id"]

    stop = threading.Event()
    beat = threading.Thread(target=heartbeat_loop,
                            args=(cid, turn, pair.runtime, stop),
                            daemon=True)
    beat.start()

    events = []
    first_text_at = None
    first_component_at = None

    def drain():
        nonlocal first_text_at, first_component_at
        try:
            for etype, data, ts in stream_events(cid, pair.runtime, stop,
                                                 args.timeout_s):
                events.append({"type": etype, "ts": ts,
                               "data_head": json.dumps(data)[:400]})
                if first_text_at is None and etype == "text" and data:
                    first_text_at = ts - t0
                if first_component_at is None and etype == "component":
                    first_component_at = ts - t0
        except Exception as e:
            events.append({"type": "stream_error", "data_head": str(e)[:200]})

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
    text = (message or {}).get("content") or ""
    components = [c.get("type") for c in (message or {}).get("components") or []]
    text_tokens = tok_count(text) if text else 0
    decode_tok_s = (text_tokens / (wall - first_text_at)
                    if first_text_at and wall > first_text_at and text_tokens
                    else None)

    out = {
        "label": args.label,
        "pair_id": args.pair,
        "prompt": CARD_PROMPT,
        "status": (message or {}).get("status"),
        "wall_s": round(wall, 2),
        "first_visible_text_s": (round(first_text_at, 3)
                                 if first_text_at else None),
        "first_visible_component_s": (round(first_component_at, 3)
                                      if first_component_at else None),
        "text_len": len(text),
        "text_tokens": text_tokens,
        "decode_tok_s": round(decode_tok_s, 3) if decode_tok_s else None,
        "components": components,
        "events_captured": len(events),
        "events": events[:40],
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "events"},
                     indent=2), flush=True)

    time.sleep(3)
    pair.kill()
    time.sleep(3)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
