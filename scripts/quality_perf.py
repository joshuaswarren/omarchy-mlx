#!/usr/bin/env python3
"""PairGates Quality idle-GPU perf harness.

Drives the assistant through:
  - One chat prefill + decode measurement (prompt 512, decode 128).
  - Prefill 2048 prompt.
  - TTFT for 170-token prompt.
  - First-visible-text latency through the assistant for a chat turn.
  - 5 card turns.

Also samples the dispatch line count and per-worker attribution.

Writes JSON summary at --out.
"""
import argparse
import json
import os
import re
import time
import urllib.request
import http.client


def http_post(url, body, cookies=None, csrf=None, timeout=600):
    parsed = urllib.request.urlparse(url)
    c = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    headers = {"Content-Type": "application/json",
               "Origin": f"http://{parsed.hostname}:{parsed.port or 80}"}
    if cookies:
        headers["Cookie"] = cookies
    if csrf:
        headers["X-Assistant-CSRF"] = csrf
    c.request("POST", parsed.path, json.dumps(body), headers)
    r = c.getresponse()
    payload = json.loads(r.read())
    c.close()
    if r.status >= 400:
        raise RuntimeError(f"POST {parsed.path}: {r.status} {payload}")
    return payload


def wait_for_sse_text(runtime, cid, turn_id, after_seq=0, deadline_s=600):
    host = "127.0.0.1"
    port = runtime["port"]
    cursor = after_seq
    first_text_ts = None
    final_text = ""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        c = http.client.HTTPConnection(host, port, timeout=300)
        c.request("GET", f"/api/conversations/{cid}/events?after={cursor}",
                  headers={"Cookie": runtime["cookie"]})
        r = c.getresponse()
        if r.status != 200:
            r.read()
            c.close()
            time.sleep(0.5)
            continue
        for line in r:
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line[5:])
            cursor = event["sequence"]
            if event.get("turn_id") != turn_id:
                continue
            etype = event.get("type")
            if etype == "text":
                if first_text_ts is None:
                    first_text_ts = time.time()
                final_text += event["data"]["text"]
            elif etype == "done":
                c.close()
                return final_text, first_text_ts, event.get("data", {})
            elif etype == "error":
                c.close()
                return final_text, first_text_ts, {"error": event["data"]}
        c.close()
        time.sleep(0.5)
    return final_text, first_text_ts, {"error": "timeout"}


def count_dispatch(path):
    try:
        with open(path) as f:
            return f.read().count("[rtmod] DISPATCH")
    except (FileNotFoundError, OSError):
        return 0


def measure_chat_through_assistant(runtime, cid, prompt, max_tokens, label,
                                    decision_log, chat_log):
    t0 = time.time()
    turn = http_post(f"http://127.0.0.1:{runtime['port']}/api/conversations/{cid}/turns",
                     {"text": prompt, "mode": "chat", "max_tokens": max_tokens},
                     cookies=runtime["cookie"], csrf=runtime["csrf"])
    text, fvt, done = wait_for_sse_text(runtime, cid, turn["turn_id"])
    wall = time.time() - t0
    return {
        "label": label,
        "wall_s": wall,
        "first_visible_text_s": (fvt - (t0)) if fvt else None,
        "text_len": len(text),
        "prompt_chars": len(prompt),
        "max_tokens": max_tokens,
        "turn_done": done,
        "dispatch_lines_chat_after": count_dispatch(chat_log),
        "dispatch_lines_decision_after": count_dispatch(decision_log),
    }


def measure_chat_through_worker(chat_port, prompt, max_tokens, label, chat_log):
    """Direct chat completion to the worker to get prefill/decode numbers
    independent of the assistant's wrapping."""
    payload = {
        "model": "qwen3.8-27b-4bit",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    t0 = time.time()
    parsed = urllib.request.urlparse(f"http://127.0.0.1:{chat_port}/v1/chat/completions")
    c = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=600)
    c.request("POST", parsed.path, json.dumps(payload),
              headers={"Content-Type": "application/json"})
    r = c.getresponse()
    body = r.read().decode()
    c.close()
    wall = time.time() - t0
    try:
        out = json.loads(body)
    except json.JSONDecodeError:
        return {"label": label, "wall_s": wall, "error": body[:300]}
    usage = out.get("usage", {})
    return {
        "label": label,
        "wall_s": wall,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "first_token": None,
        "content_head": (out.get("choices") or [{}])[0].get("message", {}).get("content", "")[:120],
        "dispatch_lines_chat_after": count_dispatch(chat_log),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assistant-app-json", required=True)
    ap.add_argument("--chat-port", type=int, required=True)
    ap.add_argument("--chat-log", required=True)
    ap.add_argument("--decision-log", default="")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.assistant_app_json) as f:
        runtime = json.load(f)
    url = f"http://127.0.0.1:{runtime['port']}"
    conv = http_post(f"{url}/api/conversations", {"save": False},
                     cookies=runtime["cookie"], csrf=runtime["csrf"])
    cid = conv["id"]

    filler = ("The quick brown fox jumps over the lazy dog. " * 30).strip()

    prefill_512 = filler * 18  # ~1.7k chars; ensure > 512 tokens.
    prefill_2048 = filler * 70  # ~6.4k chars; > 2048 tokens.
    ttft_prompt = filler * 7    # ~640 chars; ~170 tokens approximately.
    chat_prompt = "Say hello in five words or fewer."

    results = []

    # First-visible-text through the assistant for one chat turn.
    results.append(measure_chat_through_assistant(
        runtime, cid, chat_prompt, 32, "fvt_chat_turn",
        args.decision_log, args.chat_log))

    # TTFT (assistant path with short max_tokens).
    results.append(measure_chat_through_assistant(
        runtime, cid, ttft_prompt, 64, "ttft_170prompt",
        args.decision_log, args.chat_log))

    # Prefill 512 via direct worker.
    results.append(measure_chat_through_worker(
        args.chat_port, prefill_512, 1, "prefill_512_direct",
        args.chat_log))

    # Prefill 2048 via direct worker.
    results.append(measure_chat_through_worker(
        args.chat_port, prefill_2048, 1, "prefill_2048_direct",
        args.chat_log))

    # Decode 128 after 512 via direct worker.
    decode_prompt = "Repeat the last sentence five times verbatim."
    results.append(measure_chat_through_worker(
        args.chat_port, prefill_512, 128, "decode_128_after_512",
        args.chat_log))

    # 5 card turns through the assistant.
    card_prompts = [
        "Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos 21.0M.",
        "Show a bar chart of monthly rainfall (mm): Jan 30, Feb 25, Mar 35, Apr 50, May 60, Jun 70, Jul 75, Aug 70, Sep 55, Oct 45, Nov 35, Dec 30.",
        "Plot a timeline of these events: 2020 founding, 2021 first model, 2023 series A, 2024 expansion.",
        "Make a checklist for launching a new website.",
        "Create a form for collecting customer feedback with rating, comments, and email fields.",
    ]
    for i, p in enumerate(card_prompts):
        results.append(measure_chat_through_assistant(
            runtime, cid, p, 256, f"card_turn_{i+1}",
            args.decision_log, args.chat_log))

    out_obj = {
        "label": args.label,
        "results": results,
        "chat_dispatch_lines_end": count_dispatch(args.chat_log),
        "decision_dispatch_lines_end": count_dispatch(args.decision_log) if args.decision_log else 0,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"wrote {args.out}")
    for r in results:
        print(f"  {r.get('label'):30s} wall_s={r.get('wall_s', 0):.3f}  "
              f"text_len={r.get('text_len', 0)}  prompt_tok={r.get('prompt_tokens', '-')}")


if __name__ == "__main__":
    main()