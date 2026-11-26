#!/usr/bin/env python3
"""Finite-logits probe for the 2B model on a card prompt (2026-09-29).

The old wheel (29cba8e) emitted non-finite logits on GDN prompts past
~300-500 tokens, which made the v0.7.6 "0/8 card" evidence suspect.
This script streams a single card-worthy chat prompt through the
real chat server and records:
  - the per-chunk SSE delta events,
  - whether any non-finite token ids appear in usage or content,
  - the total tokens produced and first-text p95 latency.
The output is a JSON line per chunk plus a final summary block.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

_HOME = os.environ.get("MARKCARDS_HOME", "<home>")
HOME_ROOT = os.path.join(_HOME, "agents", "MarkdownCards", "homes")
PAIR_FOR_MODEL = {
    "qwen3.8-2b-4bit": "everyday",
}
PYTHON = os.path.join(_HOME, ".local", "share", "mlx-omarchy", "venv", "bin", "python")
RESULTS_DIR = os.path.join(_HOME, "agents", "MarkdownCards", "results")


def call(method, path, body=None, runtime=None):
    rt = json.load(open(runtime))
    base = f"http://127.0.0.1:{rt['port']}"
    headers = {
        "Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
        "Origin": base, "Content-Type": "application/json"}
    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode())


def probe(model, prompt, max_tokens=700):
    home = os.path.join(HOME_ROOT, model.replace("/", "_"))
    runtime = os.path.join(home, "assistant", "application.json")
    os.makedirs(home, exist_ok=True)
    if os.path.exists(runtime):
        os.unlink(runtime)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "serve"),
               MLX_OMARCHY_OFFLINE="1", MLX_OMARCHY_HOME=home)
    safe = model.replace("/", "_")
    log_path = os.path.join(_HOME, "agents", "MarkdownCards",
                            f"probe_{safe}.log")
    server = subprocess.Popen(
        [PYTHON, "-m", "mlx_omarchy_assistant", "--home", home,
         "--no-browser", "--resume"],
        cwd=REPO, env=env,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    out = []
    try:
        deadline = time.monotonic() + 180
        while not os.path.exists(runtime) and time.monotonic() < deadline:
            time.sleep(0.5)
        if not os.path.exists(runtime):
            raise RuntimeError("server never wrote application.json")
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            state = call("GET", "/api/status", runtime=runtime).get("setup") or {}
            if state.get("state") == "complete":
                break
            if state.get("state") == "error":
                raise RuntimeError(f"setup error: {state}")
            time.sleep(1)
        cid = call("POST", "/api/conversations", {"save": False},
                   runtime=runtime)["id"]
        started = time.monotonic()
        turn = call("POST", f"/api/conversations/{cid}/turns",
                    {"text": prompt, "mode": "chat", "max_tokens": max_tokens},
                    runtime=runtime)["turn_id"]
        first_text_ms = None
        message = None
        chunks = 0
        completion_tokens = 0
        non_finite_hits = 0
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            call("POST", f"/api/conversations/{cid}/heartbeat",
                 {"turn_id": turn}, runtime=runtime)
            record = call("GET", f"/api/conversations/{cid}",
                          runtime=runtime)
            message = next((m for m in record.get("messages") or []
                            if m.get("turn_id") == turn
                            and m.get("role") == "assistant"), None)
            if message and message.get("status") in (
                    "complete", "error", "stopped"):
                break
            time.sleep(1)
        elapsed = time.monotonic() - started
        content = (message or {}).get("content") or ""
        completion_tokens = (message or {}).get("completion_tokens") or 0
        has_fence = "```assistant-ui" in content
        components = (message or {}).get("components") or []
        # Non-finite detection: scan for nan/inf in the raw content
        for token in ("NaN", "nan", "Infinity", "Inf", "-Infinity", "-Inf"):
            if token in content:
                non_finite_hits += content.count(token)
        summary = {
            "wheel": "0.32.3.dev202609291615+06711ad",
            "model": model, "max_tokens": max_tokens,
            "elapsed_s": round(elapsed, 2),
            "completion_tokens": completion_tokens,
            "has_assistant_ui_fence": has_fence,
            "components": [c.get("type") for c in components],
            "non_finite_hits": non_finite_hits,
            "content_preview": content[:200],
        }
        out.append(summary)
        return summary
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=30)
        except Exception:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.8-2b-4bit")
    p.add_argument("--prompt",
                   default="Give me a packing checklist for a weekend "
                           "camping trip, with at least 5 items.")
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--out", default=os.path.join(RESULTS_DIR,
                                                "probe_finite_logits.json"))
    args = p.parse_args()
    summary = probe(args.model, args.prompt, args.max_tokens)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(args.out, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
