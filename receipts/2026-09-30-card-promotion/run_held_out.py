#!/usr/bin/env python3
"""Run the frozen HELD-OUT card-promotion suite once per chat model through
the real coordinator on the M2.  Records per-prompt outcomes and a final
summary so the orchestrator can verify the 10/12 card-worthy + 0 spurious
acceptance bar.

Usage:
    flock /tmp/m2-gpu.lock PYTHONPATH=serve python3 \\
        receipts/2026-09-30-card-promotion/run_held_out.py --model qwen3.8-2b-4bit

The script starts the assistant server with the named chat model, drives
24 prompts through HTTP, and captures the assistant-built component
events.  It writes one JSON line per prompt plus a summary block.
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
# Runtime paths.  Defaults below are placeholders; override via the
# MARKCARDS_HOME / MARKCARDS_VENV environment variables before running.
_HOME = os.environ.get("MARKCARDS_HOME", "<home>")
HOME_ROOT = os.path.join(_HOME, "agents", "MarkdownCards", "homes")
# One home per chat model so the saved pair is unambiguous.
PAIR_FOR_MODEL = {
    "qwen3.8-2b-4bit": "everyday",
    "qwen3.8-27b-4bit": "quality",
}
PYTHON = os.path.join(_HOME, ".local", "share", "mlx-omarchy", "venv", "bin", "python")


def home_for(chat_model: str) -> str:
    safe = chat_model.replace("/", "_")
    return os.path.join(HOME_ROOT, safe)


def runtime_for(chat_model: str) -> str:
    return os.path.join(home_for(chat_model), "assistant", "application.json")


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


def log(line, handle):
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def run(held_out_path, chat_model, output_path, max_tokens=700, timeout_s=600,
        resume_only=False):
    prompts = json.load(open(held_out_path))
    suite_sha = prompts.get("sha256", "")
    out = open(output_path, "w")

    home = home_for(chat_model)
    runtime = runtime_for(chat_model)
    pair_id = PAIR_FOR_MODEL[chat_model]
    os.makedirs(home, exist_ok=True)
    if os.path.exists(runtime):
        os.unlink(runtime)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "serve"),
               MLX_OMARCHY_OFFLINE="1",
               MLX_OMARCHY_HOME=home)
    safe = chat_model.replace("/", "_")
    log_path = os.path.join(_HOME, "agents", "MarkdownCards",
                            f"server_{safe}.log")
    # First run: do an explicit setup (--pair everyday --yes) so the saved
    # pair is the right chat model.  Subsequent runs: --resume.
    setup_args = ["--pair", pair_id, "--yes"] if not resume_only else ["--resume"]
    server = subprocess.Popen(
        [PYTHON, "-m", "mlx_omarchy_assistant", "--home", home, "--no-browser"]
        + setup_args,
        cwd=REPO, env=env,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    try:
        # Wait for runtime + setup.
        deadline = time.monotonic() + 300
        while not os.path.exists(runtime) and time.monotonic() < deadline:
            time.sleep(0.3)
        if not os.path.exists(runtime):
            raise RuntimeError("server never wrote application.json")
        setup_deadline = time.monotonic() + 1800
        while time.monotonic() < setup_deadline:
            state = call("GET", "/api/status", runtime=runtime).get("setup") or {}
            if state.get("state") == "complete":
                break
            if state.get("state") == "error":
                raise RuntimeError(f"setup error: {state}")
            time.sleep(1)
        else:
            raise RuntimeError("setup did not complete in time")

        summary = {"suite_sha256": suite_sha, "chat_model": chat_model,
                   "pair_id": pair_id, "max_tokens": max_tokens,
                   "prompts": []}
        valid_card_kinds = {"checklist", "comparison", "timeline", "facts"}

        for index, prompt in enumerate(prompts["prompts"]):
            cid = call("POST", "/api/conversations", {"save": False},
                       runtime=runtime)["id"]
            started = time.monotonic()
            turn = call("POST", f"/api/conversations/{cid}/turns",
                        {"text": prompt["text"], "mode": "chat",
                         "max_tokens": max_tokens},
                        runtime=runtime)["turn"]
            message = None
            deadline = time.monotonic() + timeout_s
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
            components = (message or {}).get("components") or []
            types = [c.get("type") for c in components]
            elapsed = time.monotonic() - started
            expect = prompt["expect"]
            kinds_valid = any(t in valid_card_kinds for t in types)
            pass_ = (expect == "card" and kinds_valid) or (
                expect == "none" and not kinds_valid)
            outcome = {
                "id": prompt["id"], "category": prompt["category"],
                "kind": prompt["kind"], "expect": expect,
                "components": types, "elapsed_s": round(elapsed, 1),
                "status": (message or {}).get("status"),
                "pass": pass_,
            }
            log(f"HELD_OUT {index + 1:2d}/{len(prompts['prompts'])} "
                f"id={prompt['id']} expect={expect} types={types} "
                f"elapsed={elapsed:.0f}s pass={pass_}", out)
            summary["prompts"].append(outcome)

        # Score and threshold checks.
        card_worthy = [p for p in summary["prompts"] if p["expect"] == "card"]
        plain = [p for p in summary["prompts"] if p["expect"] == "none"]
        valid = [p for p in card_worthy if p["pass"]]
        spurious = [p for p in plain if any(t in valid_card_kinds
                                            for t in p["components"])]
        summary["valid_card_worthy"] = f"{len(valid)}/{len(card_worthy)}"
        summary["spurious_cards"] = len(spurious)
        summary["thresholds"] = {
            "card_worthy_min": 10, "card_worthy_total": len(card_worthy),
            "spurious_max": 0}
        summary["gates_pass"] = (
            len(valid) >= 10 and len(spurious) == 0)
        log(f"SUMMARY chat_model={chat_model} pair_id={pair_id} "
            f"valid_card_worthy={len(valid)}/{len(card_worthy)} "
            f"spurious={len(spurious)} gates_pass={summary['gates_pass']}", out)
        # Persist a JSON summary alongside the JSONL log.
        summary_path = output_path.rsplit(".", 1)[0] + "_summary.json"
        with open(summary_path, "w") as fp:
            json.dump(summary, fp, indent=2)
    finally:
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=60)
        except Exception:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--held-out", default=os.path.join(
        REPO, "tests", "fixtures", "cards_held_out.json"))
    p.add_argument("--out", default=os.path.join(
        HERE, f"held_out_{int(time.time())}.jsonl"))
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--resume", action="store_true",
                   help="skip --pair setup; the saved pair must already exist")
    args = p.parse_args()
    run(args.held_out, args.model, args.out, max_tokens=args.max_tokens,
        resume_only=args.resume)


if __name__ == "__main__":
    main()
