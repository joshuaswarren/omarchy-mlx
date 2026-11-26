#!/usr/bin/env python3
"""PairGates memory + perf harness driven through the assistant HTTP API.

Follows the pattern in receipts/2026-09-30-card-promotion/run_suite.py:
spawn the assistant with --resume (fallback --setup on boot_id mismatch),
drive turns via /api/conversations/{cid}/turns, read the assistant
message back via /api/conversations/{cid}, sample memory at every
phase boundary, leave no resident workers.

Usage: pair_memory_turns.py <pair_id> <home_dir> <out_dir> <label>
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from pair_memory_common import (
    call,
    capture_baseline_top_rss,
    find_chat_port,
    read_context_tokens,
    sample,
    wait_for_setup,
)


class Assistant:
    def __init__(self, pair_id, home, repo_serve):
        self.pair_id = pair_id
        self.home = home
        self.repo_serve = repo_serve
        self.runtime = os.path.join(home, "assistant", "application.json")
        self.server = None
        self.log_path = os.path.join(home, "assistant", "pair_driver.log")

    def boot_id_mismatch(self):
        cur = ""
        try:
            with open("/proc/sys/kernel/random/boot_id") as f:
                cur = f.read().strip()
        except Exception:
            pass
        lock = os.path.join(self.home, "assistant", "pair-locks", f"{self.pair_id}.json")
        saved = ""
        if os.path.exists(lock):
            try:
                saved = json.load(open(lock)).get("boot_id", "")
            except Exception:
                saved = ""
        return saved != cur

    def start(self, do_setup):
        os.makedirs(self.home, exist_ok=True)
        if os.path.exists(self.runtime):
            os.unlink(self.runtime)
        env = dict(os.environ,
                   PYTHONPATH=self.repo_serve + ":" + os.environ.get("PYTHONPATH", ""),
                   MLX_OMARCHY_OFFLINE="1",
                   MLX_OMARCHY_HOME=self.home,
                   MLX_OMARCHY_PAIR_DEV_QUALIFICATION="1")
        args = ["--pair", self.pair_id, "--yes"] if do_setup else ["--resume"]
        self.server = subprocess.Popen(
            ["<home>/.local/share/mlx-omarchy/venv/bin/python",
             "-m", "mlx_omarchy_assistant",
             "--home", self.home, "--no-browser"] + args,
            cwd=os.path.dirname(self.repo_serve),
            env=env,
            stdout=open(self.log_path, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True)
        atexit_kill(self.server)
        deadline = time.monotonic() + (1500 if do_setup else 180)
        while not os.path.exists(self.runtime) and time.monotonic() < deadline:
            time.sleep(0.5)
        if not os.path.exists(self.runtime):
            raise RuntimeError("server never wrote application.json")

    def kill(self):
        if self.server is None:
            return
        try:
            os.killpg(self.server.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.server.wait(timeout=10)
        except Exception:
            try:
                os.killpg(self.server.pid, signal.SIGKILL)
            except Exception:
                pass
        subprocess.run(["bash", "-lc",
                        "ps -eo pid,cmd | grep -E "
                        "'mlx_omarchy_assistant.*PairGates|"
                        "_mlxlm_server.*PairGates|"
                        "mlx_omarchy_laya.*PairGates' | "
                        "awk '{print $1}' | xargs -r kill -9 2>/dev/null"],
                       timeout=15, check=False)


_KILL_LIST = []


def atexit_kill(server):
    _KILL_LIST.append(server)


def kill_all():
    for s in _KILL_LIST:
        try:
            os.killpg(s.pid, signal.SIGTERM)
        except Exception:
            pass


def call(method, path, body=None, runtime=None):
    rt = json.load(open(runtime))
    base = f"http://127.0.0.1:{rt['port']}"
    headers = {"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
               "Origin": base, "Content-Type": "application/json"}
    req = urllib.request.Request(base + path,
                                 data=None if body is None else json.dumps(body).encode(),
                                 method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode())


def heartbeat_loop(cid, turn, runtime, stop, gaps):
    while not stop.wait(5):
        started = time.monotonic()
        try:
            call("POST", f"/api/conversations/{cid}/heartbeat",
                 {"turn_id": turn}, runtime_path=runtime_path)
        except Exception:
            pass
        gaps.append(time.monotonic() - started)


def wait_turn(cid, turn, runtime, timeout_s=600):
    stop, gaps = threading.Event(), []
    beat = threading.Thread(target=heartbeat_loop,
                            args=(cid, turn, runtime, stop, gaps), daemon=True)
    beat.start()
    deadline = time.monotonic() + timeout_s
    message = None
    try:
        while time.monotonic() < deadline:
            record = call("GET", f"/api/conversations/{cid}", runtime_path=runtime_path)
            message = next((m for m in record.get("messages") or []
                            if m.get("turn_id") == turn
                            and m.get("role") == "assistant"), None)
            if message and message.get("status") in ("complete", "error", "stopped"):
                break
            time.sleep(1)
    finally:
        stop.set()
        beat.join()
    return message


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--repo-serve", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()

    pair = Assistant(args.pair, args.home, args.repo_serve)
    do_setup = pair.boot_id_mismatch()
    print(f"boot_id mismatch={do_setup}", flush=True)

    # Pre-start baseline BEFORE the assistant launches: sample system memory
    # + record other GPU/host processes so contamination is visible.
    baseline_samples = [capture_baseline_top_rss(
        os.path.join(os.path.dirname(args.out), "baseline-top-rss.json"))]
    print(f"baseline sys_used={baseline_samples[0]['system_used_bytes']/2**20:.1f} MiB", flush=True)

    print("starting assistant", flush=True)
    pair.start(do_setup)

    # Poll setup (raises on timeout).
    wait_for_setup(pair, pair.runtime, do_setup)
    print(f"setup complete", flush=True)

    chat_port = find_chat_port(args.home, args.pair)
    print(f"chat port: {chat_port}", flush=True)

    # Start of phase samples: baseline (before the assistant launched) then
    # idle_before (pair loaded, no turns yet).
    samples = list(baseline_samples)
    samples.append(sample("idle_before", pair.server.pid, chat_port, drm_allocated=True, include_stub_pid=True))

    # Turn 1: plain chat.
    cid = call("POST", "/api/conversations", {"save": False},
               runtime_path=pair.runtime)["id"]
    t0 = time.monotonic()
    turn = call("POST", f"/api/conversations/{cid}/turns",
                {"text": "Hello. Reply briefly.", "mode": "chat", "max_tokens": 64},
                runtime_path=pair.runtime)["turn_id"]
    msg = wait_turn(cid, turn, pair.runtime, timeout_s=args.timeout_s)
    text = (msg or {}).get("content") or ""
    samples.append({"phase": "after_plain_chat",
                    "wall_s": time.monotonic() - t0,
                    "text_len": len(text),
                    "reply_head": text[:200],
                    "status": (msg or {}).get("status"),
                    **sample("after_plain_chat", pair.server.pid, chat_port)})
    print(f"plain chat: status={(msg or {}).get('status')} len={len(text)}", flush=True)

    # Turn 2: long-context (~2k tokens).
    filler = ("The quick brown fox jumps over the lazy dog. " * 30).strip()
    long_prompt = "Summarize the following in one sentence:\n\n" + (filler + "\n") * 16
    cid2 = call("POST", "/api/conversations", {"save": False},
                runtime_path=pair.runtime)["id"]
    t0 = time.monotonic()
    turn = call("POST", f"/api/conversations/{cid2}/turns",
                {"text": long_prompt, "mode": "chat", "max_tokens": 256},
                runtime_path=pair.runtime)["turn_id"]
    msg = wait_turn(cid2, turn, pair.runtime, timeout_s=args.timeout_s)
    text = (msg or {}).get("content") or ""
    samples.append({"phase": "after_long_context",
                    "prompt_chars": len(long_prompt),
                    "wall_s": time.monotonic() - t0,
                    "text_len": len(text),
                    "status": (msg or {}).get("status"),
                    **sample("after_long_context", pair.server.pid, chat_port)})
    print(f"long context: status={(msg or {}).get('status')} len={len(text)}", flush=True)

    # Turn 3: compare options decision.
    compare = {"mode": "compare",
               "text": "Pick a winner between the alternatives.",
               "options": [{"id": "espresso", "label": "Espresso"},
                           {"id": "pourover", "label": "Pour-over"},
                           {"id": "cold_brew", "label": "Cold brew"}],
               "criteria": "bitterness and clarity"}
    cid3 = call("POST", "/api/conversations", {"save": False},
                runtime_path=pair.runtime)["id"]
    t0 = time.monotonic()
    turn = call("POST", f"/api/conversations/{cid3}/turns", compare,
                runtime_path=pair.runtime)["turn_id"]
    msg = wait_turn(cid3, turn, pair.runtime, timeout_s=args.timeout_s)
    text = (msg or {}).get("content") or ""
    samples.append({"phase": "after_compare",
                    "wall_s": time.monotonic() - t0,
                    "text_len": len(text),
                    "status": (msg or {}).get("status"),
                    "components": [c.get("type") for c in (msg or {}).get("components") or []],
                    **sample("after_compare", pair.server.pid, chat_port)})
    print(f"compare: status={(msg or {}).get('status')} len={len(text)}", flush=True)

    # Turn 4: card.
    cid4 = call("POST", "/api/conversations", {"save": False},
                runtime_path=pair.runtime)["id"]
    t0 = time.monotonic()
    turn = call("POST", f"/api/conversations/{cid4}/turns",
                {"text": "Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos 21.0M.",
                 "mode": "chat", "max_tokens": 256},
                runtime_path=pair.runtime)["turn_id"]
    msg = wait_turn(cid4, turn, pair.runtime, timeout_s=args.timeout_s)
    text = (msg or {}).get("content") or ""
    # Record the full message record for debugging (Main's audit: silent
    # empty reply on a card request is a defect to report).
    record = call("GET", f"/api/conversations/{cid4}", runtime_path=pair.runtime)
    samples.append({"phase": "after_card",
                    "wall_s": time.monotonic() - t0,
                    "text_len": len(text),
                    "status": (msg or {}).get("status"),
                    "components": [c.get("type") for c in (msg or {}).get("components") or []],
                    "message_full": msg,
                    "record_messages": [
                        {"role": m.get("role"), "turn_id": m.get("turn_id"),
                         "status": m.get("status"), "content": (m.get("content") or "")[:2000]}
                        for m in record.get("messages") or []
                    ],
                    **sample("after_card", pair.server.pid, chat_port)})
    print(f"card: status={(msg or {}).get('status')} len={len(text)}", flush=True)

    # Idle after.
    time.sleep(3)
    samples.append(sample("idle_after", pair.server.pid, chat_port, drm_allocated=True, include_stub_pid=True))

    # Post-teardown baseline.
    pair.kill()
    time.sleep(3)
    samples.append(sample("post_teardown", 0, None, drm_allocated=True, include_stub_pid=True))

    # Pair-lock context tokens (the admitted ceiling the pair actually runs).
    context_tokens = read_context_tokens(args.home, args.pair)

    out = {
        "label": args.label,
        "pair": args.pair,
        "home": args.home,
        "uname": samples[0]["uname"],
        "assistant_pid": pair.server.pid,
        "chat_port": chat_port,
        "pair_lock_context_tokens": context_tokens,
        "samples": samples,
        "baseline_sys_used_bytes": baseline_samples[0]["system_used_bytes"],
        "idle_before_sys_used_bytes": next(
            (s["system_used_bytes"] for s in samples if s["phase"] == "idle_before"), 0),
        "system_peak_used_bytes": max(s["system_used_bytes"] for s in samples),
        "system_peak_over_baseline_bytes": (
            max(s["system_used_bytes"] for s in samples)
            - baseline_samples[0]["system_used_bytes"]),
        "pair_resident_cost_bytes": next(
            (s["system_used_bytes"] for s in samples if s["phase"] == "idle_before"), 0)
            - baseline_samples[0]["system_used_bytes"],
        "tree_peak_rss_total_bytes": max(s["rss_total"] for s in samples),
        "tree_peak_pss_total_bytes": max(s["pss_total"] for s in samples),
        "backend_peak_bytes": max(
            (s["backend_peak"].get("peak") if isinstance(s["backend_peak"], dict) else 0) or 0
            for s in samples),
        "backend_peak_available": any(
            isinstance(s["backend_peak"], dict) and "peak" in s["backend_peak"]
            for s in samples),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}", flush=True)
    print(f"baseline_sys_used_MiB={out['baseline_sys_used_bytes']/2**20:.1f}")
    print(f"idle_before_sys_used_MiB={out['idle_before_sys_used_bytes']/2**20:.1f}")
    print(f"pair_resident_cost_MiB={out['pair_resident_cost_bytes']/2**20:.1f}")
    print(f"system_peak_over_baseline_MiB={out['system_peak_over_baseline_bytes']/2**20:.1f}")
    print(f"tree_peak_rss_MiB={out['tree_peak_rss_total_bytes']/2**20:.1f}")
    print(f"tree_peak_pss_MiB={out['tree_peak_pss_total_bytes']/2**20:.1f}")
    print(f"backend_peak_MiB={out['backend_peak_bytes']/2**20:.1f} (available={out['backend_peak_available']})")
    print(f"pair_lock_context_tokens={context_tokens}")

    pair.kill()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()