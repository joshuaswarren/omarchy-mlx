#!/usr/bin/env python3
"""PairGates quality27b idle-perf harness, via the assistant HTTP API.

Resumes the saved 27B pair (skips 27B setup if pair-locks exist).
Drives turns through /api/conversations/{cid}/turns, measures prefill
tok/s, TTFT, decode tok/s, first-visible-text latency, 5 card turns.
Captures SSE events for each turn so a silent empty reply can be
root-caused (token cap / repetition stop / speech-yield wait /
harness timeout).

Usage: quality27b_perf_api.py --home <home> --repo-serve <serve> --out <dir>
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def parse_meminfo():
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            if ":" in line:
                k, _, v = line.partition(":")
                out[k.strip()] = v.strip()
    return out


def meminfo_bytes(mi, key):
    s = mi.get(key)
    if not s:
        return 0
    parts = s.split()
    n = int(parts[0])
    if len(parts) > 1 and parts[1].lower() == "kb":
        n *= 1024
    return n


class Assistant:
    def __init__(self, pair_id, home, repo_serve):
        self.pair_id = pair_id
        self.home = home
        self.repo_serve = repo_serve
        self.runtime = os.path.join(home, "assistant", "application.json")
        self.server = None
        self.log_path = os.path.join(home, "assistant", "perf_driver.log")

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
            [os.path.expanduser("~/.local/share/mlx-omarchy/venv/bin/python"),
             "-m", "mlx_omarchy_assistant",
             "--home", self.home, "--no-browser"] + args,
            cwd=os.path.dirname(self.repo_serve),
            env=env,
            stdout=open(self.log_path, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True)
        _KILL_LIST.append(self.server)
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
                 {"turn_id": turn}, runtime=runtime)
        except Exception:
            pass
        gaps.append(time.monotonic() - started)


def stream_events(cid, runtime, after_seq=0, stop=None, timeout_s=600):
    """Read the SSE stream, yielding (event_type, data_dict, monotonic_ts)."""
    import http.client
    rt = json.load(open(runtime))
    base = f"127.0.0.1:{rt['port']}"
    conn = http.client.HTTPConnection("127.0.0.1", rt["port"], timeout=300)
    path = f"/api/conversations/{cid}/events?after={after_seq}"
    conn.request("GET", path, headers={"Cookie": rt["cookie"]})
    r = conn.getresponse()
    buf = b""
    started = time.monotonic()
    try:
        while time.monotonic() - started < timeout_s:
            if stop and stop.is_set():
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
                        yield event.get("type"), event.get("data") or {}, time.monotonic()
    finally:
        conn.close()


def wait_turn_with_events(cid, turn, runtime, timeout_s=600):
    """Poll the message record AND capture SSE events. Returns (message, events)."""
    events = []
    stop = threading.Event()
    ev_ended = threading.Event()

    def _drain():
        try:
            for etype, data, ts in stream_events(cid, runtime, stop=stop, timeout_s=timeout_s):
                events.append({"type": etype, "data_head": json.dumps(data)[:400], "ts": ts})
                if etype in ("done", "error"):
                    ev_ended.set()
                    return
        except Exception as e:
            events.append({"type": "stream_error", "data_head": str(e)[:200]})

    t = threading.Thread(target=_drain, daemon=True)
    t.start()

    message = None
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            record = call("GET", f"/api/conversations/{cid}", runtime=runtime)
            message = next((m for m in record.get("messages") or []
                            if m.get("turn_id") == turn
                            and m.get("role") == "assistant"), None)
            if message and message.get("status") in ("complete", "error", "stopped"):
                break
            time.sleep(0.5)
    finally:
        stop.set()
        t.join(timeout=5)
    return message, events


def heartbeat_loop(cid, turn, runtime, stop, gaps):
    while not stop.wait(5):
        try:
            call("POST", f"/api/conversations/{cid}/heartbeat",
                 {"turn_id": turn}, runtime=runtime)
        except Exception:
            pass


def submit_turn(runtime, text, mode="chat", max_tokens=256, options=None, criteria=None):
    cid = call("POST", "/api/conversations", {"save": False}, runtime=runtime)["id"]
    body = {"text": text, "mode": mode, "max_tokens": max_tokens}
    if options:
        body["options"] = options
    if criteria:
        body["criteria"] = criteria
    turn = call("POST", f"/api/conversations/{cid}/turns", body,
                runtime=runtime)["turn_id"]
    return cid, turn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--home", required=True)
    ap.add_argument("--repo-serve", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout-s", type=int, default=600)
    args = ap.parse_args()

    pair = Assistant("quality", args.home, args.repo_serve)
    do_setup = pair.boot_id_mismatch()
    print(f"boot_id mismatch={do_setup}", flush=True)
    pair.start(do_setup)

    setup_deadline = time.monotonic() + (1500 if do_setup else 180)
    while time.monotonic() < setup_deadline:
        state = call("GET", "/api/status", runtime=pair.runtime).get("setup") or {}
        if state.get("state") == "complete":
            break
        if state.get("state") == "error":
            pair.kill()
            raise RuntimeError(f"setup error: {state}")
        time.sleep(1)
    print("setup complete", flush=True)

    filler = ("The quick brown fox jumps over the lazy dog. " * 30).strip()
    # Prompts of exactly 512/2048/170 tokens: grow the filler until the
    # tokenizer counts the target (measured, not assumed). Falls back
    # to a char/4 heuristic only if no tokenizer loads.
    _tok = None
    try:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3-0.6B", trust_remote_code=False)
    except Exception as e:
        print(f"tokenizer unavailable ({e}); using char/4 heuristic")

    def _tok_count(text):
        if _tok is not None:
            return len(_tok(text)["input_ids"])
        return len(text) // 4

    def _build_exact(target):
        text = filler
        while _tok_count(text) < target:
            text = text + " " + filler
        return text

    prefill_512 = _build_exact(512)
    prefill_2048 = _build_exact(2048)
    ttft_prompt = _build_exact(170)
    decode_prompt = "Repeat the last sentence five times verbatim."
    card_prompts = [
        "Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos 21.0M.",
        "Show a bar chart of monthly rainfall (mm): Jan 30, Feb 25, Mar 35, Apr 50, May 60, Jun 70, Jul 75, Aug 70, Sep 55, Oct 45, Nov 35, Dec 30.",
        "Plot a timeline of these events: 2020 founding, 2021 first model, 2023 series A, 2024 expansion.",
        "Make a checklist for launching a new website.",
        "Create a form for collecting customer feedback with rating, comments, and email fields.",
    ]
    results = []

    def measure(label, prompt, max_tokens, mode="chat", options=None, criteria=None):
        cid, turn = submit_turn(pair.runtime, prompt, mode=mode,
                                max_tokens=max_tokens, options=options,
                                criteria=criteria)
        t0 = time.monotonic()
        stop, gaps = threading.Event(), []
        beat = threading.Thread(target=heartbeat_loop,
                                args=(cid, turn, pair.runtime, stop, gaps),
                                daemon=True)
        beat.start()
        try:
            msg, events = wait_turn_with_events(cid, turn, pair.runtime,
                                                timeout_s=args.timeout_s)
        finally:
            stop.set()
            beat.join()
        wall = time.monotonic() - t0
        text = (msg or {}).get("content") or ""
        # first-visible-text and first-visible-component from SSE events
        fvt = None
        fvc = None
        for e in events:
            if e["type"] == "text" and e["data_head"] and e["data_head"] != '""':
                fvt = e["ts"] - t0
                break
        for e in events:
            if e["type"] == "component":
                fvc = e["ts"] - t0
                break
        try:
            text_tokens = _tok_count(text)
        except NameError:
            text_tokens = len(text) // 4
        decode_tok_s = (text_tokens / (wall - fvt)
                        if fvt and wall > fvt and text_tokens else None)
        r = {
            "label": label,
            "wall_s": round(wall, 2),
            "first_visible_text_s": round(fvt, 3) if fvt else None,
            "first_visible_component_s": round(fvc, 3) if fvc else None,
            "text_tokens": text_tokens,
            "decode_tok_s": round(decode_tok_s, 3) if decode_tok_s else None,
            "text_len": len(text),
            "status": (msg or {}).get("status"),
            "prompt_chars": len(prompt),
            "prompt_tokens": _tok_count(prompt),
            "max_tokens": max_tokens,
            "mode": mode,
            "reply_head": text[:200],
            "events_captured": len(events),
            "events_sample": events[:10],
            "components": [c.get("type") for c in (msg or {}).get("components") or []],
        }
        results.append(r)
        print(f"  {label:30s} wall={wall:7.2f}s fvt={r['first_visible_text_s']} "
              f"text_len={len(text)} status={r['status']}", flush=True)
        return r

    # Prefill 512 via max_tokens=1 (prefill-bound).
    measure("prefill_512", prefill_512, 1)
    # Prefill 2048 via max_tokens=1.
    measure("prefill_2048", prefill_2048, 1)
    # TTFT 170 tokens.
    measure("ttft_170", ttft_prompt, 64)
    # Decode 128 after 512.
    measure("decode_128_after_512", prefill_512, 128)
    # 5 card turns.
    for i, p in enumerate(card_prompts):
        measure(f"card_turn_{i+1}", p, 256)

    # Additional: capture the 4B/9B card defect prompts on the 27B to
    # check that the same prompt produces a non-empty reply.
    measure("card_pop_defect_prompt",
            "Show a chart of population for: Paris 2.1M, Tokyo 13.9M, Lagos 21.0M.",
            256)

    pair.kill()
    out = {
        "label": "quality27b-perf",
        "uname": subprocess.check_output(["uname", "-r"]).decode().strip(),
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()