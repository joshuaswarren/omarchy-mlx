#!/usr/bin/env python3
"""Card defect capture: drives ONE card turn through the assistant HTTP API
and records every SSE event with full data, plus the final message record.

Usage: card_defect_capture.py --home <home> --repo-serve <serve> --out <json> --label <name> --pair-id <everyday|compact|quality>
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import http.client
import urllib.error
import urllib.request


def call(method, path, body=None, runtime_path=None):
    rt = json.load(open(runtime_path))
    base = f"http://127.0.0.1:{rt['port']}"
    headers = {"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
               "Origin": base, "Content-Type": "application/json"}
    req = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode())


def heartbeat_loop(cid, turn, runtime_path, stop, gaps):
    while not stop.wait(5):
        try:
            call("POST", f"/api/conversations/{cid}/heartbeat",
                 {"turn_id": turn}, runtime_path=runtime_path)
        except Exception:
            pass
        gaps.append(time.monotonic() - started)


def capture_sse(cid, runtime_path, after_seq, events, stop, timeout_s=600):
    """SSE reader thread; appends every event verbatim."""
    rt = json.load(open(runtime_path))
    port = rt["port"]
    started = time.monotonic()
    try:
        while time.monotonic() - started < timeout_s:
            if stop.is_set():
                return
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
                conn.request("GET", f"/api/conversations/{cid}/events?after={after_seq}",
                             headers={"Cookie": rt["cookie"]})
                resp = conn.getresponse()
                if resp.status != 200:
                    conn.close()
                    time.sleep(1)
                    continue
                buf = b""
                while True:
                    line = resp.fp.readline()
                    if not line:
                        break
                    buf += line
                    if buf.endswith(b"\n\n"):
                        for block in buf.split(b"\n\n"):
                            for bline in block.split(b"\n"):
                                if bline.startswith(b"data:"):
                                    ev = json.loads(bline[5:])
                                    events.append({"seq": ev.get("sequence"),
                                                   "type": ev.get("type"),
                                                   "turn_id": ev.get("turn_id"),
                                                   "data": ev.get("data")})
                                    if ev.get("type") in ("done", "error"):
                                        return
                        buf = b""
                conn.close()
            except Exception as e:
                events.append({"stream_error": str(e)[:200]})
                time.sleep(1)
    finally:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair-id", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--repo-serve", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--timeout-s", type=int, default=180)
    args = ap.parse_args()

    home = args.home
    runtime_path = os.path.join(home, "assistant", "application.json")
    venv = "<home>/.local/share/mlx-omarchy/venv/bin/python"
    driver_log = os.path.join(home, "assistant", "pair_driver.log")

    # --resume (boot_id matches; the pair home is persistent).
    do_setup = False
    lock = os.path.join(home, "assistant", "pair-locks", f"{args.pair_id}.json")
    saved_boot_id = ""
    try:
        saved_boot_id = json.load(open(lock)).get("boot_id", "")
    except Exception:
        pass
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            cur_boot_id = f.read().strip()
    except Exception:
        cur_boot_id = ""
    if saved_boot_id != cur_boot_id:
        do_setup = True
    print(f"do_setup={do_setup}", flush=True)

    if os.path.exists(runtime_path):
        os.unlink(runtime_path)
    env = dict(os.environ,
               PYTHONPATH=args.repo_serve + ":" + os.environ.get("PYTHONPATH", ""),
               MLX_OMARCHY_OFFLINE="1",
               MLX_OMARCHY_HOME=home,
               MLX_OMARCHY_PAIR_DEV_QUALIFICATION="1")
    setup_args = ["--pair", args.pair_id, "--yes"] if do_setup else ["--resume"]
    proc = subprocess.Popen(
        [venv, "-m", "mlx_omarchy_assistant",
         "--home", home, "--no-browser"] + setup_args,
        cwd=os.path.dirname(args.repo_serve), env=env,
        stdout=open(driver_log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    def _cleanup():
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        subprocess.run(["bash", "-lc",
                        "ps -eo pid,cmd | grep -E "
                        "'mlx_omarchy_assistant.*PairGates|_mlxlm_server.*PairGates|"
                        "mlx_omarchy_laya.*PairGates' | "
                        "awk '{print $1}' | xargs -r kill -9 2>/dev/null"],
                       timeout=10, check=False)
    import atexit
    atexit.register(_cleanup)

    deadline = time.monotonic() + (1500 if do_setup else 180)
    while not os.path.exists(runtime_path) and time.monotonic() < deadline:
        time.sleep(0.5)
    if not os.path.exists(runtime_path):
        _cleanup()
        print("FAILED: assistant never wrote application.json", flush=True)
        sys.exit(1)

    setup_deadline = time.monotonic() + (1500 if do_setup else 180)
    while time.monotonic() < setup_deadline:
        state = call("GET", "/api/status", runtime_path=runtime_path).get("setup") or {}
        if state.get("state") == "complete":
            break
        if state.get("state") == "error":
            _cleanup()
            raise RuntimeError(f"setup error: {state}")
        time.sleep(1)
    print("setup complete", flush=True)

    # Submit the card turn.
    cid = call("POST", "/api/conversations", {"save": False},
               runtime_path=runtime_path)["id"]
    t0 = time.monotonic()
    turn = call("POST", f"/api/conversations/{cid}/turns",
                {"text": args.prompt, "mode": "chat", "max_tokens": 256},
                runtime_path=runtime_path)["turn_id"]
    print(f"turn submitted: cid={cid} turn={turn}", flush=True)

    # Capture SSE events and poll for completion in parallel.
    events = []
    stop = threading.Event()
    sse = threading.Thread(target=capture_sse,
                           args=(cid, runtime_path, 0, events, stop, args.timeout_s),
                           daemon=True)
    sse.start()

    message = None
    deadline = time.monotonic() + args.timeout_s
    while time.monotonic() < deadline:
        record = call("GET", f"/api/conversations/{cid}", runtime_path=runtime_path)
        message = next((m for m in record.get("messages") or []
                        if m.get("turn_id") == turn
                        and m.get("role") == "assistant"), None)
        if message and message.get("status") in ("complete", "error", "stopped"):
            break
        time.sleep(1)
    stop.set()
    sse.join(timeout=5)
    wall = time.monotonic() - t0

    out = {
        "label": args.label,
        "pair_id": args.pair_id,
        "prompt": args.prompt,
        "wall_s": round(wall, 2),
        "final_message": message,
        "record": call("GET", f"/api/conversations/{cid}", runtime_path=runtime_path),
        "events": events,
        "event_count": len(events),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    text = (message or {}).get("content") or ""
    print(f"result: status={(message or {}).get('status')} text_len={len(text)} "
          f"events={len(events)} wall={wall:.2f}s", flush=True)
    for e in events[:20]:
        print(f"  event: {e.get('type')}: {json.dumps(e.get('data'))[:150]}", flush=True)
    print(f"wrote {args.out}", flush=True)
    _cleanup()


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()