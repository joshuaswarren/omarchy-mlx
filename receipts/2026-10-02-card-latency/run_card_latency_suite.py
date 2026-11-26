#!/usr/bin/env python3
"""Card suite runner with time-to-first-component timing.

Same scoring as receipts/2026-09-30-card-promotion/run_suite.py (valid
card on card-worthy prompts, none on plain/near-miss) plus, per prompt,
first_visible_text_s and first_visible_component_s from the SSE stream
(reconnecting cursor; the server closes each events connection after
15 s).  Checkpoints every prompt, so a reboot or an expired gpu-turn
ticket resumes where it stopped.

Usage:
  run_card_latency_suite.py --pair everyday9b --home <dir> \
      --repo-serve <dir> --suite tests/fixtures/cards_held_out_v4.json \
      --tag cand --out results.json [--start 0 --end 36] [--budget-s 750]
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

VALID_CARD_KINDS = {"checklist", "comparison", "timeline", "facts"}
PYTHON = os.path.expanduser("~/.local/share/mlx-omarchy/venv/bin/python")


def call(method, path, body=None, runtime=None):
    rt = json.load(open(runtime))
    base = f"http://127.0.0.1:{rt['port']}"
    headers = {"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
               "Origin": base, "Content-Type": "application/json"}
    request = urllib.request.Request(
        base + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} on {method} {path}: "
              f"{exc.read().decode()[:300]}", flush=True)
        raise


def heartbeat_loop(cid, turn, runtime, stop):
    while not stop.wait(2):
        try:
            call("POST", f"/api/conversations/{cid}/heartbeat",
                 {"turn_id": turn}, runtime=runtime)
        except Exception:
            pass


def stream_events(cid, runtime, stop, timeout_s):
    """Yield (type, data, ts) from the events endpoint, reconnecting."""
    import http.client
    rt = json.load(open(runtime))
    after = 0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not stop.is_set():
        conn = None
        try:
            conn = http.client.HTTPConnection("127.0.0.1", rt["port"],
                                              timeout=300)
            conn.request("GET",
                         f"/api/conversations/{cid}/events?after={after}",
                         headers={"Cookie": rt["cookie"]})
            r = conn.getresponse()
            buf = b""
            while time.monotonic() < deadline and not stop.is_set():
                chunk = r.read1(65536) if hasattr(r, "read1") else r.read(65536)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    block, buf = buf.split(b"\n\n", 1)
                    for line in block.split(b"\n"):
                        if line.startswith(b"data:"):
                            event = json.loads(line[5:])
                            seq = event.get("sequence")
                            if isinstance(seq, int) and seq > after:
                                after = seq
                            yield (event.get("type"), event.get("data") or {},
                                   time.monotonic())
        except Exception:
            pass
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass
        if stop.is_set():
            break
        time.sleep(0.2)


def boot_server(pair_id, home, repo_serve, do_setup):
    runtime = os.path.join(home, "assistant", "application.json")
    if os.path.exists(runtime):
        os.unlink(runtime)
    env = dict(os.environ, PYTHONPATH=repo_serve,
               MLX_OMARCHY_OFFLINE="1", MLX_OMARCHY_HOME=home,
               MLX_OMARCHY_PAIR_DEV_QUALIFICATION="1")
    log = open(os.path.join(home, "assistant", "suite_server.log"), "a")
    args = [PYTHON, "-m", "mlx_omarchy_assistant", "--home", home,
            "--no-browser"]
    args += (["--pair", pair_id, "--yes"] if do_setup else ["--resume"])
    server = subprocess.Popen(args, cwd=os.path.dirname(repo_serve), env=env,
                              stdout=log, stderr=subprocess.STDOUT,
                              start_new_session=True)

    def kill():
        try:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=10)
        except Exception:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except Exception:
                pass

    deadline = time.monotonic() + (1500 if do_setup else 300)
    while not os.path.exists(runtime) and time.monotonic() < deadline:
        time.sleep(0.5)
    if not os.path.exists(runtime):
        kill()
        raise RuntimeError("server never wrote application.json")
    setup_deadline = time.monotonic() + 1500
    while time.monotonic() < setup_deadline:
        state = call("GET", "/api/status", runtime=runtime).get("setup") or {}
        if state.get("state") == "complete":
            return server, kill, runtime
        if state.get("state") == "error":
            kill()
            raise RuntimeError(f"setup error: {state}")
        time.sleep(1)
    kill()
    raise RuntimeError("setup did not complete in time")


def sync(results):
    path = results["_path"]
    tmp = path + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(results, fp, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, path)


def summarize(results):
    rows = [r for r in results["prompts"] if r["category"] == "card-worthy"]
    others = [r for r in results["prompts"] if r["category"] != "card-worthy"]
    cards = sum(1 for r in rows if r["pass"])
    spurious = sum(1 for r in others if not r["pass"])
    comp = sorted(r["first_component_s"] for r in rows
                  if r["first_component_s"] is not None)
    text = sorted(r["first_text_s"] for r in results["prompts"]
                  if r["first_text_s"] is not None)

    def pct(values, q):
        if not values:
            return None
        idx = min(len(values) - 1, int(round(q / 100 * (len(values) - 1))))
        return round(values[idx], 2)

    return {
        "valid_cards": f"{cards}/{len(rows)}",
        "spurious": f"{spurious}/{len(others)}",
        "first_component_median_s": pct(comp, 50),
        "first_component_p95_s": pct(comp, 95),
        "first_text_p95_s": pct(text, 95),
        "first_text_median_s": pct(text, 50),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True)
    ap.add_argument("--home", required=True)
    ap.add_argument("--repo-serve", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=36)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--budget-s", type=int, default=None)
    ap.add_argument("--setup", action="store_true")
    args = ap.parse_args()

    suite_bytes = open(args.suite, "rb").read()
    prompts_doc = json.loads(suite_bytes)
    suite_sha = hashlib.sha256(suite_bytes).hexdigest()

    if os.path.exists(args.out):
        results = json.load(open(args.out))
    else:
        results = {"suite_sha256": suite_sha, "suite": args.suite,
                   "tag": args.tag, "pair": args.pair, "prompts": []}
    results["_path"] = args.out
    finished = {p["id"] for p in results["prompts"]}

    lock = os.path.join(args.home, "assistant", "pair-locks",
                        f"{args.pair}.json")
    saved_boot = ""
    try:
        saved_boot = json.load(open(lock)).get("boot_id", "")
    except Exception:
        saved_boot = ""
    cur_boot = open("/proc/sys/kernel/random/boot_id").read().strip()
    do_setup = args.setup or (bool(saved_boot) and saved_boot != cur_boot)

    server, kill, runtime = boot_server(args.pair, args.home,
                                        args.repo_serve, do_setup)
    lane_start = time.monotonic()
    try:
        prompts = prompts_doc["prompts"][args.start:args.end]
        for offset, prompt in enumerate(prompts, start=args.start):
            if prompt["id"] in finished:
                print(f"SKIP {prompt['id']}", flush=True)
                continue
            if (args.budget_s
                    and time.monotonic() - lane_start > args.budget_s):
                print("BUDGET exhausted; checkpoint and exit", flush=True)
                break
            cid = call("POST", "/api/conversations", {"save": False},
                       runtime=runtime)["id"]
            started = time.monotonic()
            turn = call("POST", f"/api/conversations/{cid}/turns",
                        {"text": prompt["text"], "mode": "chat",
                         "max_tokens": args.max_tokens},
                        runtime=runtime)["turn_id"]
            stop = threading.Event()
            beat = threading.Thread(target=heartbeat_loop,
                                    args=(cid, turn, runtime, stop),
                                    daemon=True)
            beat.start()
            first_text = None
            first_component = None

            def drain():
                nonlocal first_text, first_component
                for etype, data, ts in stream_events(
                        cid, runtime, stop, args.timeout_s):
                    if first_text is None and etype == "text" and data:
                        first_text = ts - started
                    if first_component is None and etype == "component":
                        first_component = ts - started

            drain_t = threading.Thread(target=drain, daemon=True)
            drain_t.start()
            message = None
            deadline = time.monotonic() + args.timeout_s
            while time.monotonic() < deadline:
                record = call("GET", f"/api/conversations/{cid}",
                              runtime=runtime)
                message = next((m for m in record.get("messages") or []
                                if m.get("turn_id") == turn
                                and m.get("role") == "assistant"), None)
                if message and message.get("status") in (
                        "complete", "error", "stopped"):
                    break
                time.sleep(1)
            stop.set()
            beat.join(timeout=5)
            drain_t.join(timeout=5)
            components = (message or {}).get("components") or []
            types = [c.get("type") for c in components]
            elapsed = time.monotonic() - started
            # cards_dev-generation fixtures carry no per-prompt "expect";
            # the category alone decides (card-worthy vs plain/near-miss).
            expect = prompt.get("expect") or (
                "card" if prompt["category"] == "card-worthy" else "none")
            kinds_valid = any(t in VALID_CARD_KINDS for t in types)
            passed = (expect == "card" and kinds_valid) or (
                expect == "none" and not kinds_valid)
            outcome = {
                "id": prompt["id"], "category": prompt["category"],
                "kind": prompt["kind"], "expect": expect,
                "components": types, "elapsed_s": round(elapsed, 1),
                "first_text_s": (round(first_text, 3)
                                 if first_text is not None else None),
                "first_component_s": (round(first_component, 3)
                                      if first_component is not None
                                      else None),
                "status": (message or {}).get("status"),
                "pass": passed,
            }
            results["prompts"].append(outcome)
            sync(results)
            print(f"{args.tag or 'suite'} {offset + 1:2d}/"
                  f"{len(prompts_doc['prompts'])} id={prompt['id']} "
                  f"expect={expect} types={types} "
                  f"text={outcome['first_text_s']} "
                  f"comp={outcome['first_component_s']} "
                  f"elapsed={elapsed:.0f}s pass={passed}", flush=True)
    finally:
        kill()
    print(json.dumps(summarize(results), indent=2), flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
