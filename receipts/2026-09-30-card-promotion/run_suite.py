#!/usr/bin/env python3
"""Run a frozen card HELD-OUT suite once per chat model through the real app.

``--suite`` names the fixture (v2 ran as cards_held_out_v2.json, v3 as
cards_held_out_v3.json).  Each chunk starts its own assistant inside the
caller's gpu-turn ticket, sends real HTTP chat turns, records one outcome per
prompt in results/<suite>_<model>.json (sha256 of the fixture bytes, card
types and titles, the reply text) and skips prompts already recorded, so a
reboot or a killed ticket loses no finished work.
"""
import atexit
import hashlib
import threading
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
    "qwen3.5-9b-mlx-4bit": "everyday",
    "qwen3.8-27b-4bit": "quality",
    "qwen3-4b-instruct-2507-4bit": "compact",
    "qwen3.8-2b-4bit": "everyday",  # the pre-8a1e25843 Everyday chat model
}
PYTHON = os.path.join(_HOME, ".local", "share", "mlx-omarchy", "venv", "bin", "python")
RESULTS_DIR = os.path.join(_HOME, "agents", "MarkdownCards", "results")


def home_for(chat_model):
    safe = chat_model.replace("/", "_")
    return os.path.join(HOME_ROOT, safe)


def runtime_for(chat_model):
    return os.path.join(home_for(chat_model), "assistant", "application.json")


def results_path(suite, chat_model):
    return os.path.join(RESULTS_DIR, f"{suite}_{chat_model.replace('/', '_')}.json")


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
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode()[:500]
        except Exception:
            pass
        print(f"HTTP {exc.code} on {method} {path}: {body_text}", flush=True)
        raise


def read_events(runtime, cid, after=0, timeout=30):
    """Yield the conversation's events after ``after`` from the server-sent
    event stream; the server closes it about 15 s in or when no turn is active."""
    rt = json.load(open(runtime))
    base = f"http://127.0.0.1:{rt['port']}"
    request = urllib.request.Request(
        f"{base}/api/conversations/{cid}/events?after={after}",
        headers={"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"], "Origin": base})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").rstrip("\n")
            if line.startswith("data: "):
                yield json.loads(line[6:])


def sync_results(results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = results["_path"] + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(results, fp, indent=2)
        fp.flush()
        try:
            os.fsync(fp.fileno())
        except OSError:
            pass
    os.replace(tmp, results["_path"])
    try:
        os.sync()
    except OSError:
        pass


def load_results(suite, chat_model):
    path = results_path(suite, chat_model)
    if os.path.exists(path):
        with open(path) as fp:
            return json.load(fp), path
    return {"suite_sha256": "", "chat_model": chat_model, "prompts": []}, path


def _heartbeat_loop(cid, turn, runtime, stop, gaps):
    """Keep the turn alive every 5 s from its own thread; the server cancels
    a turn after 20 s without one.  Records the slowest heartbeat call."""
    while not stop.wait(5):
        started = time.monotonic()
        try:
            call("POST", f"/api/conversations/{cid}/heartbeat", {"turn_id": turn}, runtime=runtime)
        except Exception:
            pass
        gaps.append(time.monotonic() - started)


def run_chunk(held_out_path, chat_model, start, end, max_tokens=700,
              timeout_s=600, setup=False, budget_s=None, tag=""):
    chunk_started = time.monotonic()
    raw = open(held_out_path, "rb").read()
    prompts_doc = json.loads(raw)
    suite_sha = hashlib.sha256(raw).hexdigest()
    suite = os.path.splitext(os.path.basename(held_out_path))[0].replace("cards_", "") + tag
    home = home_for(chat_model)
    runtime = runtime_for(chat_model)
    pair_id = PAIR_FOR_MODEL[chat_model]
    results, path = load_results(suite, chat_model)
    results["suite_sha256"] = suite_sha
    results["_path"] = path
    finished_ids = {p["id"] for p in results.get("prompts") or []}

    os.makedirs(home, exist_ok=True)
    boot_id_path = "/proc/sys/kernel/random/boot_id"
    cur_boot_id = (open(boot_id_path).read().strip()
                   if os.path.exists(boot_id_path) else "")
    lock_path = os.path.join(home, "assistant", "pair-locks", f"{pair_id}.json")
    saved_boot_id = ""
    if os.path.exists(lock_path):
        try:
            saved_boot_id = json.load(open(lock_path)).get("boot_id", "")
        except Exception:
            saved_boot_id = ""
    do_setup = saved_boot_id != cur_boot_id  # no lock yet, or locked on an earlier boot
    if do_setup:
        print(f"boot_id mismatch (saved={saved_boot_id[:8]} "
              f"current={cur_boot_id[:8]}); falling back to --setup", flush=True)

    if os.path.exists(runtime):
        os.unlink(runtime)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "serve"),
               MLX_OMARCHY_OFFLINE="1", MLX_OMARCHY_HOME=home,
               MLX_OMARCHY_PAIR_DEV_QUALIFICATION="1")
    safe = chat_model.replace("/", "_")
    log_path = os.path.join(_HOME, "agents", "MarkdownCards",
                            f"server_{suite}_{safe}.log")
    setup_args = ["--pair", pair_id, "--yes"] if do_setup else []
    resume_args = ["--resume"] if not do_setup else []
    server = subprocess.Popen(
        [PYTHON, "-m", "mlx_omarchy_assistant", "--home", home,
         "--no-browser"] + setup_args + resume_args,
        cwd=REPO, env=env,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        start_new_session=True)

    def _kill_server():
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            server.wait(timeout=10)
        except Exception:
            try:
                os.killpg(server.pid, signal.SIGKILL)
            except Exception:
                pass
        try:
            import subprocess as _sp
            _sp.run(["bash", "-lc",
                     "ps -eo pid,cmd | grep -E "
                     "'mlx_omarchy_assistant.*--home.*MarkdownCards|"
                     "_mlxlm_server.*MarkdownCards|"
                     "mlx_omarchy_assistant.gpu_stt.*MarkdownCards' | "
                     "awk '{print $1}' | xargs -r kill -9 2>/dev/null"],
                    timeout=15, check=False)
        except Exception:
            pass

    atexit.register(_kill_server)
    deadline = time.monotonic() + (1500 if do_setup else 180)
    while not os.path.exists(runtime) and time.monotonic() < deadline:
        time.sleep(0.5)
    if not os.path.exists(runtime):
        _kill_server()
        raise RuntimeError("server never wrote application.json")
    setup_deadline = time.monotonic() + (600 if not do_setup else 1500)
    while time.monotonic() < setup_deadline:
        state = call("GET", "/api/status", runtime=runtime).get("setup") or {}
        if state.get("state") == "complete":
            break
        if state.get("state") == "error":
            _kill_server()
            raise RuntimeError(f"setup error: {state}")
        if state.get("state") == "absent":
            _kill_server()
            if os.path.exists(runtime):
                os.unlink(runtime)
            server = subprocess.Popen(
                [PYTHON, "-m", "mlx_omarchy_assistant", "--home", home,
                 "--no-browser", "--pair", pair_id, "--yes"],
                cwd=REPO, env=env,
                stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                start_new_session=True)
            deadline = time.monotonic() + 1500
            while not os.path.exists(runtime) and time.monotonic() < deadline:
                time.sleep(0.5)
            setup_deadline = time.monotonic() + 1500
        time.sleep(1)
    else:
        _kill_server()
        raise RuntimeError("setup did not complete in time")

    try:
        valid_card_kinds = {"checklist", "comparison", "timeline", "facts"}
        prompts = prompts_doc["prompts"][start:end]
        for offset, prompt in enumerate(prompts, start=start):
            if prompt["id"] in finished_ids:
                continue
            if budget_s is not None and time.monotonic() - chunk_started > budget_s:
                print(f"BUDGET spent before {prompt['id']}; next ticket resumes", flush=True)
                break
            cid = call("POST", "/api/conversations", {"save": False},
                       runtime=runtime)["id"]
            started = time.monotonic()
            turn = call("POST", f"/api/conversations/{cid}/turns",
                        {"text": prompt["text"], "mode": "chat",
                         "max_tokens": max_tokens},
                        runtime=runtime)["turn_id"]
            message = None
            stop, gaps = threading.Event(), []
            beat = threading.Thread(target=_heartbeat_loop,
                                    args=(cid, turn, runtime, stop, gaps), daemon=True)
            beat.start()
            deadline = time.monotonic() + timeout_s
            try:
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
            finally:
                stop.set()
                beat.join()
            if not message or message.get("status") not in ("complete", "error", "stopped"):
                print(f"TIMEOUT {prompt['id']} after {timeout_s}s; not recorded", flush=True)
                break
            components = (message or {}).get("components") or []
            try:
                events = list(read_events(runtime, cid))
            except Exception as exc:
                events = [{"type": "error", "turn_id": turn,
                           "data": {"message": f"events unavailable: {exc}"}}]
            turn_events = [e for e in events if e.get("turn_id") == turn
                           and e.get("type") in ("error", "status")]
            types = [c.get("type") for c in components]
            elapsed = time.monotonic() - started
            expect = prompt["expect"]
            kinds_valid = any(t in valid_card_kinds for t in types)
            pass_ = (expect == "card" and kinds_valid) or (
                expect == "none" and not kinds_valid)
            outcome = {
                "id": prompt["id"], "category": prompt["category"],
                "kind": prompt["kind"], "expect": expect,
                "components": types,
                "titles": [c.get("title", "") for c in components],
                "promoted": [c.get("title", "").endswith("(from reply)") for c in components],
                "elapsed_s": round(elapsed, 1),
                "status": (message or {}).get("status"),
                "max_heartbeat_s": round(max(gaps, default=0.0), 2),
                "events": [{"type": e["type"], **(e.get("data") or {})} for e in turn_events][:10],
                "pass": pass_,
                "reply": ((message or {}).get("content") or "")[:6000],
            }
            results.setdefault("prompts", []).append(outcome)
            finished_ids.add(prompt["id"])
            sync_results(results)
            print(f"{suite.upper()} {offset + 1:2d}/{len(prompts_doc['prompts'])} "
                  f"id={prompt['id']} expect={expect} types={types} "
                  f"elapsed={elapsed:.0f}s pass={pass_}", flush=True)
    finally:
        _kill_server()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=36)
    p.add_argument("--suite", required=True,
                   help="fixture path, e.g. tests/fixtures/cards_held_out_v3.json")
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--setup", action="store_true")
    p.add_argument("--tag", default="",
                   help="suffix for the results file, e.g. _fenced for a config run")
    p.add_argument("--budget-s", type=int, default=None,
                   help="start no new prompt after this many seconds")
    args = p.parse_args()
    if args.model not in PAIR_FOR_MODEL:
        sys.exit(f"unknown model {args.model!r}")
    run_chunk(os.path.join(REPO, args.suite), args.model, args.start, args.end,
              max_tokens=args.max_tokens, timeout_s=args.timeout_s,
              setup=args.setup, budget_s=args.budget_s, tag=args.tag)


if __name__ == "__main__":
    # gpu-turn's timeout sends SIGTERM: exit through the finally blocks so the
    # assistant (its own session) is always stopped inside the ticket.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
