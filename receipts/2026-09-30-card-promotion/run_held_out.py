#!/usr/bin/env python3
"""Run the frozen HELD-OUT card-promotion suite once per chat model.

Resumable across reboots and killed gpu-turn chunks: writes
``results_<model>.json`` on every prompt and skips prompts whose id
is already recorded.  Designed to be wrapped in ``gpu-turn -m 8 --``
with a tight prompt slice (``--start``/``--end``) so each chunk fits
under the M2 5-12 min reboot cadence and the gpu-turn 8 min cap.

Usage (paths via env vars so the script is portable):
    MARKCARDS_HOME=<home> MARKCARDS_VENV=<home>/.local/share/mlx-omarchy/venv \\
        <venv>/bin/python receipts/2026-09-30-card-promotion/run_held_out.py \\
        --model qwen3.8-2b-4bit --start 0 --end 3
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
# MARKCARDS_HOME environment variable before running.
_HOME = os.environ.get("MARKCARDS_HOME", "<home>")
HOME_ROOT = os.path.join(_HOME, "agents", "MarkdownCards", "homes")
PAIR_FOR_MODEL = {
    "qwen3.8-2b-4bit": "everyday",
    "qwen3.8-27b-4bit": "quality",
}
PYTHON = os.path.join(_HOME, ".local", "share", "mlx-omarchy", "venv", "bin", "python")
RESULTS_DIR = os.path.join(_HOME, "agents", "MarkdownCards", "results")


def home_for(chat_model):
    safe = chat_model.replace("/", "_")
    return os.path.join(HOME_ROOT, safe)


def runtime_for(chat_model):
    return os.path.join(home_for(chat_model), "assistant", "application.json")


def results_path(chat_model):
    return os.path.join(RESULTS_DIR, f"held_out_{chat_model.replace('/', '_')}.json")


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


def sync_results(results):
    """Persist ``results`` to disk and fsync to survive a reboot."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = results["_path"] + ".tmp"
    with open(tmp, "w") as fp:
        json.dump(results, fp, indent=2)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(tmp, results["_path"])
    # Belt-and-braces: ask the kernel to flush.
    try:
        os.sync()
    except OSError:
        pass


def load_results(chat_model):
    path = results_path(chat_model)
    if os.path.exists(path):
        with open(path) as fp:
            return json.load(fp), path
    return {"suite_sha256": "", "chat_model": chat_model, "prompts": []}, path


def run_chunk(held_out_path, chat_model, start, end, max_tokens=700,
              timeout_s=300, setup=False):
    prompts_doc = json.load(open(held_out_path))
    suite_sha = prompts_doc.get("sha256", "")
    home = home_for(chat_model)
    runtime = runtime_for(chat_model)
    pair_id = PAIR_FOR_MODEL[chat_model]
    results, path = load_results(chat_model)
    results["suite_sha256"] = suite_sha
    results["_path"] = path
    finished_ids = {p["id"] for p in results.get("prompts") or []}

    os.makedirs(home, exist_ok=True)
    # Always start with --setup so a stale boot_id in the saved pair lock
    # does not silently turn /api/resume into an absent pair.  Setup reuses
    # cached weights when present, so the second-and-later chunk is fast.
    if setup:
        do_setup = True
    else:
        # Without --setup, try --resume but if the boot_id in the saved
        # lock differs from the current /proc/sys/kernel/random/boot_id,
        # resume will refuse; fall back to --setup.
        boot_id_path = "/proc/sys/kernel/random/boot_id"
        cur_boot_id = (open(boot_id_path).read().strip()
                       if os.path.exists(boot_id_path) else "")
        lock_path = os.path.join(home, "assistant", "pair-locks",
                                 f"{pair_id}.json")
        saved_boot_id = ""
        if os.path.exists(lock_path):
            try:
                saved_boot_id = json.load(open(lock_path)).get("boot_id", "")
            except Exception:
                saved_boot_id = ""
        do_setup = bool(saved_boot_id) and saved_boot_id != cur_boot_id
        if do_setup:
            print(f"boot_id mismatch (saved={saved_boot_id[:8]} "
                  f"current={cur_boot_id[:8]}); falling back to --setup",
                  flush=True)

    if os.path.exists(runtime):
        os.unlink(runtime)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "serve"),
               MLX_OMARCHY_OFFLINE="1", MLX_OMARCHY_HOME=home)
    safe = chat_model.replace("/", "_")
    log_path = os.path.join(_HOME, "agents", "MarkdownCards",
                            f"server_{safe}.log")
    setup_args = ["--pair", pair_id, "--yes"] if do_setup else []
    resume_args = ["--resume"] if not do_setup else []
    server = subprocess.Popen(
        [PYTHON, "-m", "mlx_omarchy_assistant", "--home", home,
         "--no-browser"] + setup_args + resume_args,
        cwd=REPO, env=env,
        stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
        start_new_session=True)

    def _kill_server():
        # Hard-kill the whole process group the server started in, then
        # any lingering GPU child (laya, _mlxlm_server, gpu_stt) that
        # somehow outlived the parent.  start_new_session=True made the
        # server its own pgid so killpg reaches every inherited fd.
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
        # Belt-and-braces: hunt for any direct mlx child still holding
        # /dev/dri/renderD* under our user and SIGKILL them.
        try:
            import subprocess as _sp
            _sp.run(["bash", "-lc",
                     "fuser -k /dev/dri/renderD128 /dev/dri/renderD129 2>/dev/null"],
                    timeout=15, check=False)
        except Exception:
            pass

    try:
        deadline = time.monotonic() + 180 if not do_setup else 1500
        while not os.path.exists(runtime) and time.monotonic() < deadline:
            time.sleep(0.5)
        if not os.path.exists(runtime):
            raise RuntimeError("server never wrote application.json")
        setup_deadline = time.monotonic() + (600 if not do_setup else 1500)
        while time.monotonic() < setup_deadline:
            state = call("GET", "/api/status", runtime=runtime).get("setup") or {}
            if state.get("state") == "complete":
                break
            if state.get("state") == "error":
                raise RuntimeError(f"setup error: {state}")
            if state.get("state") == "absent":
                # /api/resume found no usable pair; re-issue with --pair.
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
            raise RuntimeError("setup did not complete in time")

    try:
        valid_card_kinds = {"checklist", "comparison", "timeline", "facts"}
        prompts = prompts_doc["prompts"][start:end]
        for offset, prompt in enumerate(prompts, start=start):
            if prompt["id"] in finished_ids:
                print(f"SKIP {prompt['id']} already recorded", flush=True)
                continue
            cid = call("POST", "/api/conversations", {"save": False},
                       runtime=runtime)["id"]
            started = time.monotonic()
            turn = call("POST", f"/api/conversations/{cid}/turns",
                        {"text": prompt["text"], "mode": "chat",
                         "max_tokens": max_tokens},
                        runtime=runtime)["turn_id"]
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
            results.setdefault("prompts", []).append(outcome)
            finished_ids.add(prompt["id"])
            sync_results(results)
            print(f"HELD_OUT {offset + 1:2d}/{len(prompts_doc['prompts'])} "
                  f"id={prompt['id']} expect={expect} types={types} "
                  f"elapsed={elapsed:.0f}s pass={pass_}",
                  flush=True)
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
    p.add_argument("--model", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=6,
                   help="exclusive end index into the prompts list")
    p.add_argument("--held-out", default=os.path.join(
        REPO, "tests", "fixtures", "cards_held_out.json"))
    p.add_argument("--max-tokens", type=int, default=700)
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument("--setup", action="store_true",
                   help="run --pair setup before this chunk")
    args = p.parse_args()
    if args.model not in PAIR_FOR_MODEL:
        sys.exit(f"unknown model {args.model!r}")
    run_chunk(args.held_out, args.model, args.start, args.end,
              max_tokens=args.max_tokens, timeout_s=args.timeout_s,
              setup=args.setup)


if __name__ == "__main__":
    main()
