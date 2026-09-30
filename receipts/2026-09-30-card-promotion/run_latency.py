#!/usr/bin/env python3
"""First-text latency probe on the 2B (30 warm turns).

Starts the assistant server (with --setup if the saved boot_id
mismatch), waits for setup == "complete", then drives 30 warm
turns and records first-text wall-clock per turn.

Resumable across reboots and killed gpu-turn tickets: it inherits the
same boot_id-mismatch fallback and hard-kill helper as run_held_out.py.
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


def _kill_server(server):
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


def start_server(chat_model):
    """Start (or resume) the assistant server. Returns (process, runtime_path)."""
    home = home_for(chat_model)
    runtime = runtime_for(chat_model)
    pair_id = PAIR_FOR_MODEL[chat_model]

    # boot_id mismatch fallback
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

    if os.path.exists(runtime):
        os.unlink(runtime)
    env = dict(os.environ, PYTHONPATH=os.path.join(REPO, "serve"),
               MLX_OMARCHY_OFFLINE="1", MLX_OMARCHY_HOME=home,
               MLX_OMARCHY_PAIR_DEV_QUALIFICATION="1")
    setup_args = ["--pair", pair_id, "--yes"] if do_setup else []
    resume_args = ["--resume"] if not do_setup else []
    server_log = os.path.join(_HOME, "agents", "MarkdownCards",
                              f"latency_server_{chat_model.replace('/', '_')}.log")
    server = subprocess.Popen(
        [PYTHON, "-m", "mlx_omarchy_assistant", "--home", home,
         "--no-browser"] + setup_args + resume_args,
        cwd=REPO, env=env,
        stdout=open(server_log, "w"), stderr=subprocess.STDOUT,
        start_new_session=True)
    return server, runtime


def first_text_for(turn_url, runtime, turn, timeout_s=120):
    rt = json.load(open(runtime))
    base = f"http://127.0.0.1:{rt['port']}"
    headers = {"Cookie": rt["cookie"], "X-Assistant-CSRF": rt["csrf"],
               "Origin": base}
    started = time.monotonic()
    deadline = started + timeout_s
    cursor = 0
    seen_first_text = False
    while time.monotonic() < deadline:
        url = f"{base}{turn_url}?after={cursor}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                body = json.loads(resp.read().decode())
        except Exception:
            time.sleep(0.1)
            continue
        for event in body.get("events") or []:
            cursor = event.get("seq", cursor) + 1
            if event.get("type") == "text":
                if not seen_first_text:
                    return time.monotonic() - started, event
                seen_first_text = True
            if event.get("type") == "done":
                return None, None
        time.sleep(0.05)
    return None, None


def host_state():
    """Load average, GPU-capable processes and wheel provenance, recorded
    before the server starts so a foreign GPU user is visible in the result."""
    def sh(cmd):
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    return {
        "loadavg": open("/proc/loadavg").read().strip(),
        "boot_id": open("/proc/sys/kernel/random/boot_id").read().strip(),
        "gpu_processes": sh("ps -eo pid,etime,cmd | grep -E 'python|mlx' | grep -v grep"),
        "provenance": sh(f"{PYTHON} {os.path.join(REPO, 'scripts', 'mlx_provenance.py')} 2>&1"),
    }


def run(model, n_turns=30, warmup=3):
    before = host_state()
    print(json.dumps(before, indent=1), flush=True)
    server, runtime = start_server(model)
    try:
        # Wait for runtime file + setup complete.
        deadline = time.monotonic() + 300
        while not os.path.exists(runtime) and time.monotonic() < deadline:
            time.sleep(0.5)
        if not os.path.exists(runtime):
            raise RuntimeError("server never wrote application.json")
        deadline = time.monotonic() + 1500
        while time.monotonic() < deadline:
            state = call("GET", "/api/status", runtime=runtime).get("setup") or {}
            if state.get("state") == "complete":
                break
            if state.get("state") == "error":
                raise RuntimeError(f"setup error: {state}")
            time.sleep(1)
        else:
            raise RuntimeError("setup did not complete in time")

        first_text_ms = []
        for i in range(-warmup, n_turns):
            cid = call("POST", "/api/conversations", {"save": False},
                       runtime=runtime)["id"]
            text = f"Tell me a one-line fact about number {i + 100}."
            started = time.monotonic()
            turn = call("POST", f"/api/conversations/{cid}/turns",
                        {"text": text, "mode": "chat", "max_tokens": 256},
                        runtime=runtime)["turn_id"]
            elapsed, event = first_text_for(
                f"/api/conversations/{cid}/events", runtime, turn)
            if i >= 0 and elapsed is not None:
                first_text_ms.append(round(elapsed * 1000, 1))
                print(f"turn {i + 1:2d}: {elapsed * 1000:.0f} ms",
                      flush=True)
        first_text_ms.sort()
        p50 = first_text_ms[int(0.50 * len(first_text_ms))]
        p95 = first_text_ms[int(0.95 * len(first_text_ms))]
        print(f"\nFIRST_TEXT_MS p50={p50} p95={p95} gate=2000")
        summary = {
            "model": model, "n": len(first_text_ms), "host_before": before,
            "host_after": host_state(),
            "samples_ms": first_text_ms,
            "p50_ms": p50, "p95_ms": p95, "gate_ms": 2000,
            "gate_pass": p95 <= 2000,
        }
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, f"latency_{model.replace('/', '_')}.json")
        with open(out, "w") as fp:
            json.dump(summary, fp, indent=2)
        return summary
    finally:
        _kill_server(server)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.8-2b-4bit")
    p.add_argument("--n", type=int, default=30)
    args = p.parse_args()
    run(args.model, args.n)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    main()
