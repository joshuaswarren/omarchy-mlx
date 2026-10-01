#!/usr/bin/env python3
"""Worker that runs under the count_cpu gdb session.

Spawned via: gdb -batch -x count_cpu.gdb.py --args <this.py> <label>

Reads stdin lines of "JOB <json>"; for each job runs the workload
(POST /v1/chat/completions, POST /v1/decisions, etc.) against the
running worker, then writes "WORKER_PORT=<port>" to stdout so the
parent harness can drive it.

In the simplest mode (no parent harness), this script:
   1. Starts the worker subprocess (the actual chat server / Laya server / TTS server).
   2. Reads stdin for a single line "READY\n", then
   3. announces its port via "WORKER_PORT=<port>\n" to stdout.
   4. Waits for one more stdin line "DONE\n", then exits.
"""
import os
import socket
import subprocess
import sys
import threading
import time


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_port(port, deadline_s=300):
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def start_chat(model):
    port = free_port()
    cmd = ["<home>/.local/share/mlx-omarchy/venv/bin/python",
           "-u", "-m", "mlx_omarchy_serve._mlxlm_server",
           "--model", model,
           "--host", "127.0.0.1", "--port", str(port),
           "--max-tokens", "512",
           "--decode-concurrency", "1",
           "--prompt-concurrency", "1",
           "--prompt-cache-size", "0"]
    env = os.environ.copy()
    env["PYTHONPATH"] = "<home>/agents/PairGates/worktree/serve:" + env.get("PYTHONPATH", "")
    env["MLX_OMARCHY_SERVE_CONTEXT_LIMIT"] = "4096"
    print(f"WORKER_CMD={' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, env=env), port


def start_decision(model):
    port = free_port()
    code = (f"import sys; from mlx_omarchy_laya.server import serve_main; "
            f"sys.argv=['laya','--model',{model!r},'--host','127.0.0.1',"
            f"'--port',{port!r},'--max-questions','8']; serve_main(sys.argv[1:])")
    cmd = ["<home>/.local/share/mlx-omarchy/venv/bin/python",
           "-u", "-c", code]
    env = os.environ.copy()
    env["PYTHONPATH"] = "<home>/agents/PairGates/worktree/serve:" + env.get("PYTHONPATH", "")
    print(f"WORKER_CMD={' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, env=env), port


def start_tts():
    port = free_port()
    venv_python = "<home>/voice-site/.venv/bin/python"
    if not os.path.exists(venv_python):
        print(f"VOICE_VENV_MISSING {venv_python}", flush=True)
        sys.exit(2)
    model = "mlx-community/Qwen3-TTS-0.6B-CustomVoice-4bit"
    cmd = [venv_python, "-u", "-m", "mlx_audio.server",
           "--model", model, "--host", "127.0.0.1", "--port", str(port)]
    print(f"WORKER_CMD={' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd), port


def main():
    label = sys.argv[1]
    if label == "chat":
        model = sys.argv[2] if len(sys.argv) > 2 else (
            "<home>/.cache/huggingface/hub/models--SiddhJagani"
            "--Qwen3.8-2B-mlx-4Bit/snapshots/0867d98bfb174b042d88461c0e7c97b86b34b381")
        proc, port = start_chat(model)
    elif label == "decision":
        model = sys.argv[2] if len(sys.argv) > 2 else (
            "<home>/agents/PairGates/homes/everyday/models/laya-mlx")
        proc, port = start_decision(model)
    elif label == "tts":
        proc, port = start_tts()
    else:
        print(f"unknown label {label}", flush=True)
        sys.exit(2)

    print(f"WORKER_PID={proc.pid}", flush=True)
    if not wait_port(port, deadline_s=300):
        print("WORKER_LISTEN_TIMEOUT", flush=True)
        proc.terminate()
        sys.exit(3)
    print(f"WORKER_PORT={port}", flush=True)
    print("WORKER_READY", flush=True)

    # Block until parent sends "DONE" on stdin.
    try:
        for line in sys.stdin:
            if line.strip() == "DONE":
                break
    except EOFError:
        pass
    print("WORKER_SHUTDOWN", flush=True)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    print("WORKER_EXITED", flush=True)


if __name__ == "__main__":
    main()