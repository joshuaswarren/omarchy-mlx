#!/usr/bin/env python3
"""PairGates zero-CPU dispatch trace harness.

Five phases, all under MLX_OMARCHY_TRACE_DISPATCH=1:
  1) chat            — load qwen3.8-2b-4bit, run generate(), record DISPATCH
                       lines and max running count.
  2) decision        — drive a running Laya worker's HTTP /v1/decisions
                       endpoint with a small batch.
  3) cpu-control-1   — mx.set_default_device(mx.cpu), add + eval. Records
                       whether any dispatch fires on CPU stream.
  4) cpu-control-2   — explicit mx.add(a, b, stream=mx.cpu). This is the
                       positive control Main requested: an op that targets
                       the CPU stream explicitly. We record its return
                       value (or error), and whether the trace counter
                       moved.
  5) gpu-reference   — same mx.add but on the GPU stream (default device),
                       to confirm the same op on GPU emits DISPATCH lines.
                       This proves the trace counter is live.

Each phase writes its own log + trace lines + counts. summary.json rolls
them up. Counts: total [rtmod] DISPATCH lines, max running count,
first count, kernel histogram.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import http.client


INFO = "<home>/.local/share/mlx-omarchy/venv/lib/python3.14/site-packages/mlx/bin/mlx-omarchy-info"


def read_counters():
    return json.loads(subprocess.check_output([INFO, "--json"]).decode())


def run_python(code, args, extra_env=None):
    env = {"MLX_OMARCHY_TRACE_DISPATCH": "1", "PYTHONUNBUFFERED": "1"}
    if extra_env:
        env.update(extra_env)
    full_env = {**os.environ, **env}
    proc = subprocess.run(
        ["<home>/.local/share/mlx-omarchy/venv/bin/python",
         "-u", "-c", code, *args],
        env=full_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


def parse_dispatch_lines(text):
    pat = re.compile(r"\[rtmod\] DISPATCH kernel=(\d+) count=(\d+)")
    lines = 0
    max_count = 0
    min_count = None
    first_count = None
    kernels = {}
    for m in pat.finditer(text):
        lines += 1
        c = int(m.group(2))
        if first_count is None:
            first_count = c
        max_count = max(max_count, c)
        if min_count is None or c < min_count:
            min_count = c
        kernels.setdefault(int(m.group(1)), 0)
        kernels[int(m.group(1))] += 1
    return {"lines": lines, "max_count": max_count,
            "min_count": min_count if min_count is not None else 0,
            "first_count": first_count if first_count is not None else 0,
            "kernels": kernels}


def phase_chat(model_path, out_dir):
    code = """
import sys, time, json
import mlx.core as mx
from mlx_lm import load, generate
t0 = time.time()
model, tokenizer = load(sys.argv[1])
t_load = time.time()
prompt = "Say hello in five words or fewer."
result = generate(model, tokenizer, prompt=prompt, max_tokens=32, verbose=False)
t_gen = time.time()
print("LOAD_T={:.3f}".format(t_load - t0))
print("GEN_T={:.3f}".format(t_gen - t_load))
print("RESULT=", json.dumps(result))
print("DONE")
"""
    proc = run_python(code, [model_path])
    log = os.path.join(out_dir, "chat.log")
    with open(log, "w") as f:
        f.write(proc.stdout)
    counts = parse_dispatch_lines(proc.stdout)
    return {"label": "chat", "returncode": proc.returncode,
            "trace_lines": counts["lines"], "max_running_count": counts["max_count"],
            "min_running_count": counts["min_count"], "first_count": counts["first_count"],
            "kernels": counts["kernels"], "log": log,
            "stdout_tail": proc.stdout[-300:]}


def phase_decision_http(laya_url, out_dir):
    """POST /v1/decisions against a running Laya worker. The Laya
    worker is launched by the assistant; pass its http://host:port
    base."""
    code = """
import sys, json, urllib.request
url = sys.argv[1] + "/v1/decisions"
payload = {"questions": [{
    "id": "q1",
    "question": "Pick the bitter option: espresso | tea | juice",
    "options": [{"id":"a","label":"espresso"},{"id":"b","label":"tea"},{"id":"c","label":"juice"}],
    "criteria": "bitterness"
}]}
req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=60)
    body = r.read().decode()
    print("STATUS=", r.status)
    print("BODY_HEAD=", body[:400])
    print("DONE")
except Exception as e:
    print("ERR=", type(e).__name__, str(e)[:300])
    print("DONE")
"""
    proc = run_python(code, [laya_url])
    log = os.path.join(out_dir, "decision.log")
    with open(log, "w") as f:
        f.write(proc.stdout)
    counts = parse_dispatch_lines(proc.stdout)
    return {"label": "decision", "returncode": proc.returncode,
            "trace_lines": counts["lines"], "max_running_count": counts["max_count"],
            "min_running_count": counts["min_count"], "first_count": counts["first_count"],
            "kernels": counts["kernels"], "log": log,
            "stdout_tail": proc.stdout[-300:]}


def phase_cpu_control_default(out_dir):
    code = """
import mlx.core as mx
import sys, json
print("DEVICE_BEFORE=", mx.default_device())
try:
    mx.set_default_device(mx.cpu)
    print("DEVICE_AFTER=", mx.default_device())
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])
    c = a + b
    mx.eval(c)
    print("CPU_DEFAULT_RESULT=", c.tolist())
except Exception as e:
    print("CPU_DEFAULT_ERR=", type(e).__name__, str(e)[:400])
print("DONE")
"""
    proc = run_python(code, [])
    log = os.path.join(out_dir, "cpu-control-default.log")
    with open(log, "w") as f:
        f.write(proc.stdout)
    counts = parse_dispatch_lines(proc.stdout)
    return {"label": "cpu-control-default", "returncode": proc.returncode,
            "trace_lines": counts["lines"], "max_running_count": counts["max_count"],
            "min_running_count": counts["min_count"], "first_count": counts["first_count"],
            "kernels": counts["kernels"], "log": log,
            "stdout_tail": proc.stdout[-300:]}


def phase_cpu_control_stream(out_dir):
    """Main's positive control: mx.add(a, b, stream=mx.cpu)."""
    code = """
import mlx.core as mx
import sys, json
print("DEFAULT_DEVICE=", mx.default_device())
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
try:
    c = mx.add(a, b, stream=mx.cpu)
    mx.eval(c)
    print("CPU_STREAM_RESULT=", c.tolist())
    print("CPU_STREAM_RESULT_TYPE=", type(c).__name__)
    print("CPU_STREAM_DEVICE=", c.device if hasattr(c, "device") else "?")
except Exception as e:
    print("CPU_STREAM_ERR=", type(e).__name__, str(e)[:400])
print("DONE")
"""
    proc = run_python(code, [])
    log = os.path.join(out_dir, "cpu-control-stream.log")
    with open(log, "w") as f:
        f.write(proc.stdout)
    counts = parse_dispatch_lines(proc.stdout)
    return {"label": "cpu-control-stream", "returncode": proc.returncode,
            "trace_lines": counts["lines"], "max_running_count": counts["max_count"],
            "min_running_count": counts["min_count"], "first_count": counts["first_count"],
            "kernels": counts["kernels"], "log": log,
            "stdout_tail": proc.stdout[-300:]}


def phase_gpu_reference(out_dir):
    """Reference: same mx.add on GPU stream. Emits DISPATCH lines so the
    counter is proven live."""
    code = """
import mlx.core as mx
import sys, json
print("DEFAULT_DEVICE=", mx.default_device())
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
try:
    out = mx.add(a, b)  # default stream = gpu
    mx.eval(out)
    print("GPU_RESULT=", out.tolist())
    print("GPU_DEVICE=", out.device if hasattr(out, "device") else "?")
except Exception as e:
    print("GPU_ERR=", type(e).__name__, str(e)[:400])
print("DONE")
"""
    proc = run_python(code, [])
    log = os.path.join(out_dir, "gpu-reference.log")
    with open(log, "w") as f:
        f.write(proc.stdout)
    counts = parse_dispatch_lines(proc.stdout)
    return {"label": "gpu-reference", "returncode": proc.returncode,
            "trace_lines": counts["lines"], "max_running_count": counts["max_count"],
            "min_running_count": counts["min_count"], "first_count": counts["first_count"],
            "kernels": counts["kernels"], "log": log,
            "stdout_tail": proc.stdout[-300:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-model")
    ap.add_argument("--laya-url", help="http://127.0.0.1:<port> base for the Laya worker")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    phases = []
    if args.chat_model:
        phases.append(phase_chat(args.chat_model, args.out))
    if args.laya_url:
        phases.append(phase_decision_http(args.laya_url, args.out))
    phases.append(phase_cpu_control_default(args.out))
    phases.append(phase_cpu_control_stream(args.out))
    phases.append(phase_gpu_reference(args.out))

    summary = {"phases": phases, "counters_before": read_counters(),
               "counters_after": read_counters()}
    out = os.path.join(args.out, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")
    for p in phases:
        print(f"  {p['label']:25s} trace_lines={p['trace_lines']:5d} max_count={p['max_running_count']:6d}")


if __name__ == "__main__":
    main()