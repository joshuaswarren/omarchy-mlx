#!/usr/bin/env python3
"""Calibrate kBatchWorkBudget for issue #19.

Drives the live mlx-omarchy with the profiling harness on (writes an
NDJSON stream of "k":"d" dispatch GPU ticks + "k":"s"/"q" submission
boundaries), runs a short 4B decode + a 512-token prefill, then groups
dispatch ticks by their parent submission id to compute ms-per-submit
histograms. Two passes at BATCH_WORK=0 (cap off, baseline) and at the
candidate budget; the default is the smallest one that lands typical
decode submissions in 2-6 ms.

Usage:
  python3 scripts/subcap_calibrate.py \
      --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-Instruct-2507-4bit/snapshots/<sha> \
      --tokens 64 --prefill-tokens 512 --budgets 0,20000,40000,80000 \
      --profile /tmp/sc-profile.jsonl

The mlx_lm loader + a short fixed prompt keeps the run tightly bounded so
the gpu-turn window stays short. Output: medians, p50/p95/p99/max ms per
submit, and a recommended kBatchWorkBudget value.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


PROMPT = (
    "Tell me about the history of computing in three short paragraphs: "
    "the early mechanical era, the transistor era, and the modern GPU era."
)


def ms_per_submit(profile_path, grace_seconds=10.0):
    """Group per-dispatch GPU ticks by submission id; sum nanoseconds
    per submission. Returns list of float ms in submission order."""
    # Dispatch ticks record t0,t1 in raw ticks (period_ns from meta).
    # Initial: wait briefly for the file to appear (driver still booting).
    deadline = time.time() + 60
    while time.time() < deadline and not Path(profile_path).exists():
        time.sleep(0.2)
    if not Path(profile_path).exists():
        raise RuntimeError(f"profile file not written: {profile_path}")
    # Poll until no growth for grace_seconds, capped at the deadline.
    deadline = time.time() + 180
    period_ns = None
    valid_bits = None
    dispatches = []
    line_offset = 0  # bytes consumed; only parse new lines each pass
    last_size = -1
    last_growth_at = time.time()
    while True:
        try:
            with open(profile_path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(line_offset)
                new_bytes = f.read()
                line_offset += len(new_bytes)
                new_data = new_bytes.decode("utf-8", errors="replace")
        except FileNotFoundError:
            new_data = ""
            size = -1
        if size > last_size:
            last_growth_at = time.time()
            last_size = size
        for line in new_data.splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = e.get("k")
            if k == "meta":
                period_ns = e.get("period_ns", period_ns)
                valid_bits = e.get("valid_bits", valid_bits)
            elif k == "d":
                sid = e.get("s")
                if sid is None:
                    continue
                t0 = e.get("t0")
                t1 = e.get("t1")
                if t0 is None or t1 is None or period_ns is None:
                    continue
                mask = (1 << (valid_bits or 48)) - 1
                ticks = (t1 - t0) & mask
                # period_ns is nanoseconds per timestamp tick (Honeykrisp
                # reports 1.0; some drivers report 0.x).
                gpu_ns = ticks * period_ns
                dispatches.append((sid, gpu_ns))
        if time.time() - last_growth_at >= grace_seconds:
            break
        if time.time() > deadline:
            break
        time.sleep(0.5)
    sums = {}
    for sid, ns in dispatches:
        sums[sid] = sums.get(sid, 0.0) + ns
    out = []
    for sid in sorted(sums):
        out.append(sums[sid] / 1e6)
    return out


def percentile(vals, p):
    if not vals:
        return 0.0
    return statistics.quantiles(vals, n=100, method="inclusive")[p - 1]


def report(label, vals):
    if not vals:
        print(f"{label}: (empty)")
        return None
    print(
        f"{label}: n={len(vals)} p50={percentile(vals, 50):.3f} ms "
        f"p95={percentile(vals, 95):.3f} ms p99={percentile(vals, 99):.3f} ms "
        f"max={max(vals):.3f} ms mean={statistics.mean(vals):.3f} ms"
    )
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--prefill-tokens", type=int, default=512)
    ap.add_argument("--budgets", required=True,
                    help="comma-separated BATCH_WORK values to try")
    ap.add_argument("--profile", default="/tmp/sc-profile.jsonl")
    ap.add_argument("--venv-python",
                    default="/tmp/sc-venv/bin/python")
    args = ap.parse_args()

    # Build the inline mlx_lm script. It runs decode (--tokens after
    # prefill) and writes the GPU-time events for the wheel. The profile
    # file is reused across passes (each pass overwrites).
    driver = """
import os, sys, json, time, argparse
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import mlx.core as mx
from mlx_lm import load, generate

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--prefill-tokens", type=int, default=512)
ap.add_argument("--prompt",
                default="Tell me about the history of computing in three short paragraphs.")
args = ap.parse_args()
model, tok = load(args.model)
# Warm-up compile
_ = generate(model, tok, prompt="hello", max_tokens=4, verbose=False)
t0 = time.monotonic()
resp = generate(
    model, tok,
    prompt=args.prompt,
    max_tokens=args.tokens,
    prefill_step_size=args.prefill_tokens,
    verbose=False,
)
t1 = time.monotonic()
print(f"decode_tps={args.tokens / (t1 - t0):.3f} over {args.tokens} tokens "
      f"({t1 - t0:.2f}s wall)", flush=True)
"""
    script_path = Path("/tmp/sc-driver.py")
    script_path.write_text(driver)

    results = {}
    for budget_str in args.budgets.split(","):
        budget = budget_str.strip()
        env = os.environ.copy()
        env["MLX_OMARCHY_GPU_PROFILE"] = args.profile + f".{budget}"
        env["MLX_OMARCHY_GPU_PROFILE_LABEL"] = f"budget={budget}"
        env["MLX_OMARCHY_BATCH_WORK"] = budget
        env.pop("MLX_OMARCHY_DEFER_COMMIT", None)
        # Remove any stale file
        try:
            Path(args.profile + f".{budget}").unlink()
        except FileNotFoundError:
            pass
        cmd = [
            args.venv_python, str(script_path),
            "--model", args.model,
            "--tokens", str(args.tokens),
            "--prefill-tokens", str(args.prefill_tokens),
        ]
        print(f"\n=== budget={budget} ===", flush=True)
        rc = subprocess.run(
            cmd, env=env, capture_output=True, text=True, timeout=600,
        )
        print("stdout:", rc.stdout[-1500:])
        if rc.returncode != 0:
            print("stderr:", rc.stderr[-1500:])
            continue
        # Read the profile after the driver exited. The harness flushes
        # pending dispatch events when slots are reused; allow a short
        # grace window for delayed writes.
        time.sleep(2.0)
        try:
            vals = ms_per_submit(args.profile + f".{budget}", grace_seconds=10.0)
        except Exception as e:
            print(f"profile read failed: {e}")
            continue
        results[budget] = report(f"budget={budget}", vals)
        # Decode tps from stdout
        for line in rc.stdout.splitlines():
            if line.startswith("decode_tps="):
                print(f"  {line.strip()}")
    print("\n=== summary ===")
    for k, vals in results.items():
        if vals:
            print(
                f"budget={k}: n={len(vals)} p50={percentile(vals,50):.2f} "
                f"p95={percentile(vals,95):.2f} p99={percentile(vals,99):.2f} "
                f"max={max(vals):.2f} ms"
            )


if __name__ == "__main__":
    main()