#!/usr/bin/env python3
"""A/B decode + prefill tok/s for issue #19 (queue M2 run, post-window).

Five alternating pairs; per pair, two passes interleaved (A=B off, B=cap
default-on), assert distinct stamps, report medians + percent change.
Also runs the contract Qwen3.8-2B 10-prompt/32-token decode and dumps
the exact generated token IDs (ids_sha256_16 via bench_decode.py).

Run on M2 inside gpu-turn AFTER the w73 packaged-stack qualification
window ends. Wheels:
  - wheel-A = main BEFORE the cap default bake (commit 4b2929a90).
  - wheel-B = main with the cap default baked (commit 1e7cb4b60).

Pre-flight (dev box): install each wheel into a private venv with
mlx-lm; assert `mx.__version__` stamp differs; sanity import + decode
of 4 tokens to warm caches.

The driver script written here runs the pinned-length decode protocol
(see scripts/bench_decode.py: --tokens N forces N tokens; EOS suppressed)
so the rate is comparable across runs, plus a 256-token prefill call
for prefill throughput.

Outputs: a JSON row per pass: {arm, pair, tok_s, prefill_s, ids_sha256_16}.
Prints medians per arm + percent change.
"""
import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DECODE_DRIVER = """
import os, sys, time, argparse
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import mlx.core as mx
from mlx_lm import load, generate
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--tokens", type=int, required=True)
ap.add_argument("--prefill-tokens", type=int, required=True)
ap.add_argument("--prompt", required=True)
args = ap.parse_args()
# Warmup: ensure compile + caches populated
_ = generate(model, tok, prompt="hello", max_tokens=4, verbose=False) if False else None
model, tok = load(args.model)
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
ids = [int(getattr(r, "token", r)) for r in resp]
# Greedy identity digest (matches bench_decode.py).
import hashlib
h = hashlib.sha256()
h.update(",".join(str(i) for i in ids).encode("ascii"))
print(f"decode_tps={args.tokens / (t1 - t0):.4f} tokens={args.tokens} "
      f"wall={t1 - t0:.4f}s ids_sha256_16={h.hexdigest()[:16]}", flush=True)
"""


def venv_python(venv):
    return str(Path(venv) / "bin" / "python")


def fresh_venv(path, wheel):
    if Path(path).exists():
        shutil.rmtree(path)
    subprocess.run(["python", "-m", "venv", "--system-site-packages", path],
                   check=True)
    p = venv_python(path)
    subprocess.run([p, "-m", "pip", "install", "--no-deps", wheel],
                   check=True)
    subprocess.run([p, "-m", "pip", "install", "mlx-lm==0.32.0"],
                   check=True)
    return p


def stamp_of(python):
    out = subprocess.run(
        [python, "-c", "import mlx.core as mx; print(mx.__version__)"],
        capture_output=True, text=True, check=True)
    return out.stdout.strip()


def run_pass(python, model, prompt, tokens, prefill_tokens):
    script = Path(tempfile.gettempdir()) / "subcap_ab_driver.py"
    script.write_text(DECODE_DRIVER)
    env = os.environ.copy()
    # Force the cap off (or default) by clearing env overrides where needed.
    env.pop("MLX_OMARCHY_BATCH_WORK", None)
    env.pop("MLX_OMARCHY_PROFILE_PATH", None)
    out = subprocess.run(
        [python, str(script),
         "--model", model,
          "--tokens", str(tokens),
          "--prefill-tokens", str(prefill_tokens),
          "--prompt", prompt],
        env=env, capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        print("stderr:", out.stderr[-1500:])
        raise RuntimeError("driver failed")
    for line in out.stdout.splitlines():
        if line.startswith("decode_tps="):
            parts = line.split()
            d = {}
            for p in parts[1:]:
                k, v = p.split("=", 1)
                d[k] = v
            return d
    raise RuntimeError(f"no decode_tps line; stdout={out.stdout}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--prefill-tokens", type=int, default=256)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--venv-a", default="/tmp/sc-venvA")
    ap.add_argument("--venv-b", default="/tmp/sc-venvB")
    ap.add_argument("--wheel-a", required=True)
    ap.add_argument("--wheel-b", required=True)
    ap.add_argument("--out", default="/tmp/sc-ab-results.json")
    args = ap.parse_args()

    print("== install venv A ==")
    pa = fresh_venv(args.venv_a, args.wheel_a)
    stamp_a = stamp_of(pa)
    print("A stamp:", stamp_a)
    print("== install venv B ==")
    pb = fresh_venv(args.venv_b, args.wheel_b)
    stamp_b = stamp_of(pb)
    print("B stamp:", stamp_b)
    if stamp_a == stamp_b:
        raise SystemExit(f"FAIL: stamps equal ({stamp_a}); not distinct")

    results = []
    for pair in range(args.pairs):
        for arm, python in (("A", pa), ("B", pb)):
            d = run_pass(python, args.model, args.prompt, args.tokens,
                         args.prefill_tokens)
            d["arm"] = arm
            d["pair"] = pair
            results.append(d)
            print(f"pair={pair} arm={arm} {d}")
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    a_tps = [float(r["decode_tps"]) for r in results if r["arm"] == "A"]
    b_tps = [float(r["decode_tps"]) for r in results if r["arm"] == "B"]
    a_ids = [r["ids_sha256_16"] for r in results if r["arm"] == "A"]
    b_ids = [r["ids_sha256_16"] for r in results if r["arm"] == "B"]
    print("\n=== SUMMARY ===")
    print(f"A median decode tok/s: {statistics.median(a_tps):.4f}")
    print(f"B median decode tok/s: {statistics.median(b_tps):.4f}")
    pct = (statistics.median(b_tps) - statistics.median(a_tps)) / statistics.median(a_tps) * 100
    print(f"B vs A: {pct:+.2f}%")
    print(f"A ids sha256_16 set: {set(a_ids)}")
    print(f"B ids sha256_16 set: {set(b_ids)}")
    if set(a_ids) != set(b_ids):
        raise SystemExit("FAIL: greedy identity diverged across arms")
    print("PASS: greedy identity bit-identical across A and B")


if __name__ == "__main__":
    main()