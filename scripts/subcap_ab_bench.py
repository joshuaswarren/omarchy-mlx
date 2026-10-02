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
import os, sys, time, argparse, hashlib, json
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--mode", choices=["decode", "prefill"], required=True)
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--prefill-tokens", type=int, default=256)
ap.add_argument("--prompt", required=True)
args = ap.parse_args()
model, tok = load(args.model)


def ids_digest(ids):
    h = hashlib.sha256()
    h.update(",".join(str(int(i)) for i in ids).encode("ascii"))
    return h.hexdigest()[:16]


if args.mode == "decode":
    # Warmup: compile paths before the timed pass.
    for _ in stream_generate(model, tok, prompt="hi", max_tokens=2):
        pass
    ids = []
    t0 = time.monotonic()
    for r in stream_generate(model, tok, prompt=args.prompt,
                             max_tokens=args.tokens):
        ids.append(int(r.token))
    t1 = time.monotonic()
    # Rate over inter-token gaps (n-1): the first gap includes the
    # prompt prefill in mlx-lm's generate loop, identically in both
    # arms, so the medians stay comparable.
    tps = (len(ids) - 1) / (t1 - t0)
    print(json.dumps({"mode": "decode", "tps": tps, "n": len(ids),
                      "ids_sha256_16": ids_digest(ids),
                      "mx_version": mx.__version__}))
else:
    words = args.prompt.split()
    rep = max(1, args.prefill_tokens // max(1, len(words)))
    long_prompt = " ".join(words * rep)
    n_tok = len(tok.encode(long_prompt))
    t0 = time.monotonic()
    for _ in stream_generate(model, tok, prompt=long_prompt, max_tokens=1):
        pass
    t1 = time.monotonic()
    print(json.dumps({"mode": "prefill", "prompt_tokens": n_tok,
                      "wall_s": t1 - t0, "tps": n_tok / (t1 - t0),
                      "mx_version": mx.__version__}))
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


def run_pass(python, model, prompt, tokens, prefill_tokens, mode):
    script = Path(tempfile.gettempdir()) / "subcap_ab_driver.py"
    script.write_text(DECODE_DRIVER)
    env = os.environ.copy()
    # Keep the arm's env clean: BATCH_WORK must come from the wheel's
    # baked default; QUEUE_PRIORITY from the arm's setting.
    env.pop("MLX_OMARCHY_BATCH_WORK", None)
    env.pop("MLX_OMARCHY_GPU_PROFILE", None)
    out = subprocess.run(
        [python, str(script),
         "--model", model,
          "--mode", mode,
          "--tokens", str(tokens),
          "--prefill-tokens", str(prefill_tokens),
          "--prompt", prompt],
        env=env, capture_output=True, text=True, timeout=900)
    if out.returncode != 0:
        print("stderr:", out.stderr[-1500:])
        raise RuntimeError(f"driver failed ({mode})")
    for line in out.stdout.splitlines():
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"no json line; stdout={out.stdout[-800:]}")


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
            for mode in ("decode", "prefill"):
                d = run_pass(python, args.model, args.prompt, args.tokens,
                             args.prefill_tokens, mode)
                d["arm"] = arm
                d["pair"] = pair
                results.append(d)
                print(f"pair={pair} arm={arm} {mode} {json.dumps(d)}",
                      flush=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    def medians(mode, key):
        vals = {arm: [float(r[key]) for r in results
                      if r["arm"] == arm and r["mode"] == mode]
                for arm in ("A", "B")}
        return {arm: statistics.median(v) for arm, v in vals.items() if v}

    dec = medians("decode", "tps")
    pre = medians("prefill", "tps")
    a_ids = {r["ids_sha256_16"] for r in results
             if r["arm"] == "A" and r["mode"] == "decode"}
    b_ids = {r["ids_sha256_16"] for r in results
             if r["arm"] == "B" and r["mode"] == "decode"}
    print("\n=== SUMMARY ===")
    print(f"decode  tok/s medians: A={dec.get('A'):.4f} B={dec.get('B'):.4f}"
          if dec.get("A") and dec.get("B") else f"decode medians: {dec}")
    if dec.get("A"):
        pct = (dec["B"] - dec["A"]) / dec["A"] * 100
        print(f"decode B vs A: {pct:+.2f}%")
    if pre.get("A") and pre.get("B"):
        ppct = (pre["B"] - pre["A"]) / pre["A"] * 100
        print(f"prefill tok/s medians: A={pre['A']:.1f} B={pre['B']:.1f} "
              f"({ppct:+.2f}%)")
    print(f"A ids set: {a_ids}")
    print(f"B ids set: {b_ids}")
    if not a_ids or not b_ids or a_ids != b_ids:
        raise SystemExit(f"FAIL: greedy identity diverged: A={a_ids} B={b_ids}")
    print("PASS: greedy identity bit-identical across A and B")


if __name__ == "__main__":
    main()