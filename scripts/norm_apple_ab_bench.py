#!/usr/bin/env python3
"""Same-wheel ABBA for the FastRmsNormAppleBF16 packed-IO vectorization
(M2D-1). Pairs run in A-B-B-A order, one subprocess per leg:

  arm A = packed kernel (default),
  arm B = scalar kernel (MLX_OMARCHY_NORM_APPLE_PACKED=0),
  arm C = packed at local_size 128 (MLX_OMARCHY_NORM_APPLE_WG=128), the
          workgroup-size screen, run as a second pair set when --wg-screen.

Per leg: wheel stamp, prompt token count, prefill seconds, pinned greedy
decode tok/s, sha256-16 + first/last ids. One leg per arm runs with
MLX_OMARCHY_TRACE_DISPATCH=1 and its stderr census is keyed by kernel id.
A micro-bench leg isolates rms_norm at [1,1,D] bf16 (D=2560 for 4B) and
reports kernel milliseconds per call.

Done rules: identical greedy ids across arms, and A tok/s matched or
better than B within the run-to-run band. ID divergence fails the capture.

Usage (inside the GPU lock):
  python3 scripts/norm_apple_ab_bench.py \
      --model mlx-community/Qwen3-4B-Instruct-2507-4bit --norm-width 2560 \
      --pairs 5 --prefill-tokens 512 --decode-tokens 64 \
      --out /tmp/normapple-abba-4b
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

DRIVER = r"""
import argparse, hashlib, json, os, time
import mlx.core as mx
from mlx_lm import load, stream_generate

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--prefill-tokens", type=int, required=True)
ap.add_argument("--decode-tokens", type=int, required=True)
ap.add_argument("--norm-width", type=int, required=True)
ap.add_argument("--micro-iters", type=int, default=0)
args = ap.parse_args()

model, tok = load(args.model)
prompt_ids = tok.encode(open(args.prompt_file).read())[:args.prefill_tokens]
prompt = mx.array(prompt_ids)

def digest(ids):
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()[:16]

t0 = time.monotonic_ns()
first = None
for r in stream_generate(model, tok, prompt, max_tokens=1):
    first = r
    break
prefill_s = (time.monotonic_ns() - t0) / 1e9
prompt_tokens = getattr(first, "prompt_tokens", None)

ids = []
t0 = None
for r in stream_generate(model, tok, prompt, max_tokens=args.decode_tokens):
    if t0 is None:
        t0 = time.monotonic_ns()
    ids.append(r.token)
times_done = time.monotonic_ns() - t0 if t0 is not None else 0
decode_tps = (len(ids) - 1) / (times_done / 1e9) if len(ids) > 1 else None

# Micro-bench: isolate the rms_norm kernel at the decode shape. One
# warmup pass builds the pipeline; the timed loop uses eval() per call.
norm_ms = None
if args.micro_iters > 0:
    import mlx.core.fast as fast
    x = mx.random.normal((1, 1, args.norm_width)).astype(mx.bfloat16)
    w = mx.random.normal((args.norm_width,)).astype(mx.bfloat16)
    y = fast.rms_norm(x, w, 1e-6)
    mx.eval(y)
    t0 = time.monotonic_ns()
    for _ in range(args.micro_iters):
        y = fast.rms_norm(x, w, 1e-6)
    mx.eval(y)
    norm_ms = (time.monotonic_ns() - t0) / 1e6 / args.micro_iters

print(json.dumps({
    "wheel": mx.__version__,
    "packed_env": os.environ.get("MLX_OMARCHY_NORM_APPLE_PACKED"),
    "wg_env": os.environ.get("MLX_OMARCHY_NORM_APPLE_WG"),
    "prompt_tokens": prompt_tokens,
    "prefill_s": round(prefill_s, 6),
    "generated": len(ids),
    "decode_tps": round(decode_tps, 4) if decode_tps else None,
    "norm_ms": round(norm_ms, 4) if norm_ms else None,
    "ids_sha256_16": digest(ids),
    "ids_first": ids[:3],
    "ids_last": ids[-3:],
}, sort_keys=True))
"""


def kernel_ids(*names):
    """Numeric ComputeKernel ids resolved from this repo's compute.h."""
    header = (Path(__file__).resolve().parent.parent /
              "overlay/mlx/backend/omarchy/compute.h")
    text = header.read_text(encoding="utf-8")
    body = text.split("enum class ComputeKernel", 1)[1].split("};", 1)[0]
    kid = -1
    out = {}
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        name, eq, num = line.partition("=")
        name = name.strip().rstrip(",")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        kid = int(num.strip().rstrip(",")) if eq else kid + 1
        if name in names:
            out[name] = kid
    missing = set(names) - set(out)
    if missing:
        raise SystemExit(f"missing kernel ids {missing} in {header}")
    return out


def run_leg(driver_path, env_extra, args, trace):
    env = dict(os.environ)
    env.pop("MLX_OMARCHY_NORM_APPLE_PACKED", None)
    env.pop("MLX_OMARCHY_NORM_APPLE_WG", None)
    env.update(env_extra)
    if trace:
        env["MLX_OMARCHY_TRACE_DISPATCH"] = "1"
    cmd = [
        sys.executable, str(driver_path),
        "--model", args.model,
        "--prompt-file", args.prompt_file,
        "--prefill-tokens", str(args.prefill_tokens),
        "--decode-tokens", str(args.decode_tokens),
        "--norm-width", str(args.norm_width),
        "--micro-iters", str(args.micro_iters),
    ]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       timeout=1800)
    if p.returncode != 0:
        raise SystemExit(
            f"leg failed rc={p.returncode}\nstderr tail:\n{p.stderr[-3000:]}")
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")][-1]
    row = json.loads(line)
    if trace:
        events = {}
        for m in re.finditer(
                r"\[rtmod\] DISPATCH kernel=(\d+) count=\d+", p.stderr):
            k = int(m.group(1))
            events[k] = events.get(k, 0) + 1
        row["dispatch_events"] = events
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--norm-width", type=int, required=True)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--prefill-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=64)
    ap.add_argument("--micro-iters", type=int, default=400)
    ap.add_argument("--wg-screen", action="store_true")
    ap.add_argument("--prompt-file", default="/tmp/abba-prompt.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.prompt_file).exists():
        Path(args.prompt_file).write_text(
            "Explain, in a numbered list, how a Vulkan compute pipeline "
            "dispatches workgroups for a row-reduction kernel, and why "
            "packed vector loads change the memory transaction count but "
            "not the summation order. " * 12)

    driver_path = out_dir / "driver.py"
    driver_path.write_text(DRIVER)

    kids = kernel_ids("FastRmsNormAppleBF16",
                      "FastRmsNormAppleScalarBF16",
                      "FastRmsNormAppleWg128BF16")

    # ABBA pairs: legs ordered B A B A per pair so each arm wraps the
    # other against thermal drift.
    arms = {"A": {}, "B": {"MLX_OMARCHY_NORM_APPLE_PACKED": "0"}}
    if args.wg_screen:
        arms["C"] = {"MLX_OMARCHY_NORM_APPLE_WG": "128"}
    results = []
    seq = 0
    for pair in range(args.pairs):
        for arm in (("B", "A", "A", "B") if not args.wg_screen
                    else ("B", "A", "C", "A", "C", "A", "B", "B")):
            seq += 1
            trace = seq <= 4
            row = run_leg(driver_path, arms[arm], args, trace)
            row.update({"arm": arm, "pair": pair, "seq": seq})
            results.append(row)
            print(f"leg {seq} arm {arm}: "
                  f"tps={row['decode_tps']} norm_ms={row['norm_ms']} "
                  f"ids={row['ids_sha256_16']}", flush=True)

    out = {
        "model": args.model,
        "norm_width": args.norm_width,
        "kernel_ids": kids,
        "wheels": sorted({r["wheel"] for r in results}),
        "legs": results,
    }
    shas = {r["ids_sha256_16"] for r in results}
    out["ids_identical"] = len(shas) == 1
    out["ids_sha256_16"] = sorted(shas)
    for arm in arms:
        tps = [r["decode_tps"] for r in results if r["arm"] == arm]
        ms = [r["norm_ms"] for r in results if r["arm"] == arm and r["norm_ms"]]
        out[f"{arm}_tps_median"] = statistics.median(tps)
        out[f"{arm}_tps_legs"] = tps
        out[f"{arm}_norm_ms_median"] = (
            statistics.median(ms) if ms else None)
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "legs"},
                     indent=2))
    if not out["ids_identical"]:
        sys.exit("FAIL: greedy ids diverged across arms")


if __name__ == "__main__":
    main()
