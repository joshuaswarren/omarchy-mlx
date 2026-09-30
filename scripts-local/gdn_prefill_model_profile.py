#!/usr/bin/env python3
"""Profile one 512-token prefill of Qwen3.5-9B with the omarchy GPU profiler.

Run inside a venv whose mlx-omarchy wheel was built with
MLX_OMARCHY_GPU_PROFILING=ON; MLX_OMARCHY_GPU_PROFILE receives the NDJSON
path. Prints the provenance line beside the measurement (AGENTS.md rule).

Usage:
  MLX_OMARCHY_GPU_PROFILE=/tmp/qwen-prefill.jsonl \
    python3 gdn_prefill_model_profile.py --label diag.b791539 \
    [--tokens 512] [--reps 3]
"""
import argparse
import os
import statistics
import subprocess
import sys
import time

os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

MODEL = ("~/.cache/huggingface/hub/"
         "models--mlx-community--Qwen3.5-9B-MLX-4bit/snapshots/"
         "938d8919941c6e7efd3c7150eff7fe9d12afa631")


def provenance_line() -> str:
    import mlx.core as mx
    ver = getattr(mx, "__version__", "?")
    lib = subprocess.run(
        [sys.executable, "-c",
         "import mlx.core as mx, pathlib;"
         "print(pathlib.Path(mx._default_device.__module__).parent)"],
        capture_output=True, text=True)
    _ = lib
    return f"mlx={ver} label={os.environ.get('GDN_LABEL', 'unset')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="unset")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    os.environ["GDN_LABEL"] = args.label

    from mlx_lm.utils import load
    import mlx.core as mx

    t0 = time.perf_counter()
    model, tok = load(MODEL)
    mx.eval(model.parameters())
    load_s = time.perf_counter() - t0

    seed = ("The history of computing begins with early mechanical "
            "calculators, telescopes and clocks. Machines relied on "
            "vacuum tubes, transistors and integrated circuits. ")
    ids = tok.encode(seed)
    while len(ids) < args.tokens:
        ids = ids + ids
    ids = ids[:args.tokens]
    n = len(ids)

    print(f"[provenance] {provenance_line()}")
    print(f"[load] {load_s:.2f}s; prompt tokens = {n}")

    # Warmup (short prefill compiles lazy kernels, settles caches).
    mx.eval(model(mx.array(ids[:32])[None]))
    mx.synchronize()

    times = []
    for r in range(args.reps):
        t0 = time.perf_counter()
        out = model(mx.array(ids)[None])
        mx.eval(out)
        mx.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"[prefill {n}] rep {r + 1}/{args.reps}: {dt:.3f}s "
              f"({n / dt:.1f} tok/s)")
    print(f"[prefill {n}] median {statistics.median(times):.3f}s "
          f"({n / statistics.median(times):.1f} tok/s); reps={times}")


if __name__ == "__main__":
    main()
