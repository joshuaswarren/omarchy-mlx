#!/usr/bin/env python3
"""Quick load smoke test for a single model.

Loads a model, runs one short generation, records load time, peak memory,
and whether it served successfully. Used to determine if a model can be
loaded by mlx_lm 0.31.3 + project patches on this Vulkan backend.

Usage: smoke_load.py --model-id <repo> --revision <sha> --out <path.json>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MLX_DISABLE_COMPILE", "1")

HF_CACHE_ROOT = Path(os.environ.get(
    "HF_HUB_CACHE",
    str(Path.home() / ".cache" / "huggingface" / "hub"),
))


def meminfo_available_mib() -> int:
    txt = Path("/proc/meminfo").read_text()
    m = re.search(r"MemAvailable:\s+(\d+)\s+kB", txt)
    return int(m.group(1)) // 1024 if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    snapshot = (
        HF_CACHE_ROOT
        / f"models--{args.model_id.replace('/', '--')}"
        / "snapshots"
        / args.revision
    )
    out = {
        "model_id": args.model_id,
        "revision": args.revision,
        "snapshot": str(snapshot),
        "mem_avail_pre_mib": meminfo_available_mib(),
    }
    if not snapshot.exists():
        out["error"] = "snapshot_missing"
        out["status"] = "skipped_no_snapshot"
        Path(args.out).write_text(json.dumps(out, indent=2))
        return 0

    t0 = time.monotonic()
    try:
        from mlx_lm import load
        model, tok = load(str(snapshot))
        out["load_s"] = time.monotonic() - t0
    except Exception as exc:
        out["error"] = f"load: {type(exc).__name__}: {str(exc)[:400]}"
        out["traceback"] = traceback.format_exc()[:2000]
        out["status"] = "load_failed"
        import mlx.core as mx
        out["peak_mem_gb"] = mx.get_peak_memory() / 1e9 if mx else 0
        out["mem_avail_post_mib"] = meminfo_available_mib()
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))
        return 0

    import mlx.core as mx
    out["peak_mem_gb_after_load"] = mx.get_peak_memory() / 1e9
    out["mem_avail_post_mib"] = meminfo_available_mib()

    try:
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temp=0.0, top_p=1.0)
        prompt = tok.encode("The capital of France is")
        t1 = time.monotonic()
        tokens = []
        for token, _ in generate_step(mx.array(prompt), model, max_tokens=args.max_tokens, sampler=sampler):
            tokens.append(int(token))
        out["gen_s"] = time.monotonic() - t1
        out["gen_tokens"] = len(tokens)
        out["gen_tps"] = len(tokens) / out["gen_s"] if out["gen_s"] > 0 else 0
        out["gen_text"] = tok.decode(tokens)
        out["status"] = "ok"
    except Exception as exc:
        out["error"] = f"gen: {type(exc).__name__}: {str(exc)[:400]}"
        out["traceback"] = traceback.format_exc()[:2000]
        out["status"] = "gen_failed"

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())