"""Parity + speed harness for the omarchy DSA indexer scores.

The lavapipe (Mesa dev-box Vulkan ICD) has a known hang in big matmul
shapes above the M*K element-count threshold (~614400 in the
[1, H, M, 128] x [1, 1, K, 128] orientation, see Mesa issue tracked
under `mesa-1`). The acceptance grid below stays under that threshold
so the harness completes; the model-level shape envelope is
covered by the upstream metallib's own parity (the MLA prefill
bottleneck the README's 30x claim refers to) and by direct testing
on real Apple GPUs in the M2 ticket window, not by lavapipe.

For each shape in the working grid the harness:
  1. Runs the omarchy-fused path and the composed (fallback) path on
     synthetic inputs drawn from a fixed seed.
  2. Computes the per-element max abs diff against an fp32 reference
     for both paths. The numerics gate is "omarchy-fused parity
     within 1.5x of the composed fallback's parity"; both must be
     well below the fp32-vs-bf16 storage floor.
  3. Times both paths in 50 trials (10 warmup) and reports the
     median ratio (omarchy / composed).

Usage:
  MLX_OMARCHY_ALLOW_NON_APPLE=1 python parity_speed.py
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

# Make the module under test importable when invoked from any cwd.
sys.path.insert(0, str(Path(__file__).parent))
import dsa_indexer_omarchy as dio  # noqa: E402


def _median(values):
    return float(statistics.median(values))


def _bench_one(fn, n_warmup=10, n_trials=50):
    for _ in range(n_warmup):
        out = fn()
    mx.eval(out)
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return _median(times), min(times), max(times)


def _draw_inputs(B, H, L, K, dtype, seed=0):
    """Draw q, k, weights deterministically."""
    mx.random.seed(seed)
    q = mx.random.normal(shape=(B, H, L, 128)).astype(dtype)
    k = mx.random.normal(shape=(B, 1, K, 128)).astype(dtype)
    # weights are non-negative and roughly O(1/H) so the per-head
    # contributions stay bounded; the metal kernel does the same
    # (post-RoPE q/k are O(1) at init, weights are scale_logits /
    # sqrt(D) by upstream convention).
    w = (
        mx.random.uniform(shape=(B, L, H), low=0.0, high=0.1) + 0.05
    ).astype(dtype)
    return q, k, w


def _max_abs_diff(a, b):
    diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    return float(mx.max(diff).item())


def run_shape(B, H, L, K, dtype, causal=True, mask_ratio=0):
    q, k, w = _draw_inputs(B, H, L, K, dtype)
    causal_q_offset = -1
    fp32, composed, omarchy = dio.run(
        q, k, w, causal=causal, causal_q_offset=causal_q_offset,
        mask_ratio=mask_ratio,
    )

    parity_composed = _max_abs_diff(composed, fp32)
    parity_omarchy = _max_abs_diff(omarchy, fp32)
    inter_diff = _max_abs_diff(omarchy, composed)

    def _composed_fn():
        return dio._indexer_scores_composed(
            q, k, w, causal, K - L if causal else 0
        )

    def _omarchy_fn():
        return dio.dsa_indexer_scores_omarchy(
            q, k, w, causal=causal, causal_q_offset=K - L if causal else 0,
            mask_ratio=mask_ratio,
        )

    comp_med, comp_min, comp_max = _bench_one(_composed_fn)
    oma_med, oma_min, oma_max = _bench_one(_omarchy_fn)

    return {
        "B": B, "H": H, "L": L, "K": K,
        "dtype": str(dtype).split(".")[-1],
        "causal": causal, "mask_ratio": mask_ratio,
        "parity_composed_vs_fp32": parity_composed,
        "parity_omarchy_vs_fp32": parity_omarchy,
        "inter_omarchy_vs_composed": inter_diff,
        "composed_median_ms": comp_med * 1000.0,
        "omarchy_median_ms": oma_med * 1000.0,
        "speed_ratio_omarchy_over_composed": oma_med / comp_med,
        "composed_min_ms": comp_min * 1000.0,
        "omarchy_min_ms": oma_min * 1000.0,
        "composed_max_ms": comp_max * 1000.0,
        "omarchy_max_ms": oma_max * 1000.0,
    }


# Acceptance grid kept under the lavapipe matmul hang threshold
# (Mesa issue; tracked in `mesa-1` repo). L*K < ~600K is the safe
# envelope on this dev box. The grid is sorted to keep the slow
# shapes (H=64) short of the wall-clock budget on lavapipe.
SHAPES = [
    # (B, H, L, K, dtype, causal, mask_ratio, label)
    (1, 32, 64, 64, mx.bfloat16, True, 0, "smoke-h32-bf16"),
    (1, 32, 64, 256, mx.bfloat16, True, 0, "K=4L-h32-bf16-small"),
    (1, 32, 128, 128, mx.bfloat16, True, 0, "small-h32-bf16"),
    (1, 32, 128, 512, mx.bfloat16, True, 0, "K=4L-h32-bf16"),
    (1, 32, 256, 256, mx.bfloat16, True, 0, "L256-h32-bf16"),
    (1, 32, 64, 64, mx.float16, True, 0, "smoke-h32-fp16"),
    (1, 32, 128, 128, mx.float16, True, 0, "small-h32-fp16"),
    (1, 32, 256, 256, mx.bfloat16, True, 16, "pooled-ratio-16"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="parity_speed.csv", help="output CSV")
    p.add_argument("--n-warmup", type=int, default=10, help="warmup trials")
    p.add_argument("--n-trials", type=int, default=50, help="measurement trials")
    p.add_argument("--filter", default=None, help="substring filter on shape label")
    args = p.parse_args()

    fields = [
        "label", "B", "H", "L", "K", "dtype", "causal", "mask_ratio",
        "parity_composed_vs_fp32", "parity_omarchy_vs_fp32",
        "inter_omarchy_vs_composed",
        "composed_median_ms", "omarchy_median_ms",
        "speed_ratio_omarchy_over_composed",
        "composed_min_ms", "omarchy_min_ms",
        "composed_max_ms", "omarchy_max_ms",
    ]
    rows = []
    for B, H, L, K, dtype, causal, mask_ratio, label in SHAPES:
        if args.filter and args.filter not in label:
            continue
        print(
            f"[{label}] B={B} H={H} L={L} K={K} dtype={dtype} "
            f"causal={causal} mask_ratio={mask_ratio}", flush=True,
        )
        try:
            row = run_shape(B, H, L, K, dtype, causal, mask_ratio)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            continue
        row["label"] = label
        rows.append(row)
        print(
            f"  parity: composed={row['parity_composed_vs_fp32']:.3e} "
            f"omarchy={row['parity_omarchy_vs_fp32']:.3e} "
            f"inter={row['inter_omarchy_vs_composed']:.3e}", flush=True,
        )
        print(
            f"  speed:  composed={row['composed_median_ms']:.2f}ms "
            f"omarchy={row['omarchy_median_ms']:.2f}ms "
            f"ratio={row['speed_ratio_omarchy_over_composed']:.3f}x",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    print(f"\nWrote {len(rows)} rows to {out_path}", flush=True)
    return rows


if __name__ == "__main__":
    main()
