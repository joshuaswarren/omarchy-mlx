"""Large-shape parity for the DSA indexer on real GPUs.

These are the shapes that hang the dev box's lavapipe (the Mesa
big-matmul defect); on the M2's real Vulkan driver they must complete.
Parity check: the omarchy-fused path vs the composed fallback, both
on the GPU. At these magnitudes the two paths are bit-identical (same
primitive set, same order), so the expected diff is exactly 0; any
nonzero diff is reported per shape. No fp32 reference here (a CPU
reference at K=16384 is out of ticket budget); the small-shape grid
already pins both paths to the fp32 ground truth.

This is correctness only - the timing columns are informational.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
import dsa_indexer_omarchy as dio  # noqa: E402

# (B, H, L, K, dtype, label) - the lavapipe-hang envelope plus the
# largest prefill shape the metallib serves (K=16L at L=1024).
SHAPES = [
    (1, 32, 1024, 1024, mx.bfloat16, "L1024-K1024-h32"),
    (1, 64, 1024, 1024, mx.bfloat16, "L1024-K1024-h64"),
    (1, 32, 1024, 4096, mx.bfloat16, "L1024-K4096-h32"),
    (1, 64, 1024, 4096, mx.bfloat16, "L1024-K4096-h64"),
    (1, 32, 512, 4096, mx.bfloat16, "L512-K4096-h32"),
    (1, 32, 1024, 16384, mx.bfloat16, "L1024-K16384-h32"),
]


def _max_abs(a, b):
    return float(
        mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="large_parity.csv")
    args = p.parse_args()

    rows = []
    for B, H, L, K, dtype, label in SHAPES:
        print(f"[{label}] B={B} H={H} L={L} K={K}", flush=True)
        mx.random.seed(0)
        q = mx.random.normal(shape=(B, H, L, 128)).astype(dtype)
        k = mx.random.normal(shape=(B, 1, K, 128)).astype(dtype)
        w = (mx.random.uniform(shape=(B, L, H), low=0.0, high=0.1) + 0.05).astype(
            dtype
        )
        offset = K - L
        try:
            t0 = time.perf_counter()
            fused = dio.dsa_indexer_scores_omarchy(
                q, k, w, causal=True, causal_q_offset=offset
            )
            mx.eval(fused)
            t1 = time.perf_counter()
            composed = dio._indexer_scores_composed(q, k, w, True, offset)
            mx.eval(composed)
            t2 = time.perf_counter()
            finite = bool(mx.all(mx.isfinite(fused.astype(mx.float32))).item())
            row = {
                "label": label,
                "H": H, "L": L, "K": K,
                "fused_vs_composed": _max_abs(fused, composed),
                "all_finite": finite,
                "fused_ms": (t1 - t0) * 1000.0,
                "composed_ms": (t2 - t1) * 1000.0,
            }
        except Exception as e:
            row = {
                "label": label, "H": H, "L": L, "K": K,
                "fused_vs_composed": None, "all_finite": False,
                "fused_ms": None, "composed_ms": None, "error": str(e),
            }
            print(f"  ERROR: {e}", flush=True)
        rows.append(row)
        if row.get("error") is None:
            print(
                f"  diff={row['fused_vs_composed']:.3e} "
                f"finite={row['all_finite']} "
                f"fused={row['fused_ms']:.0f}ms "
                f"composed={row['composed_ms']:.0f}ms",
                flush=True,
            )

    fields = ["label", "H", "L", "K", "fused_vs_composed", "all_finite",
              "fused_ms", "composed_ms", "error"]
    with open(args.out, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=fields)
        cw.writeheader()
        for row in rows:
            cw.writerow({k: row.get(k) for k in fields})
    print(f"\nWrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
