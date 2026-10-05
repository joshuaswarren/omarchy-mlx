"""Acceptance harness for the NATIVE omarchy DSA indexer
(mx.fast.dsa_indexer_scores, shaders/dsa_indexer.comp).

Gate (receipts/2026-10-04-omlx-family-glm-dsa/README.md): the native
kernel accumulates dots in float32 with a single 16-bit round at the
store, so its error vs the fp32 reference must be <= the composed
fallback's error at every shape (strictly tighter, never worse), and
its wall time vs the composed fallback is the A25a speed number.

The fp32 reference is a CPU-stream batched matmul chain (fp32
storage, different reduction path than both GPU routes).
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))


def _draw(B, H, L, K, dtype, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal(shape=(B, H, L, 128)).astype(dtype)
    k = mx.random.normal(shape=(B, 1, K, 128)).astype(dtype)
    w = (mx.random.uniform(shape=(B, L, H), low=0.0, high=0.1) + 0.05).astype(
        dtype
    )
    return q, k, w


def _fp32_ref(q, k, w, causal_q_offset, mask_ratio, causal=True):
    def _ref():
        q32 = q.astype(mx.float32)
        k32 = k.astype(mx.float32)
        w32 = w.astype(mx.float32)
        s = mx.matmul(q32, mx.transpose(k32, (0, 1, 3, 2)))
        B, H, L, K = s.shape
        l_idx = mx.arange(L)[None, None, :, None]
        c_idx = mx.arange(K)[None, None, None, :]
        valid = mx.ones((1, 1, L, K), mx.bool_)
        if causal:
            valid = c_idx <= (causal_q_offset + l_idx)
        if mask_ratio > 0:
            thr = (causal_q_offset + l_idx + 1) // mask_ratio
            valid = valid & (c_idx < thr)
        s = mx.where(valid, s, mx.array(-1e30, mx.float32))
        s = mx.maximum(s, 0.0)
        w_b = w32[:, None, :, :].transpose(0, 3, 2, 1)
        s = s * w_b
        return mx.sum(s, axis=1, keepdims=True)

    with mx.stream(mx.cpu):
        return _ref()


def _composed(q, k, w, causal_q_offset, mask_ratio, dtype, causal=True):
    s = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2)))
    B, H, L, K = s.shape
    l_idx = mx.arange(L)[None, None, :, None]
    c_idx = mx.arange(K)[None, None, None, :]
    valid = mx.ones((1, 1, L, K), mx.bool_)
    if causal:
        valid = c_idx <= (causal_q_offset + l_idx)
    if mask_ratio > 0:
        thr = (causal_q_offset + l_idx + 1) // mask_ratio
        valid = valid & (c_idx < thr)
    floor = float.fromhex("-0x1.fep+127")
    neg = mx.array(floor, mx.float32).astype(dtype)
    s = mx.where(valid, s, neg)
    s = mx.maximum(s, 0.0)
    w_b = w[:, None, :, :].transpose(0, 3, 2, 1)
    s = s * w_b
    return mx.sum(s, axis=1, keepdims=True)


def _max_abs(a, b):
    return float(
        mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item()
    )


def _bench(fn, warmup=3, trials=15):
    for _ in range(warmup):
        out = fn()
    mx.eval(out)
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(statistics.median(times)) * 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="native_parity.csv")
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--skip-ref", action="store_true",
                   help="skip the CPU fp32 reference (timing-only runs)")
    args = p.parse_args()

    fields = [
        "label", "H", "L", "K", "dtype", "mask_ratio",
        "native_vs_fp32", "composed_vs_fp32", "native_vs_composed",
        "native_ms", "composed_ms", "ratio_native_over_composed",
    ]
    rows = []
    grid = [
        (1, 32, 64, 64, mx.bfloat16, 0, "smoke-h32"),
        (1, 64, 64, 64, mx.bfloat16, 0, "smoke-h64"),
        (1, 32, 128, 512, mx.bfloat16, 0, "K4L-h32"),
        (1, 64, 256, 256, mx.bfloat16, 0, "L256-h64"),
        (1, 32, 256, 256, mx.bfloat16, 16, "pooled16-h32"),
        (1, 64, 256, 256, mx.bfloat16, 16, "pooled16-h64"),
        (1, 32, 512, 512, mx.bfloat16, 0, "L512-h32"),
        (1, 64, 512, 512, mx.bfloat16, 0, "L512-h64"),
        (1, 32, 1024, 1024, mx.bfloat16, 0, "L1024-h32"),
        (1, 64, 1024, 1024, mx.bfloat16, 0, "L1024-h64"),
        (1, 32, 1024, 4096, mx.bfloat16, 0, "L1024-K4096-h32"),
        (1, 64, 1024, 4096, mx.bfloat16, 0, "L1024-K4096-h64"),
        (1, 32, 512, 4096, mx.bfloat16, 0, "L512-K4096-h32"),
        (1, 32, 1024, 16384, mx.bfloat16, 0, "L1024-K16384-h32"),
        (1, 64, 64, 64, mx.float16, 0, "smoke-h64-fp16"),
        (2, 32, 256, 256, mx.bfloat16, 0, "B2-L256-h32"),
    ]
    for B, H, L, K, dtype, mask_ratio, label in grid:
        print(f"[{label}] B={B} H={H} L={L} K={K} mr={mask_ratio}", flush=True)
        q, k, w = _draw(B, H, L, K, dtype)
        off = K - L
        row = {
            "label": label, "H": H, "L": L, "K": K,
            "dtype": str(dtype).split(".")[-1], "mask_ratio": mask_ratio,
        }
        try:
            native = mx.fast.dsa_indexer_scores(
                q, k, w,
                causal=True, causal_q_offset=off, mask_ratio=mask_ratio,
            )
            mx.eval(native)
        except Exception as e:
            print(f"  NATIVE ERROR: {e}", flush=True)
            row["native_vs_fp32"] = None
            row["native_vs_composed"] = None
            rows.append(row)
            continue
        comp = _composed(q, k, w, off, mask_ratio, dtype)
        mx.eval(comp)
        row["native_vs_composed"] = _max_abs(native, comp)
        if not args.skip_ref:
            ref = _fp32_ref(q, k, w, off, mask_ratio)
            mx.eval(ref)
            row["native_vs_fp32"] = _max_abs(native, ref)
            row["composed_vs_fp32"] = _max_abs(comp, ref)
        row["native_ms"] = _bench(
            lambda: mx.fast.dsa_indexer_scores(
                q, k, w, causal=True, causal_q_offset=off,
                mask_ratio=mask_ratio),
            trials=args.trials,
        )
        row["composed_ms"] = _bench(
            lambda: _composed(q, k, w, off, mask_ratio, dtype),
            trials=args.trials,
        )
        row["ratio_native_over_composed"] = (
            row["native_ms"] / row["composed_ms"]
        )
        rows.append(row)
        print(
            f"  native_vs_fp32={row.get('native_vs_fp32')} "
            f"composed_vs_fp32={row.get('composed_vs_fp32')} "
            f"inter={row['native_vs_composed']:.3e}",
            flush=True,
        )
        print(
            f"  speed: native={row['native_ms']:.1f}ms "
            f"composed={row['composed_ms']:.1f}ms "
            f"ratio={row['ratio_native_over_composed']:.3f}x",
            flush=True,
        )

    with open(args.out, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=fields)
        cw.writeheader()
        for row in rows:
            cw.writerow({k: row.get(k) for k in fields})
    print(f"\nWrote {len(rows)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
