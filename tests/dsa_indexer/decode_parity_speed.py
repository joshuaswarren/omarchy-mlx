"""Parity + speed for the omarchy decode indexer scan.

Grid: S in {1024, 4096, 16384, 65536}, bf16 and fp16, both output
modes. fp32-ref ground truth. The numerics gate: the fp32-chain port
must be no worse than the deployed fallback against fp32-ref, per op.
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
import decode_scores_omarchy as dso  # noqa: E402


def _draw(B, S, dtype, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal(shape=(B, 32, 1, 128)).astype(dtype)
    k = mx.random.normal(shape=(B, 1, S, 128)).astype(dtype)
    w = (mx.random.uniform(shape=(B, 32), low=0.0, high=0.1) + 0.05).astype(
        dtype
    )
    return q, k, w


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
    return float(statistics.median(times))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="decode_parity_speed.csv")
    args = p.parse_args()

    fields = [
        "label", "B", "S", "dtype", "fp32_scores",
        "ref_vs_ref_f32chain", "ref_vs_ref_composed",
        "f32chain_vs_composed",
        "composed_ms", "f32chain_ms",
        "ratio_f32chain_over_composed",
    ]
    rows = []
    for B, S, dtype, fp32_scores, label in [
        (1, 1024, mx.bfloat16, False, "S1024-bf16"),
        (1, 4096, mx.bfloat16, False, "S4096-bf16"),
        (1, 16384, mx.bfloat16, False, "S16384-bf16"),
        (1, 65536, mx.bfloat16, False, "S65536-bf16"),
        (1, 4096, mx.bfloat16, True, "S4096-bf16-of32"),
        (1, 4096, mx.float16, False, "S4096-fp16"),
        (2, 4096, mx.bfloat16, False, "S4096-bf16-B2"),
    ]:
        print(f"[{label}] B={B} S={S} dtype={dtype} fp32_scores={fp32_scores}",
              flush=True)
        q, k, w = _draw(B, S, dtype)
        ref = dso.dsa_decode_scores_reference(q, k, w, fp32_scores)
        chain = dso.dsa_decode_scores_omarchy(q, k, w, fp32_scores=fp32_scores)
        comp = dso.dsa_decode_scores_composed(q, k, w, fp32_scores=fp32_scores)
        mx.eval(ref, chain, comp)
        row = {
            "label": label, "B": B, "S": S,
            "dtype": str(dtype).split(".")[-1],
            "fp32_scores": fp32_scores,
            "ref_vs_ref_f32chain": _max_abs(chain, ref),
            "ref_vs_ref_composed": _max_abs(comp, ref),
            "f32chain_vs_composed": _max_abs(chain, comp),
            "composed_ms": _bench(
                lambda: dso.dsa_decode_scores_composed(
                    q, k, w, fp32_scores=fp32_scores)
            ) * 1000.0,
            "f32chain_ms": _bench(
                lambda: dso.dsa_decode_scores_omarchy(
                    q, k, w, fp32_scores=fp32_scores)
            ) * 1000.0,
        }
        row["ratio_f32chain_over_composed"] = (
            row["f32chain_ms"] / row["composed_ms"]
        )
        rows.append(row)
        print(
            f"  parity vs fp32-ref: f32chain={row['ref_vs_ref_f32chain']:.3e} "
            f"composed={row['ref_vs_ref_composed']:.3e} "
            f"inter={row['f32chain_vs_composed']:.3e}",
            flush=True,
        )
        print(
            f"  speed: composed={row['composed_ms']:.1f}ms "
            f"f32chain={row['f32chain_ms']:.1f}ms "
            f"ratio={row['ratio_f32chain_over_composed']:.3f}x",
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
