"""M1MD-1 per-shape QMV isolation (jw16, G13C M1 Max).

Times the decode multi-GEMV at the exact Qwen3-4B member shapes with the
small-N tile off and on, reporting effective weight-stream GB/s so the
R2S8W2 vs R2S4W2 vs R1S8W2 pick is measured per shape, not guessed.

Shapes (n_out, k): qkv trio shares x@2560 (4096+1024+1024), o_proj
(2560,2560), gate+up pair shares x@2560 (9728+9728), down_proj (2560,9728),
lm_head (151936,2560) as the saturated reference.

Run three times, once per mode:
  MLX_OMARCHY_QMM_SMALLN=0     python3 gemv_smalln.py
  MLX_OMARCHY_QMM_SMALLN=2x4w2 python3 gemv_smalln.py
  MLX_OMARCHY_QMM_SMALLN=1x8w2 python3 gemv_smalln.py

Each member group is evaluated against independent weight copies sharing
one x, which is how decode folds them into multi dispatches. GB/s counts
packed weights + scales + biases only (x and out are negligible).
"""
import os
import sys
import time

import mlx.core as mx

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8

SMALLN = os.environ.get("MLX_OMARCHY_QMM_SMALLN", "<unset>")


def wbytes(n, k):
    return n * k // 2 + 2 * 2 * n * (k // 64)  # 4-bit words + bf16 scale/bias


def make(n, k):
    q = mx.quantize(
        mx.random.normal((n, k)).astype(mx.bfloat16), group_size=64, bits=4)
    mx.eval(q)
    return q


def timed(fn, inner):
    fn()
    ts = []
    for _ in range(REPS):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2] / inner


def bench_group(label, x, members, copies):
    """members: list of (n,k). Copies of the member list share x -> multi."""
    ws = []
    for _ in range(copies):
        group = [make(n, k) for n, k in members]
        ws.append(group)

    def run():
        outs = [mx.quantized_matmul(x, w, s, b, transpose=True,
                                    group_size=64, bits=4)
                for group in ws for w, s, b in group]
        mx.eval(outs)

    inner = copies * len(members)
    t = timed(run, inner)
    total_bytes = sum(wbytes(n, k) for n, k in members)
    print(f"{label:14s} {t * 1e6:8.1f} us/member  "
          f"{total_bytes * copies / (t * inner) / 1e9 * len(members):6.1f} GB/s",
          flush=True)
    del ws


def main():
    print(f"MLX_OMARCHY_QMM_SMALLN={SMALLN}", flush=True)
    x2560 = mx.random.normal((1, 2560)).astype(mx.bfloat16)
    x9728 = mx.random.normal((1, 9728)).astype(mx.bfloat16)
    mx.eval(x2560, x9728)
    # 3-member small-N group (the qkv trio shape, all n <= 4096)
    bench_group("qkv 4096+1024+1024", x2560, [(4096, 2560), (1024, 2560), (1024, 2560)], 96)
    # single small-N member (o_proj)
    bench_group("o_proj 2560x2560", x2560, [(2560, 2560)], 192)
    # single small-N member, deep K (down_proj)
    bench_group("down 2560x9728", x9728, [(2560, 9728)], 64)
    # 2-member large group (gate+up, stays on the wide tile)
    bench_group("gate+up 9728+9728", x2560, [(9728, 2560), (9728, 2560)], 32)
    # saturated reference
    bench_group("lm_head 151936", x2560, [(151936, 2560)], 3)


if __name__ == "__main__":
    main()
