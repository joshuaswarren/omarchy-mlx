#!/usr/bin/env python3
"""RingTurnaround chain slopes: dependent-op chain wall vs n -> us/op.

Reproduces the Jw16DecodeGap3 chain_costs_decode.py method (add bf16 [1,2048]
RAW chain; q4 2048x2048 GEMV RAW chain), n=32..160, 7 reps, min-of-reps wall.
Run inside a gpuwin window; arms differ ONLY by HK_CDM_BARRIER_MASK env.
Output: JSON on stdout.
"""
import json
import statistics
import time

import mlx.core as mx

NS = [32, 64, 96, 128, 160]
REPS = 7


def chain_wall(build, n):
    ts = []
    for _ in range(REPS):
        y = build(n)
        t0 = time.perf_counter()
        mx.eval(y)
        ts.append((time.perf_counter() - t0) * 1e6)
    return min(ts)


def slope_us_per_op(ns, walls):
    # least squares over (n, wall) with intercept
    n0 = ns[0]
    xs = [n - n0 for n in ns]
    ys = walls
    mx_ = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx_) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx_) ** 2 for x in xs)
    return num / den


def main():
    x = mx.ones((1, 2048), dtype=mx.bfloat16)

    def add_chain(n):
        y = x
        for _ in range(n):
            y = y + x
        return y

    w = mx.random.normal((2048, 2048)) * 0.02
    wq, scales, biases = mx.quantize(w)
    mx.eval(wq, scales, biases)

    def gemv_chain(n):
        y = x
        for _ in range(n):
            y = mx.quantized_matmul(y, wq, scales, biases, transpose=True)
        return y

    # warmup
    mx.eval(add_chain(32), gemv_chain(32))

    add_ns = [chain_wall(add_chain, n) for n in NS]
    gemv_ns = [chain_wall(gemv_chain, n) for n in NS]
    add_slope = slope_us_per_op(NS, add_ns)
    gemv_slope = slope_us_per_op(NS, gemv_ns)
    gemv_bytes = (2048 * 2048) // 2 + (2048 * 2048 // 64) * (2 + 2) + 2048 * 2 * 2
    out = {
        "instrument": "ring-turnaround-chain-slopes v1 (DecodeGap3 method)",
        "add_us_per_op": round(add_slope, 3),
        "add_walls_us": dict(zip(NS, (round(v, 1) for v in add_ns))),
        "gemv_us_per_op": round(gemv_slope, 3),
        "gemv_walls_us": dict(zip(NS, (round(v, 1) for v in gemv_ns))),
        "gemv_gbps": round(gemv_bytes / (gemv_slope * 1e-6) / 1e9, 1),
        "mask_env": __import__("os").environ.get("HK_CDM_BARRIER_MASK", "<unset>"),
        "mlx_version": getattr(mx, "__version__", "?"),
        "utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    print(json.dumps(out))


if __name__ == "__main__":
    assert statistics  # keep import explicit
    main()
