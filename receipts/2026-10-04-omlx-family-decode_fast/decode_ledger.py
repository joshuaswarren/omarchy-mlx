#!/usr/bin/env python3
"""Op-level decode SDPA ledger for the hd-256 bf16 route (FamDecodeFast).

Times mx.fast.scaled_dot_product_attention on the Qwen3.5-class decode
shape (q (1,16,1,256), k/v (1,4,k,256) strided capacity-backed views,
bf16) at k in {64, 128, 512, 1024, 4096, 8192}, reporting median us/call
and the route's dispatch structure. Run it twice for the ON/OFF ledger:

    MLX_OMARCHY_SDPA_DECODE_NATIVE=0 python3 decode_ledger.py --label off
    python3 decode_ledger.py --label on

The OFF side forces the ~10-dispatch f32-score composition; the ON side
is the fused one-pass (inside the window) / two-pass split-KV (past it)
route. Every call's output is checked against the composed reference the
first time each k is seen, and the bf16 word-agreement fraction is
printed alongside the timing. Peak transient memory is read from
mx.get_peak_memory around a steady-state window.

Provenance: the caller must run scripts/mlx_provenance.py beside this
(an unlabelled number is not evidence).
"""

import argparse
import json
import statistics
import sys
import time

import mlx.core as mx

HEADS = 16
KV_HEADS = 4
HD = 256
SCALE = HD**-0.5
KEYS = (64, 128, 512, 1024, 4096, 8192)
REPS = 200
ROUNDS = 5


def make_inputs(keys, seed=0):
    state = {"s": seed}

    def pattern(count):
        state["s"] = (state["s"] * 1664525 + 1013904223) % (2**32)
        # Deterministic small generator; values in [-1, 1).
        out = []
        s = state["s"]
        for _ in range(count):
            s = (s * 1664525 + 1013904223) % (2**32)
            out.append((s % 20000) / 10000.0 - 1.0)
        state["s"] = s
        return mx.array(out)

    q = pattern(HEADS * HD).reshape(1, HEADS, 1, HD).astype(mx.bfloat16)
    k_cache = (
        pattern(KV_HEADS * keys * HD)
        .reshape(1, KV_HEADS, keys, HD)
        .astype(mx.bfloat16)
    )
    v_cache = (
        pattern(KV_HEADS * keys * HD)
        .reshape(1, KV_HEADS, keys, HD)
        .astype(mx.bfloat16)
    )
    mx.eval(q, k_cache, v_cache)
    return q, k_cache, v_cache


def sdpa(q, k, v):
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=SCALE)


def composition(q, k, v):
    q32 = (q.astype(mx.float32) * SCALE).reshape(
        1, KV_HEADS, HEADS // KV_HEADS, 1, HD
    )
    k32 = k.astype(mx.float32).reshape(1, KV_HEADS, 1, k.shape[2], HD)
    v32 = v.astype(mx.float32).reshape(1, KV_HEADS, 1, k.shape[2], HD)
    scores = q32 @ k32.swapaxes(-1, -2)
    probs = mx.softmax(scores, axis=-1)
    out = probs @ v32
    return out.reshape(1, HEADS, 1, HD).astype(mx.bfloat16)


def bits(a):
    return a.astype(mx.uint16).astype(mx.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()

    results = []
    for keys in KEYS:
        q, k_cache, v_cache = make_inputs(keys)
        k = k_cache[:, :, :keys, :]  # identity slice keeps strided base
        v = v_cache[:, :, :keys, :]

        # Correctness once per k against the composition.
        got = sdpa(q, k, v)
        want = composition(q, k, v)
        mx.eval(got, want)
        gb = bits(got)
        wb = bits(want)
        diff = mx.abs(got.astype(mx.float32) - want.astype(mx.float32))
        max_diff = float(diff.max())
        agree = int((gb == wb).sum())
        total = int(gb.size)

        # Steady-state timing, alternating sides would fight allocator
        # state; each process is one side by construction.
        for _ in range(20):
            mx.eval(sdpa(q, k, v))
        samples = []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            for _ in range(args.reps):
                out = sdpa(q, k, v)
            mx.eval(out)
            samples.append(
                (time.perf_counter() - t0) / args.reps * 1e6)
        med = statistics.median(samples)
        mx.reset_peak_memory()
        out = sdpa(q, k, v)
        mx.eval(out)
        peak = mx.get_peak_memory()
        row = {
            "keys": keys,
            "us_per_call": round(med, 2),
            "max_abs_diff_vs_composition": max_diff,
            "bf16_words_identical": f"{agree}/{total}",
            "peak_transient_bytes": int(peak),
            "samples_us": [round(s, 2) for s in samples],
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    print(json.dumps({"label": args.label, "rows": results}), flush=True)


if __name__ == "__main__":
    sys.exit(main())
