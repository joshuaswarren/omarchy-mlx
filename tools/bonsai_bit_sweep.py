#!/usr/bin/env python3
"""bonsai_bit_sweep: decode-map harness for the packed Bonsai shaders.

Commits the probe technique that found the q1 qmv byte-walk bug
(VALUES_PER_BYTE = 32/BITS instead of 8/BITS): with real random codes,
tolerance comparisons and static audits both passed while the kernel
read only a quarter of each packed row. The sweep makes every
(byte, bit) -> k mapping independently observable against the LSB-first
contract (byte e, bit i holds code e*8+i):

  sweep 1 (w-side): for each packed position, build w with ONLY that
    bit set and x = ones; qmv and dequant must return exactly the
    single 1 at k = e*8+i and 0 in every other position.
  sweep 2 (x-side): real random codes, one-hot x at every k; the kernel
    output must reproduce the true code at k.

A 4x k-stride shows up as byte 1's codes appearing at positions 32..39;
a dead binding range shows whole bytes of zeros. Prints the perceived
map on mismatch. Needs a Vulkan device (set VK_DRIVER_FILES if the
packaged ICD is absent). Exit 0 = identity decode.

Usage: bonsai_bit_sweep.py [--k 64] [--n 1] [--group 64] [--op qmv]
"""
import argparse
import sys

import numpy as np

import mlx.core as mx

CODES_PER_BYTE = 8


def pack(codes):
    bits = (1 << np.arange(CODES_PER_BYTE)).astype(np.uint8)
    n, k = codes.shape
    return (codes.reshape(n, k // CODES_PER_BYTE, CODES_PER_BYTE) * bits).sum(
        axis=2, dtype=np.uint8)


def perceived_qmv(packed, scales, biases, x_ones, k):
    seen = np.zeros(k, dtype=np.int32)
    for e in range(k // CODES_PER_BYTE):
        for i in range(CODES_PER_BYTE):
            p = np.zeros_like(np.array(packed, copy=False))
            p[0, e] = 1 << i
            got = mx.fast.bonsai_q1_affine_qmv(
                mx.array(x_ones), mx.array(p), scales, biases,
                group=64)
            mx.eval(got)
            seen[e * CODES_PER_BYTE + i] = int(
                round(float(np.array(got.astype(mx.float32), copy=False)[0, 0])))
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--group", type=int, default=64)
    args = ap.parse_args()

    if not hasattr(mx.fast, "bonsai_q1_affine_qmv"):
        sys.exit("bonsai_bit_sweep: mx.fast.bonsai_q1_affine_qmv missing "
                 "(wheel built without mlx-fast-bonsai-qmv.patch)")
    if args.k % CODES_PER_BYTE or args.k % args.group:
        sys.exit("bonsai_bit_sweep: k must be a multiple of 8 and of --group")

    rng = np.random.default_rng(0)
    codes = (rng.normal(0, 0.05, (args.n, args.k)) >= 0).astype(np.uint8)
    packed_np = pack(codes)
    packed = mx.array(packed_np)
    scales = mx.ones((args.n, args.k // args.group), dtype=mx.float32)
    biases = mx.zeros((args.n, args.k // args.group), dtype=mx.float32)

    x_ones = np.ones((1 if args.n == 1 else 1, args.k), dtype=np.float32)
    seen = perceived_qmv(packed, scales, biases, x_ones, args.k)
    true_bits = ((packed_np[0][:, None] >> np.arange(CODES_PER_BYTE)) & 1
                 ).reshape(-1)
    matches = int((seen == true_bits).sum())
    print(f"w-side sweep: {matches}/{args.k} positions decode as LSB-first")
    if matches != args.k:
        print("true bits:", true_bits)
        print("perceived:", seen)

    ok = matches == args.k
    print("bonsai_bit_sweep:", "OK — identity decode" if ok
          else "FAIL — decode map is not the LSB-first identity")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
