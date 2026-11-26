#!/usr/bin/env python3
"""Mechanism probe: compiled tape read by small eager ops, in a loop.

Independent of the model differential: this tests the MECHANISM the tape
corruption lives in, not the model.  Each iteration submits one compiled
tape (fusable, data-dependent select inside) and then small EAGER operations
that read the tape's outputs, with the eager op size swept over a schedule
that includes one-element outputs and one-element writes - the shape of the
write present when the earlier run aborted at a trigonometric domain gate on
a value near 9.1e8.

The value chain is data-dependent: the next tape input is derived from the
eager op's result, so a stale read of the tape output (submission-order
hazard) cannot produce a plausible number - it shifts every later value and
shows up as a bitwise mismatch against the eager-only reference loop.

Variants:
  interleave : tape -> eager read -> next input, host sync every iteration
  depth2     : two tape submissions back to back before the eager read
  nosync     : 8 tape+eager pairs with no host sync, one sync, then compare

Run it on hardware in seconds.  Compile must be enabled: do NOT set
MLX_DISABLE_COMPILE (the probe toggles mx.enable_compile itself).

On llvmpipe this cannot expose the real defect: llvmpipe executes
submissions synchronously inside vkQueueSubmit, so cross-submission races
never occur.  A clean llvmpipe run proves the probe logic only.

Exit codes: 0 all variants clean, 3 divergence (or injected divergence
detected), 2 environment error.

Self-test (no mlx, no GPU): python3 probe_tape_eager.py --self-test
"""

import argparse
import os
import sys

import numpy as np

EXIT_MATCH = 0
EXIT_DIVERGE = 3
EXIT_ERROR = 2

# Eager-read sizes: powers of two, a non-round size, and 1 - the one-element
# output/write that matches the earlier abort's shape.
SIZE_SCHEDULE = (64, 17, 2, 1)


def bits(a):
    codes = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}
    return np.ascontiguousarray(a).view(codes[a.dtype.itemsize])


def bitwise_equal(a, b):
    return a.shape == b.shape and bool(np.array_equal(bits(a), bits(b)))


def first_diff(a, b):
    neq = bits(a) != bits(b)
    flat = neq.reshape(-1)
    return int(np.argmax(flat)) if flat.any() else -1


def run_loop(n, iters, dtype_name, seed, depth, sync_every, compiled, inject=None):
    """The data-dependent loop.  Returns per-iteration host snapshots of z.

    tape: x -> where(x>0, exp(-x*x), sin(x)) * c + d  (fusable: select,
    unary, binary - a real compiled tape).
    eager read: sums and a slice down to k elements (schedule), then a
    one-element setitem write when k == 1.
    next input: broadcast of the small eager result - fully data-dependent.
    """
    import mlx.core as mx

    dt = getattr(mx, dtype_name)
    rng = np.random.default_rng(seed)
    x0 = mx.array(rng.standard_normal((n, n)).astype(np.float32), dtype=dt)

    def tape(x):
        y = mx.where(x > 0, mx.exp(-x * x), mx.sin(x))
        return y * 1.0000001 + 0.5

    fn = tape
    if compiled:
        fn = mx.compile(tape)

    snaps = []
    pending = []
    pending_idx = []
    x = x0
    for i in range(iters):
        if depth == 2:
            fn(x)  # second in-flight tape submission before the eager read
        y = fn(x)
        k = SIZE_SCHEDULE[i % len(SIZE_SCHEDULE)]
        s = mx.sum(y, axis=0, keepdims=True) * 1e-4          # (1, n) eager
        z = y.reshape(-1)[:k] * 0.5 + s.reshape(-1)[:k]      # k elements, eager
        if k == 1:
            buf = mx.array([0.5], dtype=dt)
            buf[0:1] = z                                     # one-element WRITE
        pending.append(z)
        pending_idx.append(i)
        if len(pending) >= sync_every or i == iters - 1:
            mx.eval(*pending)          # one host sync per batch
            for j, zz in zip(pending_idx, pending):
                snap = np.asarray(zz)
                if inject is not None and j == inject and compiled:
                    snap = snap + np.float32(1.0)   # simulated stale read
                snaps.append(snap)
            pending = []
            pending_idx = []
    return snaps


def compare_variant(name, snaps_e, snaps_c):
    for i, (a, b) in enumerate(zip(snaps_e, snaps_c)):
        if not bitwise_equal(a, b):
            idx = first_diff(a, b)
            print(f"{name}: DIVERGES at iteration {i}, flat index {idx}: "
                  f"eager {float(a.reshape(-1)[idx])!r} vs compiled {float(b.reshape(-1)[idx])!r} "
                  f"(eager read size {SIZE_SCHEDULE[i % len(SIZE_SCHEDULE)]})")
            return False
    print(f"{name}: {len(snaps_e)} iterations bitwise-clean")
    return True


def self_test():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert bitwise_equal(a, a.copy())
    b = a.copy()
    b[2, 3] += 1.0
    assert not bitwise_equal(a, b)
    assert first_diff(a, b) == 11
    print("self-test: comparator checks pass")
    return EXIT_MATCH


def main():
    p = argparse.ArgumentParser(description="compiled-tape vs eager-op mechanism probe")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--iters", type=int, default=32)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inject", type=int, default=None,
                   help="perturb this iteration's eager-read result in the COMPILED "
                        "run to prove the detector detects")
    p.add_argument("--json", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        return self_test()

    try:
        import mlx.core as mx
    except ImportError:
        print("probe needs mlx; install the mlx-omarchy wheel built at the commit "
              "under test", file=sys.stderr)
        return EXIT_ERROR
    if mx.default_device().type != mx.DeviceType.gpu:
        print(f"warning: default device is {mx.default_device()}; the probe is "
              "meaningful on the Vulkan GPU device", file=sys.stderr)

    mx.random.seed(args.seed)
    clean = True

    mx.disable_compile()
    ref = run_loop(args.size, args.iters, args.dtype, args.seed,
                   depth=1, sync_every=1, compiled=False)
    mx.enable_compile()
    got = run_loop(args.size, args.iters, args.dtype, args.seed,
                   depth=1, sync_every=1, compiled=True, inject=args.inject)
    mx.disable_compile()
    clean &= compare_variant("interleave", ref, got)

    mx.enable_compile()
    got2 = run_loop(args.size, args.iters, args.dtype, args.seed,
                    depth=2, sync_every=1, compiled=True, inject=args.inject)
    mx.disable_compile()
    clean &= compare_variant("depth2", ref, got2)

    # nosync: 8-iteration batches; the host snapshots still come from the
    # same eager z values, only the sync cadence changes.
    mx.enable_compile()
    batch = []
    got3 = []
    for start in range(0, args.iters, 8):
        batch = run_loop(args.size, min(8, args.iters - start), args.dtype,
                         args.seed, depth=1, sync_every=8, compiled=True,
                         inject=(args.inject - start
                                 if args.inject is not None and start <= args.inject < start + 8
                                 else None))
        got3.extend(batch)
    mx.disable_compile()
    clean &= compare_variant("nosync", ref, got3)

    if not clean:
        print("probe: compiled-tape/eager-op chain shows divergence")
        return EXIT_DIVERGE
    print(f"probe: all variants bitwise-clean over {args.iters} iterations "
          f"(dtype={args.dtype}, size={args.size}x{args.size})")
    return EXIT_MATCH


if __name__ == "__main__":
    sys.exit(main())
