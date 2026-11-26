# Probe: shapeless compiled fragment reused at a new input shape.
#
# The compile cache matches shapeless entries by ndim and dtype only, so
# one traced tape legally serves every shape. The interpreter must derive
# node output shapes from the eval-time inputs; using the traced shapes
# poisons the fragment (stale-shape outputs, broadcast-Sigmoid refusal on
# the M1 at [1,30,4864] / [1,36,...]).
#
# Modes mirror the mlx-lm fragment class where the refusal was observed:
#   swiglu: sigmoid(x) * y + scalar  (qwen2 swiglu shape)
#   chain:  a longer elementwise fragment
#
# Exit codes: 0 clean, 3 mismatch, 4 refusal raised.

import os
import sys

os.environ.setdefault("MLX_OMARCHY_ALLOW_NON_APPLE", "1")

import mlx.core as mx


def swiglu(x, y):
    return mx.sigmoid(x) * y + 0.5


def chain(x, y):
    z = mx.sigmoid(x) * y
    z = z + 1.0
    z = mx.sigmoid(z) * 2.0
    return z * z + x


def bitwise(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return mx.array_equal(a, b, equal_nan=True).item()


def run_case(name, fn, trace_shape, eval_shape, dtype=mx.float16):
    print(f"== case {name}: trace {trace_shape} then eval {eval_shape} ({dtype}) ==")
    x1 = mx.random.normal(trace_shape).astype(dtype)
    y1 = mx.random.normal(trace_shape).astype(dtype)
    ref1 = fn(x1, y1)
    compiled = mx.compile(fn, shapeless=True)
    out1 = compiled(x1, y1)
    mx.eval(out1)
    if not bitwise(out1, ref1):
        print(f"FAIL: trace-shape call already mismatches ({name})")
        return 3

    x2 = mx.random.normal(eval_shape).astype(dtype)
    y2 = mx.random.normal(eval_shape).astype(dtype)
    ref2 = fn(x2, y2)
    out2 = compiled(x2, y2)
    try:
        mx.eval(out2)
    except RuntimeError as e:
        print(f"REFUSAL at eval shape: {e}")
        return 4
    if not bitwise(out2, ref2):
        bad = mx.abs(out2.astype(mx.float32) - ref2.astype(mx.float32)).max().item()
        print(f"FAIL: eval-shape call mismatches (max abs diff {bad}) ({name})")
        return 3
    print(f"clean ({name})")
    return 0


def main():
    rcs = []
    for name, fn in (("swiglu", swiglu), ("chain", chain)):
        rcs.append(run_case(name, fn, (1, 8, 64), (1, 36, 64)))
        rcs.append(run_case(name + "-decode", fn, (1, 36, 64), (1, 1, 64)))
    if any(r == 3 for r in rcs):
        return 3
    if any(r == 4 for r in rcs):
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
