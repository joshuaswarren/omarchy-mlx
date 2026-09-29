# Randomized equivalence check: vulkan_mel.py fma32/add32/sub32/mul32 hardware fast path vs the integer emulation.
# usage: python check-mel-fma-fastpath.py <overlay/tools dir> [batches of 8M triples per category]; expects 0 mismatches.
import json
import sys
import time

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + "/coreml")
import mlx.core as mx
import numpy as np
from coreml import vulkan_mel as vm

N = 1 << 23  # 8M triples per batch
SRC = """
uint i = thread_position_in_grid.x;
uint ab = a[i];
uint bb = b[i];
uint cb = c[i];
float fa = uintBitsToFloat(ab);
float fb = uintBitsToFloat(bb);
float fc = uintBitsToFloat(cb);
uint m = 0u;
if (floatBitsToUint(fma32(fa, fb, fc)) != fma32_bits(ab, bb, cb)) m |= 1u;
if (floatBitsToUint(add32(fa, fb)) != add32_bits(ab, bb)) m |= 2u;
if (floatBitsToUint(sub32(fa, fb)) != sub32_bits(ab, bb)) m |= 4u;
if (floatBitsToUint(mul32(fa, fb)) != mul32_bits(ab, bb)) m |= 8u;
out[i] = m;
"""
kernel = mx.fast.metal_kernel(
    name="fma_equiv", input_names=["a", "b", "c"], output_names=["out"],
    header=vm._FMA_HEADER, source=SRC, compile_options={"math_mode": "safe"})


def run(a, b, c):
    o = kernel(inputs=[mx.array(a), mx.array(b), mx.array(c)], output_shapes=[(N,)],
               output_dtypes=[mx.uint32], grid=(N, 1, 1), threadgroup=(256, 1, 1), stream=mx.gpu)[0]
    mx.eval(o)
    o = np.array(o)
    return o, [int(((o >> k) & 1).sum()) for k in range(4)]


rng = np.random.default_rng(12345)


def f2u(x):
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def u2f(x):
    return np.asarray(x, dtype=np.uint32).view(np.float32)


def rand_bits():
    return [rng.integers(0, 2**32, N, dtype=np.uint64).astype(np.uint32) for _ in range(3)]


def normal_only(lo=1, hi=254):
    out = []
    for _ in range(3):
        e = rng.integers(lo, hi + 1, N).astype(np.uint32)
        m = rng.integers(0, 1 << 23, N).astype(np.uint32)
        s = rng.integers(0, 2, N).astype(np.uint32)
        out.append((s << 31) | (e << 23) | m)
    return out


def near_cancel():
    e = rng.integers(60, 190, N).astype(np.uint32)
    m1 = rng.integers(0, 1 << 23, N).astype(np.uint32)
    m2 = rng.integers(0, 1 << 23, N).astype(np.uint32)
    a = (e << 23) | m1
    b = ((rng.integers(100, 154, N).astype(np.uint32)) << 23) | m2
    fa, fb = u2f(a), u2f(b)
    prod = (fa.astype(np.float64) * fb.astype(np.float64))
    c = f2u(-prod.astype(np.float32)) + rng.integers(-3, 4, N).astype(np.int64).astype(np.uint32)
    return [a, b, c]


def add_cancel():
    a = normal_only(80, 170)[0]
    d = rng.integers(-3, 4, N).astype(np.int64)
    b = (a ^ np.uint32(0x80000000)) + d.astype(np.uint32)
    c = normal_only(1, 254)[2]
    return [a, b, c]


def small_boundary():
    return normal_only(1, 40)


def zeros_specials():
    vals = np.array([0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x00000001, 0x007fffff, 0x00800000,
                     0x7f7fffff, 0x7f800000, 0xff800000, 0x7fc00000, 0x3f000000, 0x40000000, 0x00800001], dtype=np.uint32)
    return [vals[rng.integers(0, len(vals), N)] for _ in range(3)]


def big_boundary():
    return normal_only(200, 254)


cats = [("uniform_bits", rand_bits), ("normal_all", normal_only), ("moderate", lambda: normal_only(100, 154)),
        ("near_cancel_fma", near_cancel), ("cancel_add", add_cancel), ("small_exp", small_boundary),
        ("big_exp", big_boundary), ("zeros_specials", zeros_specials)]
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
total = np.zeros(4, dtype=np.int64)
n_total = 0
detail = {}
first_bad = None
t0 = time.time()
for name, gen in cats:
    cnt = np.zeros(4, dtype=np.int64)
    for _ in range(reps):
        a, b, c = gen()
        o, k = run(a, b, c)
        cnt += k
        if first_bad is None and o.any():
            idx = int(np.argmax(o != 0))
            first_bad = (name, hex(int(a[idx])), hex(int(b[idx])), hex(int(c[idx])), int(o[idx]))
    detail[name] = cnt.tolist()
    total += cnt
    n_total += N * reps
print("EQUIV", json.dumps({"triples_per_op": n_total, "mismatch_fma_add_sub_mul": total.tolist(), "by_category": detail,
                           "first_bad": first_bad, "seconds": round(time.time() - t0, 1)}))
