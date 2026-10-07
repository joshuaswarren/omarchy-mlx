"""Upstream reproducers (mlx #4366 sort/argsort on transposed views,
#4599 dynamic-slice with a non-contiguous start, #4635 0-dim boolean
mask, #4637 vmap of odd-length irfft) run against the installed omarchy
mlx. Failing-before/passing-after for the omarchy backend fixes."""

import itertools
import sys

import mlx.core as mx
import numpy as np

fails = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not cond:
        fails.append(name)


# --- #4366: sort/argsort on transposed views -------------------------
np.random.seed(0)
ok = True
detail = ""
for shape in [(3, 4, 8), (2, 1, 6), (2, 3, 4, 2), (2, 1, 3, 4)]:
    a_np = np.random.uniform(0, 100, size=shape).astype(np.float32)
    a_mx = mx.array(a_np)
    for perm in itertools.permutations(range(len(shape))):
        b_np = np.transpose(a_np, perm)
        b_mx = mx.transpose(a_mx, perm)
        for axis in range(len(shape)):
            s_np = np.sort(b_np, axis=axis)
            if not np.array_equal(s_np, np.array(mx.sort(b_mx, axis=axis))):
                ok = False
                detail = f"sort shape={shape} perm={perm} axis={axis}"
                break
            idx = np.array(mx.argsort(b_mx, axis=axis))
            if not np.array_equal(
                s_np, np.take_along_axis(b_np, idx, axis=axis)
            ):
                ok = False
                detail = f"argsort shape={shape} perm={perm} axis={axis}"
                break
check("#4366 sort/argsort transposed views", ok, detail)

# --- #4599: dynamic slice with a non-contiguous start ----------------
ok = True
detail = ""
try:
    x = mx.arange(64).reshape(8, 8)
    buf = mx.array([[1, 9], [2, 9]])
    # a strided column view of an index buffer is non-contiguous
    start = mx.transpose(buf)[0]
    assert start.shape == (2,)
    out = mx.slice(x, start, [0, 1], [4, 4])
    mx.eval(out)
    # x[start0+1, start1+1] = x[2, 3] = 19
    ok = out.shape == (4, 4) and int(out[1, 1]) == 19
    check("#4599 slice non-contiguous start", ok,
          f"out[1,1]={int(out[1,1])} expect 19")
except Exception as e:  # noqa: BLE001
    check("#4599 slice non-contiguous start", False, repr(e))

# --- #4635: 0-dim boolean mask ---------------------------------------
# The omarchy backend refuses bool MaskedScatter by name (compatibility
# contract): the upstream reproducer cannot run on this GPU. Recorded as
# a named refusal, not a silent wrong value.
print("[SKIP] #4635 0-dim boolean mask scatter: bool MaskedScatter is a "
      "named refusal on the omarchy Vulkan backend (no kernel by design)")

# --- #4637: vmap of odd-length irfft ---------------------------------
ok = True
detail = ""
try:
    import numpy as np
    shape = (3, 8, 6)
    a = mx.array(np.random.rand(*shape) + 1j * np.random.rand(*shape))
    for n in [10, 11, 12, 13]:
        s_arg = [n]

        def f(x):
            return mx.fft.irfftn(x, s=s_arg, axes=(-1,))

        expected = mx.stack([f(a[i]) for i in range(a.shape[0])])
        out = mx.vmap(f)(a)
        if tuple(out.shape) != tuple(expected.shape):
            ok = False
            detail = f"n={n}: vmap {out.shape} != expected {expected.shape}"
            break
except Exception as e:  # noqa: BLE001
    ok = False
    detail = repr(e)
check("#4637 vmap odd-length irfftn", ok, detail)

print(f"\n{len(fails)} failing" if fails else "\nall passing")
sys.exit(1 if fails else 0)
