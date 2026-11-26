#!/usr/bin/env python3
"""Consumer-boundary dense-normalization contract checks (no flags).

Exercises the primitives.cpp ensure_dense sites with VALID STRIDED VIEWS
against numpy references, plus the ordinary dense path. Two view classes
reach the backend while the eval.cpp Slice densifier is active:

1. Transposed views (Transpose primitive): gapless, flags().contiguous
   but NOT row_contiguous. These previously hit the named
   "non-contiguous <op>" refusals; they now materialize once at the
   consumer through contiguous_copy_gpu.
2. as_strided views (AsStrided primitive): arbitrary strides, including
   gapped (contiguous=false). The elementwise/int/complex/DivMod funnels
   previously refused these; they now densify.

Bounded shapes only, no models. Exit 0 = all checks pass.
"""
import sys

import numpy as np
import mlx.core as mx

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} {detail}")
        FAILURES.append(name)


def draw(key, shape, dtype=mx.float32):
    a = mx.random.normal(shape, key=mx.random.key(key))
    if dtype in (mx.int32, mx.uint32):
        return (a * 100).astype(dtype)
    if dtype == mx.complex64:
        return a.astype(mx.float32) + 1j * mx.random.normal(
            shape, key=mx.random.key(key + 1000))
    return a.astype(dtype)


def as_strided_gapped(a, step=2):
    """A gapless-per-row but gapped-across-rows view (contiguous=false)."""
    shape = a.shape
    full = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        full[i] = full[i + 1] * shape[i + 1]
    rows, cols = shape[-2], shape[-1]
    kept = (cols + step - 1) // step
    return mx.as_strided(
        a, shape=tuple(shape[:-2]) + (rows, kept),
        strides=tuple(full[:-2]) + (full[-2], step))


def close(a, b, tol=1e-5):
    a = np.asarray(a, dtype=np.float64) if not np.iscomplexobj(
        np.asarray(a)) else np.asarray(a)
    return np.allclose(a, b, rtol=tol, atol=tol)


print("== float elementwise funnel ==")
x = draw(1, (6, 8))
ref = np.asarray(x)[:, ::2] * 2.0 + 1.0
got = as_strided_gapped(x) * 2.0 + 1.0
mx.eval(got)
check("elementwise f32 gapped view", close(got, ref))

xt = x.T
ref = np.asarray(xt) * 3.0 - 2.0
got = xt * 3.0 - 2.0
mx.eval(got)
check("elementwise f32 transposed view", close(got, ref))

print("== int elementwise funnel ==")
xi = draw(2, (6, 8), mx.int32)
ref = np.asarray(xi)[:, ::2] - 7
got = as_strided_gapped(xi) - 7
mx.eval(got)
check("elementwise i32 gapped view", close(got, ref))
ref = np.asarray(xi.T) ^ 0x0F
got = xi.T ^ 0x0F
mx.eval(got)
check("bitwise i32 transposed view", close(got, ref))

print("== complex funnel ==")
xc = draw(3, (6, 8), mx.complex64)
ref = np.asarray(xc)[:, ::2] * (1 + 1j)
got = as_strided_gapped(xc) * (1 + 1j)
mx.eval(got)
check("complex gapped view", close(got, ref))
got = xc.T * (2 - 1j)
mx.eval(got)
check("complex transposed view", close(got, np.asarray(xc.T) * (2 - 1j)))
xi = draw(4, (6, 8), mx.int32)
yi = mx.abs(draw(5, (6, 8), mx.int32)) | 1
q, r = mx.divmod(as_strided_gapped(xi), as_strided_gapped(yi))
mx.eval(q, r)
ref_a = np.asarray(xi)[:, ::2]
ref_b = (np.abs(np.asarray(yi)) | 1)[:, ::2]
check("divmod gapped view quotient", close(q, ref_a // ref_b))
check("divmod gapped view remainder", close(r, ref_a % ref_b))

print("== sort family ==")
x = draw(6, (10, 32))
for name, op, np_op in [
        ("sort", mx.sort, np.sort),
        ("argsort", mx.argsort, np.argsort)]:
    got = op(x.T, axis=-1)
    mx.eval(got)
    check(f"{name} transposed view",
          np.array_equal(np.asarray(got), np_op(np.asarray(x.T), axis=-1)))
    got = op(as_strided_gapped(x), axis=-1)
    mx.eval(got)
    check(f"{name} gapped view",
          np.array_equal(
              np.asarray(got), np_op(np.asarray(x)[:, ::2], axis=-1)))
got = mx.partition(x.T, kth=5, axis=-1)
mx.eval(got)
# The backend serves Partition with a full sort (documented redirect),
# so the sorted order is the expected partition result for any kth.
ref = np.sort(np.asarray(x.T), axis=-1)
check("partition transposed view (full-sort redirect)",
      np.array_equal(np.asarray(got), ref))
got = mx.argpartition(x.T, kth=5, axis=-1)
mx.eval(got)
src = np.asarray(x.T)
idx = np.asarray(got)
prefix_ok = True
for row in range(src.shape[0]):
    sel = src[row, idx[row]]
    prefix_ok &= bool(
        (sel[:5] <= sel[5]).all() and (sel[5] <= sel[6:]).all())
check("argpartition transposed view valid kth order", prefix_ok)
print("== softmax / argreduce / logsumexp ==")
x = draw(7, (8, 16))
got = mx.softmax(x.T, axis=-1)
ref = np.exp(np.asarray(x.T) - np.asarray(x.T).max(-1, keepdims=True))
ref = ref / ref.sum(-1, keepdims=True)
mx.eval(got)
check("softmax transposed view", close(got, ref, tol=1e-6))
check("argmax transposed view",
      np.array_equal(np.asarray(mx.argmax(x.T, axis=-1)),
                     np.asarray(x.T).argmax(axis=-1)))
got = mx.logsumexp(x.T, axis=-1, keepdims=True)
ref = np.log(np.exp(np.asarray(x.T) - np.asarray(x.T).max(
    -1, keepdims=True)).sum(-1, keepdims=True)) + np.asarray(x.T).max(
        -1, keepdims=True)
mx.eval(got)
check("logsumexp transposed view", close(got, ref, tol=1e-5))
print("== take / take_axis ==")
table = draw(8, (12, 6))
idx = mx.abs(draw(9, (5,), mx.int32)) % 6
got = mx.take(table.T, idx, axis=0)
ref = np.asarray(table.T)[np.asarray(idx)]
mx.eval(got)
check("take strided table", close(got, ref))
src = draw(10, (4, 9))
idx2 = mx.abs(draw(11, (9, 5), mx.int32)) % 4
got = mx.take_along_axis(src.T, as_strided_gapped(idx2), axis=1)
mx.eval(got)
check("take_along_axis strided table and indices", close(
    got, np.take_along_axis(
        np.asarray(src.T), np.asarray(idx2)[:, ::2], axis=1)))

print("== searchsorted ==")
seq = mx.sort(draw(11, (16,)))
vals = draw(12, (10,))
got = mx.searchsorted(seq, as_strided_gapped(vals[:, None])[:, 0])
ref = np.searchsorted(np.asarray(seq), np.asarray(vals))
mx.eval(got)
check("searchsorted strided values", np.array_equal(
    np.asarray(got), ref.astype(np.asarray(got).dtype)))

print("== fast norms ==")
x = draw(13, (2, 6, 16))
xt = np.asarray(x).transpose(0, 2, 1)
w = draw(14, (6,))
got = mx.fast.rms_norm(x.transpose(0, 2, 1), w, 1e-5)
ref = (xt * (1.0 / np.sqrt((xt ** 2).mean(-1, keepdims=True) + 1e-5))
       * np.asarray(w))
mx.eval(got)
check("rms_norm transposed view", close(got, ref, tol=1e-4))
got = mx.fast.layer_norm(x.transpose(0, 2, 1), w, w, 1e-5)
ref = ((xt - xt.mean(-1, keepdims=True))
       / np.sqrt(xt.var(-1, keepdims=True) + 1e-5) * np.asarray(w)
       + np.asarray(w))
mx.eval(got)
check("layer_norm transposed view", close(got, ref, tol=1e-4))

print("== quantized matmul / dequantize ==")
x = draw(15, (64, 4))
wq = mx.array(np.random.default_rng(16).integers(
    0, 2 ** 31, size=(8, 8), dtype=np.uint32))
scales = draw(17, (8, 2), mx.float16).astype(mx.float32)
biases = mx.zeros((8, 2), mx.float32)
got = mx.quantized_matmul(x.T, wq, scales, biases, transpose=True,
                          group_size=32, bits=4)
w_ref = mx.dequantize(wq, scales, biases, group_size=32, bits=4)
ref = np.asarray(x.T).astype(np.float64) @ np.asarray(w_ref).T.astype(
    np.float64)
mx.eval(got)
check("quantized_matmul transposed x", close(got, ref, tol=1e-3))
wv = as_strided_gapped(wq)
sv = as_strided_gapped(scales)
bv = as_strided_gapped(biases)
got = mx.dequantize(wv, sv, bv, group_size=32, bits=4)
# Reference: the same logical views densified host-independently via
# mx.contiguous, so the kernel sees identical dense bytes either way.
ref2 = mx.dequantize(
    mx.contiguous(wv), mx.contiguous(sv), mx.contiguous(bv),
    group_size=32, bits=4)
mx.eval(got)
check("dequantize gapped views", np.array_equal(
    np.asarray(got), np.asarray(ref2)))

logits = draw(18, (16, 10))
targets = (mx.random.uniform(
    shape=(10,), key=mx.random.key(19)) * 16).astype(mx.int32)
got = mx.fast.cross_entropy(logits.T, targets)
mx.eval(got)
lg = np.asarray(logits.T)
ref = (-lg[np.arange(10), np.asarray(targets)]
       + np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1))
       + lg.max(-1))
check("cross_entropy transposed logits", close(got, ref, tol=1e-4))

print("== dense fast-path identity ==")
x = draw(20, (8, 16))
got = x * 2.0 + 1.0
mx.eval(got)
check("dense elementwise unchanged", close(
    got, np.asarray(x) * 2.0 + 1.0))
check("dense sort unchanged", np.array_equal(
    np.asarray(mx.sort(x, axis=-1)), np.sort(np.asarray(x), axis=-1)))

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all strided-consumer checks passed")
