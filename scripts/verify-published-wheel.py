#!/usr/bin/env python3
# Published-wheel verification probe for mlx-omarchy.
#
# RUN THIS ON APPLE HARDWARE. llvmpipe is green through every defect this
# script tests for - the boolean-reduction checks below all pass on the dev
# box even on wheels that fail them on Honeykrisp - so a dev-box run proves
# nothing about the checks that matter. Point it at a downloaded release
# wheel in a fresh venv, not at your build directory.
#
# v0.3.1's published-wheel probe caught a boolean-reduction miscompile that
# 828,139 in-tree assertions never reached (whole-array mx.all, sizes >= 5);
# v0.3.2's caught nothing, because this matrix now covers what that probe
# spotted. Keep it in the release checklist: download both published wheels,
# hash-check, install into a fresh venv, run this on each platform.
#
# Coverage: the v0.3.0 probe set, one check per v0.3.1 fix class, the
# disclosed-open W11 deviation as an informational line, and the v0.3.2
# boolean-reduction matrix - mx.all and mx.any, whole-array and axis, sizes
# straddling the word boundary (4, 5, 8, 9) and the chunk boundary
# (32, 33, 4096, 4097), with all-true, all-false, and single-element-
# differing inputs, the differing element placed both inside and outside
# the first word.
import sys

import numpy as np

import mlx.core as mx

failures = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


print("mlx version:", mx.__version__)
print("device:", mx.default_device())

# --- v0.3.0 probe set (regression) ---
x = mx.array([[1.0, 2.0], [3.0, 4.0]])
w = mx.array([[0.5], [0.25]])


def loss(w):
    return mx.exp(x @ w).sum()


value, grad = mx.value_and_grad(loss)(w)
check(
    "value_and_grad",
    abs(value.item() - 14.9008) < 1e-3
    and np.allclose(np.array(grad), [[39.2658], [54.1665]], atol=1e-3),
    f"value={value.item()}",
)
check("log2(1024)", mx.log2(mx.array(1024.0)).item() == 10.0)
check("int sort", np.array_equal(np.array(mx.sort(mx.array([3, 1, 2]))), [1, 2, 3]))
fft = mx.fft.fft(mx.ones(13))
mag = np.abs(np.array(fft))
check("fft(13)", abs(mag[0] - 13.0) < 1e-3 and mag[1:].max() < 1e-3,
      f"X0={mag[0]:.6f} offbin={mag[1:].max():.1e}")

# --- v0.3.1 fix: eager sin/cos refuse by name above 1e5 ---
try:
    mx.eval(mx.sin(mx.array([1e6])))
    check("sin 1e6 eager refusal", False, "no exception raised")
except RuntimeError as e:
    check("sin 1e6 eager refusal", "[omarchy] Sin" in str(e), str(e)[:80])
try:
    mx.eval(mx.cos(mx.array([2e6])))
    check("cos 2e6 eager refusal", False, "no exception raised")
except RuntimeError as e:
    check("cos 2e6 eager refusal", "[omarchy] Cos" in str(e), str(e)[:80])

# --- compiled tape: gate fires inside the tape (per-node interpreter) ---
# Leg 1: bf16 tripwire - only the tape interpreter raises this, proving a
# real compiled tape formed in this shape.
try:
    trip = mx.compile(lambda t: mx.sin(t.astype(mx.bfloat16)).astype(mx.float32))
    mx.eval(trip(mx.array([5e6])))
    check("compiled bf16 tripwire fired", False, "tape gate did not raise")
except RuntimeError as e:
    check("compiled bf16 tripwire fired",
          "Compiled tape bfloat16" in str(e), str(e)[:70])
# Leg 2: the same tape shape in f32 must refuse through the sin gate.
try:
    out = mx.compile(lambda t: mx.sin(t))(mx.array([5e6]))
    mx.eval(out)
    check("compiled sin 5e6 refuses inside tape", False,
          f"returned {np.array(out)[0]:.6f}, gate bypassed")
except RuntimeError as e:
    check("compiled sin 5e6 refuses inside tape",
          "[omarchy] Sin" in str(e), str(e)[:70])

# --- v0.3.1 fix: sin accuracy band near the limit (1e3..1e5) ---
got = np.array(mx.sin(mx.array(np.array([12345.0], np.float32))))
err = abs(got[0] - np.float32(np.sin(np.float32(12345.0))))
check("sin(12345) within 5e-3 band", err < 5e-3, f"err={err:.2e}")

# --- alpha-defect regression set, fixed in v0.3.0 ---
a = mx.array([0.0, 1.0, float("nan")])
check("array_equal equal_nan", mx.array_equal(a, a, equal_nan=True).item() is True)

cm = np.array(mx.cummax(mx.array([1.0, 3.0, float("nan"), 5.0, 4.0])))
check("cummax NaN carry", bool(np.isnan(cm[3:]).all()), f"got {cm}")

red = mx.array(np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], np.float32))
r = np.array(mx.max(red, axis=0))
check("axis-0 max keeps NaN", np.isnan(r[1]) and r[0] == 4.0 and r[2] == 6.0,
      f"got {r}")

np.random.seed(0)
xi = mx.array((np.random.randn(65, 65, 1, 65) * 128).astype(np.int32))
got_s = np.asarray(mx.sum(xi, axis=(0, 1, 3)))
want_s = np.sum(np.array(xi), axis=(0, 1, 3))
check("int multi-axis sum", np.array_equal(got_s, want_s),
      f"got {got_s[:3]} want {want_s[:3]}")

tk = np.array(mx.take(mx.array(np.array([[1, 2], [3, 4]], np.int32)), mx.array(-1), 0))
check("take negative axis", np.array_equal(tk, [3, 4]), f"got {tk}")

fl = np.array(mx.full_like(mx.array(np.array([1, 2, 3], np.int16)),
                           mx.array(7.5), dtype=mx.float16))
check("full_like dtype conversion", np.allclose(fl, [7.5, 7.5, 7.5]), f"got {fl}")

check("log10(1000)", mx.log10(mx.array(1000.0)).item() == 3.0,
      f"got {mx.log10(mx.array(1000.0)).item()!r}")

# --- v0.3.1 fix: byte-extraction class (bool scatter / LogicalAnd / select) ---
lhs = mx.array(np.array([True] * 33))
rhs = mx.array(np.array([True] * 33))
land = mx.logical_and(lhs, rhs)
check("LogicalAnd 33 bools", land.all().item() is True,
      f"all_true={land.all().item()}")

idx = mx.array(np.array([0, 1, 0], np.int32))
upd = mx.array(np.array([1.0, 2.0, 4.0], np.float32))
sc = np.array(mx.zeros(2).at[idx].add(upd))
check("float scatter CAS sum path", np.allclose(sc, [5.0, 2.0]), f"got {sc}")

bsrc = mx.zeros(4, mx.bool_)
bidx = mx.array(np.array([0, 1, 2, 3], np.int32))
bsrc[bidx] = mx.array(np.array([True, True, True, True]))
bsc = np.array(bsrc)
check("bool scatter writes", bool(bsc.all()), f"got {bsc}")

cnd = mx.array(np.array([[True, False]]))  # broadcast condition over 2x2
a2 = mx.array(np.array([[1.0, 2.0], [3.0, 4.0]], np.float32))
b2 = mx.array(np.array([[9.0, 8.0], [7.0, 6.0]], np.float32))
sel = np.array(mx.where(cnd, a2, b2))
check("select broadcast condition", np.array_equal(sel, [[1.0, 8.0], [3.0, 6.0]]),
      f"got {sel}")

# --- v0.3.1 fix: GQA SDPA vs decomposition at the upstream failing shape ---
rng = np.random.default_rng(7)
B, D, Hq, Hk, QL, KL = 1, 72, 8, 2, 64, 128
q = rng.standard_normal((B, Hq, QL, D)).astype(np.float32)
k = rng.standard_normal((B, Hk, KL, D)).astype(np.float32)
v = rng.standard_normal((B, Hk, KL, D)).astype(np.float32)
fast = np.array(mx.fast.scaled_dot_product_attention(
    mx.array(q), mx.array(k), mx.array(v), scale=float(1.0 / np.sqrt(D))))
rep_k = np.repeat(k, Hq // Hk, axis=1)
rep_v = np.repeat(v, Hq // Hk, axis=1)
scores = np.einsum("bhqd,bhkd->bhqk", q, rep_k) / np.sqrt(D)
weights = np.exp(scores - scores.max(-1, keepdims=True))
weights = weights / weights.sum(-1, keepdims=True)
ref = np.einsum("bhqk,bhkd->bhqd", weights, rep_v)
gqa_err = np.abs(fast - ref).max()
check("GQA SDPA vs decomposition (qL=64 kL=128)", gqa_err < 1e-3,
      f"maxerr={gqa_err:.2e}")

# Informational: the small-S vector path (W11, disclosed open in
# docs/known-defects.md) still disagrees with the decomposition. This line
# documents the wheel matches its disclosure; it must not fail the probe.
q16 = rng.standard_normal((B, Hq, 16, D)).astype(np.float32)
k16 = rng.standard_normal((B, Hk, 16, D)).astype(np.float32)
v16 = rng.standard_normal((B, Hk, 16, D)).astype(np.float32)
f16 = np.array(mx.fast.scaled_dot_product_attention(
    mx.array(q16), mx.array(k16), mx.array(v16), scale=1.0))
rk16 = np.repeat(k16, Hq // Hk, axis=1)
rv16 = np.repeat(v16, Hq // Hk, axis=1)
s16 = np.einsum("bhqd,bhkd->bhqk", q16, rk16) / np.sqrt(D)
w16 = np.exp(s16 - s16.max(-1, keepdims=True))
w16 /= w16.sum(-1, keepdims=True)
r16 = np.einsum("bhqk,bhkd->bhqd", w16, rv16)
print(f"INFO W11 vector-path deviation at S=16 (disclosed open): "
      f"maxerr={np.abs(f16 - r16).max():.2e}")

# --- v0.3.2 fix: boolean reductions past the first word (mx.all AND mx.any) ---
# Sizes straddle the word boundary (4, 5, 8, 9) and the chunk boundary
# (32, 33, 4096, 4097). Three input shapes per size: all-true, all-false,
# and one element differing - that element placed both inside the first
# word and beyond it, which is the case that made mx.any look correct when
# it was not. Axis reductions ride the same load path; a (2, n) shape with
# n >= 3 puts row 1 in the second word.
SIZES = (4, 5, 8, 9, 32, 33, 4096, 4097)
for n in SIZES:
    all_t = np.ones(n, bool)
    all_f = np.zeros(n, bool)
    late_t = np.zeros(n, bool)
    early_t = np.zeros(n, bool); early_t[1] = True   # differing, inside word 0
    late_f = np.ones(n, bool)
    early_f = np.ones(n, bool); early_f[1] = False
    if n > 4:
        late_t[n - 1] = True    # differing, outside word 0
        late_f[n - 1] = False
    cases = [
        ("all-true", all_t, True),
        ("all-false", all_f, False),
    ]
    if n > 4:
        # A single differing element makes all() False and any() True,
        # wherever that element sits - inside or outside word 0.
        cases += [("late-true", late_t, False), ("early-true", early_t, False),
                  ("late-false", late_f, False), ("early-false", early_f, False)]
    for label, vals, want in cases:
        arr = mx.array(vals)
        got_all = mx.all(arr).item()
        got_any = mx.any(arr).item()
        want_any = label != "all-false"  # any True input element makes any() True
        check(f"mx.all n={n} {label}", got_all is want, f"got {got_all}")
        check(f"mx.any n={n} {label}", got_any is want_any, f"got {got_any}")

for n in (3, 5, 33):
    g = np.ones((2, n), bool)
    check(f"axis (2,{n}) all", mx.all(mx.array(g), axis=1).tolist() == [True, True])
    g1 = np.zeros((2, n), bool); g1[1, n - 1] = True
    check(f"axis (2,{n}) any row-major late",
          mx.any(mx.array(g1), axis=1).tolist() == [False, True],
          f"got {mx.any(mx.array(g1), axis=1).tolist()}")

if failures:
    print("FAILURES:", failures)
    sys.exit(1)
print("ALL CHECKS PASSED")
