#!/usr/bin/env python3
"""Bit-exactness & dispatch-count verification for MLX_OMARCHY_QMM_RESIDUAL_NORM."""
import ctypes
import os
import sys

import mlx.core as mx


class TraceSnapshot(ctypes.Structure):
    _fields_ = [
        ("gpu_primitive_dispatches", ctypes.c_uint64),
        ("vk_submissions", ctypes.c_uint64),
        ("vk_buffer_copies", ctypes.c_uint64),
        ("vk_buffer_fills", ctypes.c_uint64),
        ("vk_compute_dispatches", ctypes.c_uint64),
        ("omarchy_finalize_calls", ctypes.c_uint64),
        ("commit_calls_with_work", ctypes.c_uint64),
        ("commit_calls_noop", ctypes.c_uint64),
        ("barriers_emitted", ctypes.c_uint64),
        ("barriers_skipped", ctypes.c_uint64),
    ]


def get_trace_fn():
    lib_path = os.path.join(os.path.dirname(mx.__file__), "lib", "libmlx.so")
    if not os.path.exists(lib_path):
        lib_path = ctypes.util.find_library("mlx")
    lib = ctypes.CDLL(lib_path)
    fn = lib.mlx_omarchy_trace_snapshot
    fn.argtypes = [ctypes.POINTER(TraceSnapshot)]
    fn.restype = None
    return fn


def snap(fn):
    s = TraceSnapshot()
    fn(ctypes.byref(s))
    return s


def make_quantized(k, n, seed):
    rng = mx.random.key(seed)
    k1, _ = mx.random.split(rng)
    w_f = (mx.random.normal((n, k), dtype=mx.bfloat16, key=k1) * 0.05).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w_f, group_size=64, bits=4)
    mx.eval(wq, scales, biases)
    return wq, scales, biases


def run_plain_case(trace_fn, k, n, scalar_weight=False, iters=10):
    wq, scales, biases = make_quantized(k, n, seed=100 + n + k)
    rng = mx.random.key(42)
    x = (mx.random.normal((1, k), dtype=mx.bfloat16, key=rng) * 0.1).astype(mx.bfloat16)
    resid = (mx.random.normal((1, n), dtype=mx.bfloat16, key=rng) * 0.1).astype(mx.bfloat16)
    other = (mx.random.normal((1, n), dtype=mx.bfloat16, key=rng) * 0.1).astype(mx.bfloat16)
    nw = (
        None
        if scalar_weight
        else (1.0 + 0.1 * mx.random.normal((n,), dtype=mx.bfloat16, key=rng)).astype(mx.bfloat16)
    )
    mx.eval(x, resid, other) if nw is None else mx.eval(x, resid, other, nw)

    def eval_once():
        q = mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=4)
        h = resid + q
        hn = mx.fast.rms_norm(h, nw, 1e-6)
        h2 = h + other
        mx.eval(hn, h2)
        return hn, h2

    os.environ["MLX_OMARCHY_QMM_RESIDUAL_NORM"] = "0"
    ref_hn, ref_h2 = eval_once()
    s0 = snap(trace_fn)
    ref_hn, ref_h2 = eval_once()
    s1 = snap(trace_fn)
    off_dispatches = s1.vk_compute_dispatches - s0.vk_compute_dispatches

    os.environ["MLX_OMARCHY_QMM_RESIDUAL_NORM"] = "1"
    for it in range(iters):
        if it == iters - 1:
            t0 = snap(trace_fn)
        test_hn, test_h2 = eval_once()
        if it == iters - 1:
            t1 = snap(trace_fn)
            on_dispatches = t1.vk_compute_dispatches - t0.vk_compute_dispatches
        assert mx.array_equal(test_hn, ref_hn), f"plain hn mismatch (k={k}, n={n}, sw={scalar_weight}, iter={it})"
        assert mx.array_equal(test_h2, ref_h2), f"plain h2 mismatch (k={k}, n={n}, sw={scalar_weight}, iter={it})"

    print(f"PASS plain k={k} n={n} scalar_w={scalar_weight} dispatches {off_dispatches} -> {on_dispatches}")
    return off_dispatches, on_dispatches


def run_outgate_case(trace_fn, k, n, iters=10):
    wq, scales, biases = make_quantized(k, n, seed=300 + n + k)
    rng = mx.random.key(99)
    out = (mx.random.normal((1, k), dtype=mx.bfloat16, key=rng) * 0.1).astype(mx.bfloat16)
    gate = (mx.random.normal((1, k), dtype=mx.bfloat16, key=rng) * 0.5).astype(mx.bfloat16)
    resid = (mx.random.normal((1, n), dtype=mx.bfloat16, key=rng) * 0.1).astype(mx.bfloat16)
    other = (mx.random.normal((1, n), dtype=mx.bfloat16, key=rng) * 0.1).astype(mx.bfloat16)
    nw = (1.0 + 0.1 * mx.random.normal((n,), dtype=mx.bfloat16, key=rng)).astype(mx.bfloat16)
    mx.eval(out, gate, resid, other, nw)

    def eval_once():
        x_g = out * mx.sigmoid(gate)
        q = mx.quantized_matmul(x_g, wq, scales, biases, transpose=True, group_size=64, bits=4)
        h = resid + q
        hn = mx.fast.rms_norm(h, nw, 1e-6)
        h2 = h + other
        mx.eval(hn, h2)
        return hn, h2

    os.environ["MLX_OMARCHY_QMM_RESIDUAL_NORM"] = "0"
    ref_hn, ref_h2 = eval_once()
    s0 = snap(trace_fn)
    ref_hn, ref_h2 = eval_once()
    s1 = snap(trace_fn)
    off_dispatches = s1.vk_compute_dispatches - s0.vk_compute_dispatches

    os.environ["MLX_OMARCHY_QMM_RESIDUAL_NORM"] = "1"
    for it in range(iters):
        if it == iters - 1:
            t0 = snap(trace_fn)
        test_hn, test_h2 = eval_once()
        if it == iters - 1:
            t1 = snap(trace_fn)
            on_dispatches = t1.vk_compute_dispatches - t0.vk_compute_dispatches
        assert mx.array_equal(test_hn, ref_hn), f"outgate hn mismatch (k={k}, n={n}, iter={it})"
        assert mx.array_equal(test_h2, ref_h2), f"outgate h2 mismatch (k={k}, n={n}, iter={it})"

    print(f"PASS outgate k={k} n={n} dispatches {off_dispatches} -> {on_dispatches}")
    return off_dispatches, on_dispatches


def run_fallback_checks(trace_fn):
    # Test M=2 and NORM_APPLE=0 clean fallback (bit-exact)
    k, n = 2048, 2048
    wq, scales, biases = make_quantized(k, n, seed=555)
    rng = mx.random.key(77)
    x2 = (mx.random.normal((2, k), dtype=mx.bfloat16, key=rng) * 0.1).astype(mx.bfloat16)
    r2 = (mx.random.normal((2, n), dtype=mx.bfloat16, key=rng) * 0.1).astype(mx.bfloat16)
    x1 = x2[:1]
    r1 = r2[:1]
    nw = (1.0 + 0.1 * mx.random.normal((n,), dtype=mx.bfloat16, key=rng)).astype(mx.bfloat16)
    mx.eval(x2, r2, x1, r1, nw)

    os.environ["MLX_OMARCHY_QMM_RESIDUAL_NORM"] = "0"
    ref_m2 = mx.fast.rms_norm(r2 + mx.quantized_matmul(x2, wq, scales, biases, transpose=True, group_size=64, bits=4), nw, 1e-6)
    mx.eval(ref_m2)
    os.environ["MLX_OMARCHY_QMM_RESIDUAL_NORM"] = "1"
    test_m2 = mx.fast.rms_norm(r2 + mx.quantized_matmul(x2, wq, scales, biases, transpose=True, group_size=64, bits=4), nw, 1e-6)
    mx.eval(test_m2)
    assert mx.array_equal(test_m2, ref_m2), "M=2 fallback mismatch"

    os.environ["MLX_OMARCHY_NORM_APPLE"] = "0"
    os.environ["MLX_OMARCHY_QMM_RESIDUAL_NORM"] = "0"
    ref_noa = mx.fast.rms_norm(r1 + mx.quantized_matmul(x1, wq, scales, biases, transpose=True, group_size=64, bits=4), nw, 1e-6)
    mx.eval(ref_noa)
    os.environ["MLX_OMARCHY_QMM_RESIDUAL_NORM"] = "1"
    test_noa = mx.fast.rms_norm(r1 + mx.quantized_matmul(x1, wq, scales, biases, transpose=True, group_size=64, bits=4), nw, 1e-6)
    mx.eval(test_noa)
    assert mx.array_equal(test_noa, ref_noa), "NORM_APPLE=0 fallback mismatch"
    del os.environ["MLX_OMARCHY_NORM_APPLE"]
    print("PASS fallbacks (M=2, NORM_APPLE=0)")


def main():
    fn = get_trace_fn()
    d0, d1 = run_plain_case(fn, 2048, 2048)
    assert d1 == d0 - 1, f"expected -1 dispatch for plain 2048x2048, got {d0} -> {d1}"
    d0, d1 = run_plain_case(fn, 6144, 2048)
    assert d1 == d0 - 1, f"expected -1 dispatch for plain 6144x2048, got {d0} -> {d1}"
    run_plain_case(fn, 2048, 2048, scalar_weight=True)
    run_plain_case(fn, 2048, 2560)
    d0, d1 = run_outgate_case(fn, 2048, 2048)
    assert d1 == d0 - 1, f"expected -1 dispatch for outgate 2048x2048, got {d0} -> {d1}"
    run_outgate_case(fn, 4096, 2048)
    run_fallback_checks(fn)
    print("ALL_QMM_RESNORM_CHECKS_PASSED")


if __name__ == "__main__":
    main()
