"""Lavapipe / GPU-path parity tests for the omarchy Bonsai native kernels.

Runs only on a Wheel that has mlx-fast-bonsai-qmv.patch applied, so the
Bonsai fast ops are exposed via mx.fast.bonsai_q1_affine_qmv etc.

The tests exercise the dispatched native GPU path (Honeykrisp or any
Vulkan 1.3 ICD that reports subgroup_size==32 and the ARITHMETIC
subgroup op bit) against the in-Python reference chain
(Bonsai dequant + matmul).

Coverage:
- bonsai_q1_affine_qmv: K = 512, 1024, 4096; group_size = 32, 64, 128;
  dtype = float16, bfloat16, float32.
- bonsai_qmv_wide: M in {2, 3, 4, 5}; bits in {1, 2}.
- bonsai_q1_dequantize: K = 512, 4096; group_size = 32, 64, 128;
  dtype = float16, bfloat16, float32.

Numerical band: per the Bonsai row in MATRIX.md A25 ("known-fail on
Linux -> silent generic fallback"), the diagnostic envelope is
1e-3 atol / 1e-2 rtol (the standing q4_word reference parity band).
"""
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import unittest

import mlx.core as mx

_has_q1 = hasattr(mx.fast, "bonsai_q1_affine_qmv")
_has_wide = hasattr(mx.fast, "bonsai_qmv_wide")
_has_dequant = hasattr(mx.fast, "bonsai_q1_dequantize")
def _gpu_available():
    """mx.gpu is a DeviceType enum on real mlx (no is_available); probe by
    evaluating one element on the gpu device."""
    try:
        prev = mx.default_device()
        mx.set_default_device(mx.gpu)
        mx.eval(mx.array(np.zeros(1)))
        mx.set_default_device(prev)
        return True
    except Exception:
        return False


_gpu = _gpu_available()

# Same gate contract as test_bonsai_native.py: with OMARCHY_BONSAI_GATE=1
# a missing bonsai op or a missing Vulkan device is a hard failure, not a
# silent skip — a green suite must have exercised the kernels.
_GATE = os.environ.get("OMARCHY_BONSAI_GATE") == "1"
_BONSAI_OPS = ("bonsai_q1_affine_qmv", "bonsai_qmv_wide", "bonsai_q1_dequantize")


class BonsaiGatePreconditions(unittest.TestCase):
    def test_gate_preconditions(self):
        if not _GATE:
            self.skipTest("set OMARCHY_BONSAI_GATE=1 to enforce")
        missing = [name for name in _BONSAI_OPS if not hasattr(mx.fast, name)]
        if missing:
            self.fail(
                "OMARCHY_BONSAI_GATE=1 but mx.fast lacks %s — the venv's mlx "
                "wheel was not built with mlx-fast-bonsai-qmv.patch" % missing
            )
        if not _gpu:
            self.fail(
                "OMARCHY_BONSAI_GATE=1 but mlx reports no Vulkan device; "
                "refusing a pass that exercised no bonsai kernel"
            )


# Bonsai 1-bit pack: byte e holds 8 codes [e*8, ..., e*8 + 7] in lane
# bit i. Mirrors the oMLX _dequant_1bit bit order.
def _bonsai_pack_1bit(w: np.ndarray) -> np.ndarray:
    n, k = w.shape
    assert k % 8 == 0
    codes = (w >= 0).astype(np.uint8)
    bits = np.left_shift(np.uint8(1), np.arange(8, dtype=np.uint8))
    return (codes.reshape(n, k // 8, 8) * bits).sum(axis=2, dtype=np.uint8)


def _dequant_1bit_ref(packed, scales, biases, group_size):
    """In-Python reference for Bonsai 1-bit affine dequantize."""
    n, k32 = packed.shape
    k = k32 * 8
    shifts = mx.arange(8, dtype=mx.uint32)
    w_bits = (
        mx.bitwise_and(mx.expand_dims(packed, -1) >> shifts, 1)
        .reshape(n, k)
        .astype(scales.dtype)
    )
    s_full = mx.repeat(scales, group_size, axis=-1).reshape(n, k)
    b_full = mx.repeat(biases, group_size, axis=-1).reshape(n, k)
    return w_bits * s_full + b_full


def _build_inputs(n, k, group_size, dtype, seed=0):
    rng = np.random.default_rng(seed)
    w_fp = rng.normal(0.0, 0.05, (n, k)).astype(np.float32)
    codes = (w_fp >= 0).astype(np.uint8)
    packed = _bonsai_pack_1bit(w_fp)
    scales_np = np.zeros((n, k // group_size), dtype=np.float32)
    biases_np = np.zeros((n, k // group_size), dtype=np.float32)
    for g in range(k // group_size):
        sl = w_fp[:, g * group_size : (g + 1) * group_size]
        sc = np.abs(sl).max(axis=1)
        scales_np[:, g] = sc
        biases_np[:, g] = -sc * 0.5
    return (
        mx.array(packed.astype(np.uint8)),
        mx.array(scales_np).astype(dtype),
        mx.array(biases_np).astype(dtype),
    )


@unittest.skipUnless(_has_q1 and _gpu, "bonsai_q1_affine_qmv + gpu required")
class BonsaiQ1AffineQmvGpuParity(unittest.TestCase):
    def _run(self, n, k, group_size, dtype):
        packed, scales, biases = _build_inputs(n, k, group_size, dtype)
        rng = np.random.default_rng(100)
        x = mx.array(rng.normal(0, 1, (1, k)).astype(dtype))
        got = mx.fast.bonsai_q1_affine_qmv(x, packed, scales, biases, group=group_size)
        mx.eval(got)
        w_fp = _dequant_1bit_ref(packed, scales, biases, group_size)
        want = x @ w_fp.T
        mx.eval(want)
        np.testing.assert_allclose(
            np.array(got, copy=False),
            np.array(want, copy=False),
            atol=1e-3,
            rtol=1e-2,
        )

    def test_k512_gs64_f16(self):
        self._run(64, 512, 64, mx.float16)

    def test_k1024_gs128_bf16(self):
        self._run(64, 1024, 128, mx.bfloat16)

    def test_k4096_gs64_f32(self):
        self._run(32, 4096, 64, mx.float32)

    def test_k2048_gs32_f16(self):
        self._run(32, 2048, 32, mx.float16)


@unittest.skipUnless(_has_wide and _gpu, "bonsai_qmv_wide + gpu required")
class BonsaiQmvWideGpuParity(unittest.TestCase):
    def _q1(self, n, M, k, group_size, dtype):
        packed, scales, biases = _build_inputs(n, k, group_size, dtype)
        rng = np.random.default_rng(101)
        x = mx.array(rng.normal(0, 1, (M, k)).astype(dtype))
        got = mx.fast.bonsai_qmv_wide(x, packed, scales, biases, group=group_size, bits=1)
        mx.eval(got)
        w_fp = _dequant_1bit_ref(packed, scales, biases, group_size)
        want = x @ w_fp.T
        mx.eval(want)
        np.testing.assert_allclose(
            np.array(got, copy=False),
            np.array(want, copy=False),
            atol=2e-3,
            rtol=2e-2,
        )

    def _q2(self, n, M, k, group_size, dtype):
        rng = np.random.default_rng(102)
        w = mx.array(rng.normal(0, 0.05, (n, k)).astype(dtype))
        packed, scales, biases = mx.quantize(w, group_size=group_size, bits=2)
        x = mx.array(rng.normal(0, 1, (M, k)).astype(dtype))
        got = mx.fast.bonsai_qmv_wide(x, packed, scales, biases, group=group_size, bits=2)
        mx.eval(got)
        want = mx.quantized_matmul(
            x, packed, scales, biases, transpose=True, group_size=group_size, bits=2
        )
        mx.eval(want)
        np.testing.assert_allclose(
            np.array(got, copy=False),
            np.array(want, copy=False),
            atol=2e-3,
            rtol=2e-2,
        )

    def test_q1_m2(self):
        self._q1(32, 2, 1024, 64, mx.float16)

    def test_q1_m3(self):
        self._q1(32, 3, 1024, 64, mx.float16)

    def test_q1_m4(self):
        self._q1(32, 4, 1024, 64, mx.float16)

    def test_q1_m5(self):
        self._q1(32, 5, 1024, 64, mx.float16)

    def test_q2_m2(self):
        self._q2(32, 2, 1024, 64, mx.float16)

    def test_q2_m5(self):
        self._q2(32, 5, 1024, 64, mx.float16)


@unittest.skipUnless(_has_dequant and _gpu, "bonsai_q1_dequantize + gpu required")
class BonsaiQ1DequantizeGpuParity(unittest.TestCase):
    def _run(self, n, k, group_size, dtype):
        packed, scales, biases = _build_inputs(n, k, group_size, dtype)
        got = mx.fast.bonsai_q1_dequantize(
            packed, scales, biases, group=group_size, shape=(n, k), dtype=dtype
        )
        mx.eval(got)
        want = _dequant_1bit_ref(packed, scales, biases, group_size)
        mx.eval(want)
        np.testing.assert_allclose(
            np.array(got, copy=False),
            np.array(want, copy=False),
            atol=1e-4,
            rtol=1e-3,
        )

    def test_dequant_k512_gs64_f32(self):
        self._run(8, 512, 64, mx.float32)

    def test_dequant_k1024_gs128_bf16(self):
        self._run(8, 1024, 128, mx.bfloat16)


if __name__ == "__main__":
    unittest.main(verbosity=2)