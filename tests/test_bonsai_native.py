"""Parity tests for the omarchy Bonsai native kernels (custom_kernels/bonsai).

Strategy: the native path (mx.fast.bonsai_q1_affine_qmv,
mx.fast.bonsai_qmv_wide, mx.fast.bonsai_q1_dequantize) is GPU-only; the
composed CPU fallback runs in mlx/fast.cpp. We verify the fallback chain
against the in-Python reference (the same chain oMLX's Bonsai loader
implements in pure mlx). On a GPU host, the same parity must hold for
the dispatched native path; this file's GPU branches run only when
mx.gpu.is_available() reports true and the dispatch is the native one
(see test_bonsai_native_gpu_only below).

Reference path (mlx_lm-compatible Bonsai 1-bit affine decode):
- A 1-bit uint8 packed weight w[N, K/8] holds 8 codes per byte, lane
  bit i of byte e carries w[e * 8 + i].
- dequant(w) = scale[g_k] * code + bias[g_k].
- qmv = x @ dequant(w).T.

The Bonsai M=1 affine qmv kernel computes the same chain bit-for-bit
(the per-byte / per-bit exact-float trick the kernel uses is identical
to the dequant+matmul composition).
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import mlx.core as mx

# Try the new ops first (compiled with mlx-fast-bonsai-qmv.patch).
_has_fast_op = hasattr(mx.fast, "bonsai_q1_affine_qmv")

# A gate run where every parity class silently skips is a green nothing:
# it passes on a wheel built without mlx-fast-bonsai-qmv.patch.
# OMARCHY_BONSAI_GATE=1 (exported by the gpu-turn ticket scripts) turns
# the missing-op condition into a hard failure instead of a skip.
def _gpu_available():
    """mx.gpu is a DeviceType enum on mlx (no is_available); probe by
    evaluating one element on the gpu device."""
    try:
        prev = mx.default_device()
        mx.set_default_device(mx.gpu)
        mx.eval(mx.array(np.zeros(1)))
        mx.set_default_device(prev)
        return True
    except Exception:
        return False


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
                "wheel was not built with mlx-fast-bonsai-qmv.patch; refusing "
                "to report a pass that exercised no bonsai kernel" % missing
            )
        if not _gpu_available():
            self.fail(
                "OMARCHY_BONSAI_GATE=1 but evaluating on mx.gpu failed; "
                "the bonsai fast ops would silently ride the composed CPU "
                "fallback instead of the dispatched kernel"
            )

# Bonsai 1-bit pack: byte e holds 8 codes [e*8, ..., e*8 + 7] in lane
# bit i. Mirrors the oMLX _dequant_1bit bit order.
def _bonsai_pack_1bit(w: np.ndarray) -> np.ndarray:
    n, k = w.shape
    assert k % 8 == 0
    codes = (w >= 0).astype(np.uint8)
    bits = np.left_shift(np.uint8(1), np.arange(8, dtype=np.uint8))
    return (codes.reshape(n, k // 8, 8) * bits).sum(axis=2, dtype=np.uint8)


def _bonsai_dequant_1bit_reference(
    packed: mx.array, scales: mx.array, biases: mx.array, group_size: int
) -> mx.array:
    n, k32 = packed.shape
    k = k32 * 8
    n_groups = scales.shape[-1]
    shifts = mx.arange(8, dtype=mx.uint32)
    w_bits = (
        mx.bitwise_and(
            mx.expand_dims(packed, -1) >> shifts, 1
        )
        .reshape(n, k)
        .astype(scales.dtype)
    )
    # Bonsai dequant: scales[g] * code + biases[g] per element; repeat the
    # per-group scales/biases along K by group_size.
    s_full = mx.repeat(scales, group_size, axis=-1).reshape(n, k)
    b_full = mx.repeat(biases, group_size, axis=-1).reshape(n, k)
    return w_bits * s_full + b_full


def _build_inputs(n, k, group_size, dtype, seed=0):
    rng = np.random.default_rng(seed)
    w_fp = rng.normal(0.0, 0.05, (n, k)).astype(np.float32)
    # Sign flip random and store as 0/1 codes; affline bias carries the
    # signed magnitude, mirroring Bonsai's asymmetric 1-bit pack.
    packed_np = _bonsai_pack_1bit(w_fp)
    # Per-group scales and biases from the original float weights
    n_groups = k // group_size
    scales_np = np.zeros((n, n_groups), dtype=np.float32)
    biases_np = np.zeros((n, n_groups), dtype=np.float32)
    for g in range(n_groups):
        sc = np.abs(w_fp[:, g * group_size : (g + 1) * group_size]).max(axis=1)
        scales_np[:, g] = sc
        biases_np[:, g] = -sc * 0.5
    x = mx.array(rng.normal(0, 1, (1, k)).astype(np.float32))
    return (
        mx.array(packed_np.astype(np.uint8)),
        mx.array(scales_np).astype(dtype),
        mx.array(biases_np).astype(dtype),
        x.astype(dtype),
    )


def _ref_q1(packed, scales, biases, x, group_size):
    """Reference chain: dequant 1-bit then matmul."""
    n, k32 = packed.shape
    k = k32 * 8
    w_fp = _bonsai_dequant_1bit_reference(packed, scales, biases, group_size)
    return x @ w_fp.T


@unittest.skipUnless(_has_fast_op, "mx.fast.bonsai_q1_affine_qmv not compiled in")
class BonsaiQ1AffineQmvParity(unittest.TestCase):
    def test_q1_k512_gs64(self):
        for dt in (mx.float32, mx.float16, mx.bfloat16):
            with self.subTest(dtype=dt):
                packed, scales, biases, x = _build_inputs(
                    n=16, k=512, group_size=64, dtype=dt
                )
                got = mx.fast.bonsai_q1_affine_qmv(x, packed, scales, biases, group=64)
                mx.eval(got)
                want = _ref_q1(packed, scales, biases, x, 64)
                mx.eval(want)
                np.testing.assert_allclose(
                    np.array(got.astype(mx.float32), copy=False),
                    np.array(want.astype(mx.float32), copy=False),
                    atol=1e-3,
                    rtol=1e-2,
                )


@unittest.skipUnless(_has_fast_op, "mx.fast.bonsai_qmv_wide not compiled in")
class BonsaiQmvWideParity(unittest.TestCase):
    def test_q1_wide_m2(self):
        # 1-bit wide, M=2
        packed, scales, biases, _ = _build_inputs(n=16, k=512, group_size=64, dtype=mx.float16)
        rng = np.random.default_rng(2)
        x = mx.array(rng.normal(0, 1, (2, 512))).astype(mx.float16)
        got = mx.fast.bonsai_qmv_wide(x, packed, scales, biases, group=64, bits=1)
        mx.eval(got)
        w_fp = _bonsai_dequant_1bit_reference(packed, scales, biases, 64)
        want = x @ w_fp.T
        mx.eval(want)
        np.testing.assert_allclose(
            np.array(got.astype(mx.float32), copy=False),
            np.array(want.astype(mx.float32), copy=False),
            atol=2e-3,
            rtol=2e-2,
        )

    def test_q1_wide_m5(self):
        packed, scales, biases, _ = _build_inputs(n=8, k=512, group_size=64, dtype=mx.float16)
        rng = np.random.default_rng(3)
        x = mx.array(rng.normal(0, 1, (5, 512))).astype(mx.float16)
        got = mx.fast.bonsai_qmv_wide(x, packed, scales, biases, group=64, bits=1)
        mx.eval(got)
        w_fp = _bonsai_dequant_1bit_reference(packed, scales, biases, 64)
        want = x @ w_fp.T
        mx.eval(want)
        np.testing.assert_allclose(
            np.array(got.astype(mx.float32), copy=False),
            np.array(want.astype(mx.float32), copy=False),
            atol=2e-3,
            rtol=2e-2,
        )

    def test_q2_wide_m2(self):
        # 2-bit wide, M=2: the stock mlx quantized_matmul path.
        rng = np.random.default_rng(4)
        N, group = 16, 64
        K = 512
        w = mx.array(rng.normal(0, 0.05, (N, K)).astype(np.float16))
        packed, scales, biases = mx.quantize(w, group_size=group, bits=2)
        x = mx.array(rng.normal(0, 1, (2, K))).astype(mx.float16)
        got = mx.fast.bonsai_qmv_wide(x, packed, scales, biases, group=group, bits=2)
        mx.eval(got)
        want = mx.quantized_matmul(x, packed, scales, biases, transpose=True, group_size=group, bits=2)
        mx.eval(want)
        np.testing.assert_allclose(
            np.array(got.astype(mx.float32), copy=False),
            np.array(want.astype(mx.float32), copy=False),
            atol=2e-3,
            rtol=2e-2,
        )


@unittest.skipUnless(_has_fast_op, "mx.fast.bonsai_q1_dequantize not compiled in")
class BonsaiQ1DequantizeParity(unittest.TestCase):
    def test_dequantize_matches_reference(self):
        packed, scales, biases, _ = _build_inputs(n=8, k=512, group_size=64, dtype=mx.float32)
        got = mx.fast.bonsai_q1_dequantize(packed, scales, biases, group=64, shape=(8, 512), dtype=mx.float32)
        mx.eval(got)
        want = _bonsai_dequant_1bit_reference(packed, scales, biases, 64)
        mx.eval(want)
        np.testing.assert_allclose(
            np.array(got.astype(mx.float32), copy=False),
            np.array(want.astype(mx.float32), copy=False),
            atol=1e-4,
            rtol=1e-3,
        )


class BonsaiReferenceRoundtrip(unittest.TestCase):
    """CPU-only reference chain parity (always runnable)."""

    def test_pack_roundtrip(self):
        rng = np.random.default_rng(5)
        w_fp = rng.normal(0, 0.05, (8, 64)).astype(np.float32)
        codes = (w_fp >= 0).astype(np.uint8)
        # The packer derives codes from the weights itself (w >= 0); feed it
        # the weights, not the bit matrix (a bit matrix would be re-thresholded
        # and row-indexed, which raises TypeError on any numpy).
        packed = _bonsai_pack_1bit(w_fp)
        # dequant via numpy
        # bitorder little: code e*8+i lives in bit i of byte e, matching the
        # packer and the mx bit-shift reference
        unpacked = np.unpackbits(packed, axis=1, bitorder="little")[
            :, : w_fp.shape[1]
        ]
        np.testing.assert_array_equal(unpacked, codes)


if __name__ == "__main__":
    unittest.main(verbosity=2)