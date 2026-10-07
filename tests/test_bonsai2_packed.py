"""Bonsai-2 packed-module math tests against naive references (CPU).

Proves the two packed paths do what the pack contract says, without any
pack runtime import:

- fwht is the normalized Walsh-Hadamard transform of the sign-flipped
  activation (forward) with the sign flip applied after the transform
  (inverse), and is involutory.
- Packed linear output equals (reference transform of x) @ dequant(W).T
  within pinned tolerances (quantized matmul keeps fp32 group
  accumulators; the difference to a float32 dequant reference is
  accumulation order only).
- Packed embedding output equals the inverse transform of the gathered
  dequantized rows.

Provenance: prints the mlx build used; CPU device only (tests must run
on the development box, no Apple hardware, no network).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "serve") not in sys.path:
    sys.path.insert(0, str(REPO / "serve"))

import pytest

mx = pytest.importorskip("mlx.core")

from mlx_omarchy_bonsai2.packed import Packed, PackedError, fwht

BLOCK = 512
WIDTH = 512


def _wht(a: np.ndarray) -> np.ndarray:
    """Naive iterative Walsh-Hadamard transform on the last axis."""
    a = a.astype(np.float64).copy()
    leading = a.shape[:-1]
    n = a.shape[-1]
    h = 1
    while h < n:
        a = a.reshape(*leading, n // (2 * h), 2, h)
        a = np.concatenate(
            [
                a[..., 0:1, :] + a[..., 1:2, :],
                a[..., 0:1, :] - a[..., 1:2, :],
            ],
            axis=-2,
        )
        h *= 2
    return a.reshape(*leading, n)


class FwhtTests(unittest.TestCase):
    def test_forward_matches_naive_wht_reference(self):
        rng = np.random.default_rng(1)
        x = mx.array(rng.normal(0, 1, (3, WIDTH)).astype(np.float32))
        signs = mx.array(rng.choice([-1.0, 1.0], (WIDTH,)).astype(np.float32))
        got = np.array(fwht(x, BLOCK, signs).astype(mx.float32))
        want = _wht(np.array(x) * np.array(signs)) / np.sqrt(WIDTH)
        np.testing.assert_allclose(got, want, atol=1e-4, rtol=1e-4)

    def test_fwht_is_involutory(self):
        rng = np.random.default_rng(2)
        x = mx.array(rng.normal(0, 1, (2, WIDTH)).astype(np.float16))
        signs = mx.array(rng.choice([-1.0, 1.0], (WIDTH,)).astype(np.float32))
        twice = fwht(fwht(x, BLOCK, signs), BLOCK, signs, inverse=True)
        np.testing.assert_allclose(np.array(twice.astype(mx.float32)), np.array(x.astype(mx.float32)), atol=5e-3)

    def test_block_must_divide_width(self):
        x = mx.zeros((1, 10))
        with self.assertRaises(PackedError):
            fwht(x, 8, mx.ones(10))


class PackedLinearTests(unittest.TestCase):
    def test_linear_matches_dequant_reference(self):
        rng = np.random.default_rng(3)
        rows = 64
        w = mx.array(rng.normal(0, 0.05, (rows, WIDTH)).astype(np.float16))
        packed, scales, biases = mx.quantize(w, group_size=128, bits=2)
        signs = mx.array(rng.choice([-1.0, 1.0], (WIDTH,)).astype(np.float32))
        layer = Packed((packed, scales, biases), BLOCK, signs, embedding=False)
        x = mx.array(rng.normal(0, 1, (2, WIDTH)).astype(np.float16))
        got = np.array(layer(x).astype(mx.float32))

        xh = _wht(np.array(x.astype(mx.float32)) * np.array(signs)) / np.sqrt(WIDTH)
        deq = np.array(mx.dequantize(packed, scales, biases, group_size=128, bits=2).astype(mx.float32))
        want = xh.astype(np.float32) @ deq.T
        # Both sides use the same dequantized grid and fp16 inputs; the
        # residual is accumulation order (grouped fp32 vs global fp32).
        np.testing.assert_allclose(got, want, atol=0.02 + 0.02 * np.abs(want).max(), rtol=0.02)

    def test_untransformed_linear_skips_fwht(self):
        rng = np.random.default_rng(4)
        rows = 16
        w = mx.array(rng.normal(0, 0.05, (rows, WIDTH)).astype(np.float16))
        packed, scales, biases = mx.quantize(w, group_size=128, bits=2)
        layer = Packed((packed, scales, biases), 0, None, embedding=False)
        x = mx.array(rng.normal(0, 1, (2, WIDTH)).astype(np.float16))
        got = np.array(layer(x).astype(mx.float32))
        deq = np.array(mx.dequantize(packed, scales, biases, group_size=128, bits=2).astype(mx.float32))
        want = np.array(x.astype(mx.float32)) @ deq.T
        np.testing.assert_allclose(got, want, atol=0.02 + 0.02 * np.abs(want).max(), rtol=0.02)


class PackedEmbeddingTests(unittest.TestCase):
    def test_embedding_inverse_matches_reference(self):
        rng = np.random.default_rng(5)
        rows = 32
        w = mx.array(rng.normal(0, 0.05, (rows, WIDTH)).astype(np.float16))
        packed, scales, biases = mx.quantize(w, group_size=128, bits=2)
        signs = mx.array(rng.choice([-1.0, 1.0], (WIDTH,)).astype(np.float32))
        layer = Packed((packed, scales, biases), BLOCK, signs, embedding=True)
        ids = mx.array([0, 5, 31], mx.int32)
        got = np.array(layer(ids).astype(mx.float32))
        deq = np.array(mx.dequantize(packed, scales, biases, group_size=128, bits=2).astype(mx.float32))
        want = _wht(deq[[0, 5, 31]]) / np.sqrt(WIDTH) * np.array(signs)
        # fp16 output rounding only; the transform itself is exact in fp32.
        np.testing.assert_allclose(got, want, atol=0.02, rtol=0.02)


class PackedValidationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(6)
        w = mx.array(rng.normal(0, 0.05, (8, WIDTH)).astype(np.float16))
        self.arrays = mx.quantize(w, group_size=128, bits=2)
        self.signs = mx.array(rng.choice([-1.0, 1.0], (WIDTH,)).astype(np.float32))

    def test_rejects_unsupported_block(self):
        with self.assertRaises(PackedError):
            Packed(self.arrays, 768, self.signs, embedding=False)

    def test_rejects_non_sign_vector(self):
        bad = np.ones(WIDTH, np.float32)
        bad[7] = 0.0
        with self.assertRaises(PackedError):
            Packed(self.arrays, BLOCK, mx.array(bad), embedding=False)

    def test_rejects_sign_vector_without_block(self):
        with self.assertRaises(PackedError):
            Packed(self.arrays, 0, self.signs, embedding=False)

    def test_rejects_block_without_signs(self):
        with self.assertRaises(PackedError):
            Packed(self.arrays, BLOCK, None, embedding=False)


if __name__ == "__main__":
    print("provenance: mlx %s on %s (CPU reference)" % (mx.__version__, mx.default_device()))
    unittest.main()
