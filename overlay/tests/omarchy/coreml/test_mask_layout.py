# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the mask-chain tail-swap layout rewrite."""

import unittest
from pathlib import Path

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.mask_layout import (
    MaskLayoutError,
    numpy_tail_swap_mul,
    plan_mask_layout,
    rewrite_mask_layout,
)

_N = 375

_PREAMBLE = """program(1)
{
  func main<CoreML8>(tensor<bool, [1, 375]> output_mask) {
    tensor<int32, [1]> var_285_axes_0 = const()[name = tensor<string, []>("op_285_axes_0"), val = tensor<int32, [1]>(1)];
    tensor<bool, [1, 1, 375]> var_285 = expand_dims(axes = var_285_axes_0, x = output_mask)[name = tensor<string, []>("op_285")];
    tensor<int32, [3]> attention_mask_3_reps_0 = const()[name = tensor<string, []>("attention_mask_3_reps_0"), val = tensor<int32, [3]>([1, 375, 1])];
    tensor<bool, [1, 375, 375]> attention_mask_3 = tile(reps = attention_mask_3_reps_0, x = var_285)[name = tensor<string, []>("attention_mask_3")];
    tensor<int32, [3]> var_289_perm_0 = const()[name = tensor<string, []>("op_289_perm_0"), val = tensor<int32, [3]>([0, 2, 1])];
"""

_AND_CONSUMER = """    tensor<bool, [1, 375, 375]> var_289 = transpose(perm = var_289_perm_0, x = attention_mask_3)[name = tensor<string, []>("transpose_144")];
    tensor<bool, [1, 375, 375]> attention_mask_5 = logical_and(x = attention_mask_3, y = var_289)[name = tensor<string, []>("attention_mask_5")];
  } -> (attention_mask_5);
}
"""

_MUL_CONSUMER = """    tensor<fp16, [1, 375, 375]> var_289 = transpose(perm = var_289_perm_0, x = attention_mask_3)[name = tensor<string, []>("transpose_144")];
    tensor<fp16, [1, 375, 375]> attention_mask_5 = mul(x = attention_mask_3, y = var_289)[name = tensor<string, []>("attention_mask_5")];
  } -> (attention_mask_5);
}
"""


def _prefix_mask(length: int, n: int = _N) -> np.ndarray:
    return (np.arange(n) < length).astype(np.float16).reshape(1, n)


class NumpyProofTest(unittest.TestCase):
    def test_encoder_shaped_mul_is_bit_exact(self):
        rng = np.random.default_rng(20260913)
        cases = [
            _prefix_mask(0),
            _prefix_mask(1),
            _prefix_mask(200),
            _prefix_mask(374),
            _prefix_mask(375),
            rng.integers(0, 2, size=(1, _N)).astype(np.float16),
        ]
        for mask in cases:
            original, rewritten = numpy_tail_swap_mul(mask)
            if not np.array_equal(
                original.view("<u2"), rewritten.view("<u2")
            ):
                mismatch = np.argwhere(
                    original.view("<u2") != rewritten.view("<u2")
                )
                i, j, k = (int(v) for v in mismatch[0])
                raise MaskLayoutError(
                    f"inexact at [{i},{j},{k}]: original="
                    f"{original[i, j, k]!r} rewritten="
                    f"{rewritten[i, j, k]!r} mask_j={mask[i, k]!r} "
                    f"mask_i={mask[i, j]!r}"
                )

    def test_bool_and_matches_mul_of_zeros_and_ones(self):
        mask = _prefix_mask(200).astype(bool)
        original, rewritten = numpy_tail_swap_mul(mask)
        self.assertTrue(np.array_equal(original, rewritten))
        col = np.tile(np.expand_dims(mask, 1), (1, _N, 1))
        trans = np.transpose(col, (0, 2, 1))
        self.assertTrue(np.array_equal(original, np.logical_and(col, trans)))


class MaskLayoutRewriteTest(unittest.TestCase):
    def test_rewrites_encoder_and_consumer(self):
        mil = _PREAMBLE + _AND_CONSUMER
        plan = plan_mask_layout(mil)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].disposition, "REWRITTEN")
        self.assertEqual(plan[0].output_name, "var_289")

        text, report = rewrite_mask_layout(mil)
        self.assertEqual(report.rewritten, ["var_289"])
        self.assertEqual(report.refused, [])
        self.assertNotIn("transpose(perm = var_289_perm_0", text)
        self.assertIn(
            "tensor<bool, [1, 375, 375]> var_289 = tile(", text
        )
        self.assertIn("x = output_mask", text)
        self.assertIn(
            "logical_and(x = attention_mask_3, y = var_289)", text
        )
        self.assertEqual(
            report.to_dict()["schema"],
            "mlx-omarchy.mask-layout-rewrite.v1",
        )

    def test_rewrites_mul_consumer(self):
        # Compiler folds logical_and to mul; attempt-3 stderr names mul.
        mil = _PREAMBLE.replace("tensor<bool, [1, 1, 375]>",
                                "tensor<fp16, [1, 1, 375]>")
        mil = mil.replace(
            "tensor<bool, [1, 375, 375]> attention_mask_3",
            "tensor<fp16, [1, 375, 375]> attention_mask_3",
        )
        mil = mil.replace("tensor<bool, [1, 375]> output_mask",
                          "tensor<fp16, [1, 375]> output_mask")
        mil = mil + _MUL_CONSUMER
        text, report = rewrite_mask_layout(mil)
        self.assertEqual(report.rewritten, ["var_289"])
        self.assertNotIn("transpose(perm = var_289_perm_0", text)
        self.assertIn("mul(x = attention_mask_3, y = var_289)", text)

    def test_matmul_consumer_is_refused_by_name(self):
        mil = (
            _PREAMBLE
            + """    tensor<bool, [1, 375, 375]> var_289 = transpose(perm = var_289_perm_0, x = attention_mask_3)[name = tensor<string, []>("transpose_144")];
    tensor<bool, [1, 375, 375]> out = matmul(x = attention_mask_3, y = var_289)[name = tensor<string, []>("op_mm")];
  } -> (out);
}
"""
        )
        plan = plan_mask_layout(mil)
        self.assertEqual(plan[0].disposition, "REFUSED")
        self.assertIn("matmul", plan[0].reason)
        text, report = rewrite_mask_layout(mil)
        self.assertEqual(report.rewritten, [])
        self.assertIn("transpose(perm = var_289_perm_0", text)

    def test_non_tile_producer_is_refused(self):
        mil = """program(1)
{
  func main<CoreML8>(tensor<bool, [1, 375, 375]> attention_mask_3) {
    tensor<int32, [3]> var_289_perm_0 = const()[name = tensor<string, []>("op_289_perm_0"), val = tensor<int32, [3]>([0, 2, 1])];
    tensor<bool, [1, 375, 375]> var_289 = transpose(perm = var_289_perm_0, x = attention_mask_3)[name = tensor<string, []>("transpose_144")];
    tensor<bool, [1, 375, 375]> attention_mask_5 = mul(x = attention_mask_3, y = var_289)[name = tensor<string, []>("attention_mask_5")];
  } -> (attention_mask_5);
}
"""
        plan = plan_mask_layout(mil)
        self.assertEqual(plan[0].disposition, "REFUSED")
        self.assertIn("not a tile", plan[0].reason)

    def test_real_encoder_mil_drops_the_tail_swap(self):
        mil_path = (
            Path(__file__).resolve().parents[4]
            / "receipts"
            / "2026-09-13-int32-view-fold"
            / "model-fp16-noview.mil"
        )
        if not mil_path.is_file():
            self.skipTest("folded encoder MIL receipt is not present")
        mil = mil_path.read_text()
        self.assertIn(
            "transpose(perm = var_289_perm_0, x = attention_mask_3)", mil
        )
        text, report = rewrite_mask_layout(mil)
        self.assertEqual(report.rewritten, ["var_289"])
        self.assertEqual(report.refused, [])
        self.assertNotIn(
            "transpose(perm = var_289_perm_0, x = attention_mask_3)", text
        )
        self.assertIn(
            "tensor<bool, [1, 375, 375]> var_289 = tile(", text
        )
        self.assertIn(
            "logical_and(x = attention_mask_3, y = var_289)", text
        )


if __name__ == "__main__":
    unittest.main()
