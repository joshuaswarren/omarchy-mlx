# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the noncontiguous slice_by_index rewrite."""

import unittest
from pathlib import Path

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.slice_layout import (
    SliceLayoutError,
    numpy_last_dim_prefix,
    numpy_last_dim_suffix,
    numpy_relpos_drop_first,
    numpy_relpos_drop_last,
    plan_slice_layout,
    require_exact,
    rewrite_slice_layout,
)

_B, _H, _T, _D = 1, 8, 750, 375

_ENCODER_SLICE = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 8, 750, 375]> attention_scores_5_cast_fp16) {
    tensor<int32, [4]> var_367_begin_0 = const()[name = tensor<string, []>("op_367_begin_0"), val = tensor<int32, [4]>([0, 0, 1, 0])];
    tensor<int32, [4]> var_367_end_0 = const()[name = tensor<string, []>("op_367_end_0"), val = tensor<int32, [4]>([1, 8, 750, 375])];
    tensor<bool, [4]> var_367_end_mask_0 = const()[name = tensor<string, []>("op_367_end_mask_0"), val = tensor<bool, [4]>([true, true, true, true])];
    tensor<fp16, [1, 8, 749, 375]> var_367_cast_fp16 = slice_by_index(begin = var_367_begin_0, end = var_367_end_0, end_mask = var_367_end_mask_0, x = attention_scores_5_cast_fp16)[name = tensor<string, []>("op_367_cast_fp16")];
    tensor<int32, [4]> var_368 = const()[name = tensor<string, []>("op_368"), val = tensor<int32, [4]>([1, 8, 375, 749])];
    tensor<fp16, [1, 8, 375, 749]> matrix_bd_1_cast_fp16 = reshape(shape = var_368, x = var_367_cast_fp16)[name = tensor<string, []>("matrix_bd_1_cast_fp16")];
  } -> (matrix_bd_1_cast_fp16);
}
"""

_LAST_DIM_SLICE = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 8, 375, 749]> matrix_bd_1_cast_fp16) {
    tensor<int32, [4]> matrix_bd_3_begin_0 = const()[name = tensor<string, []>("matrix_bd_3_begin_0"), val = tensor<int32, [4]>([0, 0, 0, 0])];
    tensor<int32, [4]> matrix_bd_3_end_0 = const()[name = tensor<string, []>("matrix_bd_3_end_0"), val = tensor<int32, [4]>([1, 8, 375, 375])];
    tensor<bool, [4]> matrix_bd_3_end_mask_0 = const()[name = tensor<string, []>("matrix_bd_3_end_mask_0"), val = tensor<bool, [4]>([true, true, true, false])];
    tensor<fp16, [1, 8, 375, 375]> matrix_bd_3_cast_fp16 = slice_by_index(begin = matrix_bd_3_begin_0, end = matrix_bd_3_end_0, end_mask = matrix_bd_3_end_mask_0, x = matrix_bd_1_cast_fp16)[name = tensor<string, []>("matrix_bd_3_cast_fp16")];
  } -> (matrix_bd_3_cast_fp16);
}
"""


class NumpyProofTest(unittest.TestCase):
    def test_encoder_shaped_drop_first_is_bit_exact(self):
        rng = np.random.default_rng(20260913)
        cases = [
            rng.standard_normal((_B, _H, _T, _D)).astype(np.float16),
            np.zeros((_B, _H, _T, _D), np.float16),
            np.full((_B, _H, _T, _D), np.float16("-0")),
        ]
        cases[0][0, 0, 0, 0] = np.float16("-0")
        cases[0][0, 7, 1, 374] = np.float16("-0")
        for source in cases:
            original, rewritten = numpy_relpos_drop_first(source)
            require_exact(original, rewritten, "rel-pos drop first")
            two_step = np.concatenate(
                [
                    source[:, head : head + 1, :, :][:, :, 1:, :]
                    for head in range(_H)
                ],
                axis=1,
            )
            require_exact(original, two_step, "rel-pos two-step")
            self.assertEqual(original.shape, (_B, _H, _T - 1, _D))

    def test_drop_last_is_inexact(self):
        rng = np.random.default_rng(20260913)
        source = rng.standard_normal((_B, _H, _T, _D)).astype(np.float16)
        original, rewritten = numpy_relpos_drop_last(source)
        with self.assertRaises(SliceLayoutError) as caught:
            require_exact(original, rewritten, "rel-pos drop last")
        self.assertIn("inexact rel-pos drop last", str(caught.exception))

    def test_encoder_shaped_last_dim_prefix_is_bit_exact(self):
        rng = np.random.default_rng(20260913)
        shape = (1, 8, 375, 749)
        keep = 375
        cases = [
            rng.standard_normal(shape).astype(np.float16),
            np.zeros(shape, np.float16),
            np.full(shape, np.float16("-0")),
        ]
        cases[0][0, 0, 0, 0] = np.float16("-0")
        cases[0][0, 7, 374, 374] = np.float16("-0")
        for source in cases:
            original, rewritten = numpy_last_dim_prefix(source, keep)
            require_exact(original, rewritten, "last-dim prefix")
            self.assertEqual(original.shape, (1, 8, 375, keep))

    def test_last_dim_suffix_is_inexact(self):
        rng = np.random.default_rng(20260913)
        source = rng.standard_normal((1, 8, 375, 749)).astype(np.float16)
        original, rewritten = numpy_last_dim_suffix(source, 375)
        with self.assertRaises(SliceLayoutError) as caught:
            require_exact(original, rewritten, "last-dim suffix")
        self.assertIn("inexact last-dim suffix", str(caught.exception))



class SliceLayoutRewriteTest(unittest.TestCase):
    def test_rewrites_encoder_relpos_slice(self):
        plan = plan_slice_layout(_ENCODER_SLICE)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].disposition, "REWRITTEN")
        self.assertEqual(plan[0].output_name, "var_367_cast_fp16")
        self.assertIn("8 chunks of 280875 spaced 281250", plan[0].reason)
        self.assertIn("axes=[1]", plan[0].reason)

        text, report = rewrite_slice_layout(_ENCODER_SLICE)
        self.assertEqual(report.rewritten, ["var_367_cast_fp16"])
        self.assertEqual(report.refused, [])
        self.assertNotIn(
            "slice_by_index(begin = var_367_begin_0, end = var_367_end_0",
            text,
        )
        self.assertIn("concat(axis =", text)
        self.assertIn("x0 =", text)
        self.assertIn("x7 =", text)
        self.assertEqual(text.count("slice_by_index("), 2 * _H)
        self.assertIn("tensor<fp16, [1, 1, 750, 375]>", text)
        self.assertIn("tensor<fp16, [1, 1, 749, 375]>", text)
        self.assertIn(
            "reshape(shape = var_368, x = var_367_cast_fp16)",
            text,
        )
        self.assertEqual(
            report.to_dict()["schema"],
            "mlx-omarchy.slice-layout-rewrite.v1",
        )

    def test_rewrites_last_dim_prefix(self):
        plan = plan_slice_layout(_LAST_DIM_SLICE)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].disposition, "REWRITTEN")
        self.assertEqual(plan[0].output_name, "matrix_bd_3_cast_fp16")
        self.assertIn("3000 chunks of 375 spaced 749", plan[0].reason)
        self.assertIn("axes=[1, 2]", plan[0].reason)

        text, report = rewrite_slice_layout(_LAST_DIM_SLICE)
        self.assertEqual(report.rewritten, ["matrix_bd_3_cast_fp16"])
        self.assertEqual(report.refused, [])
        self.assertNotIn(
            "slice_by_index(begin = matrix_bd_3_begin_0",
            text,
        )
        self.assertIn(
            "tensor<fp16, [1, 8, 375, 375]> matrix_bd_3_cast_fp16 = concat(",
            text,
        )
        self.assertIn("tensor<fp16, [1, 1, 375, 749]>", text)
        self.assertIn("tensor<fp16, [1, 1, 1, 749]>", text)
        self.assertIn("tensor<fp16, [1, 1, 1, 375]>", text)
        self.assertIn("x7 =", text)


    def test_real_encoder_mil_drops_the_749_slice(self):
        mil_path = (
            Path(__file__).resolve().parents[4]
            / "receipts"
            / "2026-09-13-heads-layout-rewrite"
            / "model-fp16-noheads.mil"
        )
        if not mil_path.is_file():
            self.skipTest("heads-layout encoder MIL receipt is not present")
        mil = mil_path.read_text()
        marker = (
            "slice_by_index(begin = var_367_begin_0, end = var_367_end_0, "
            "end_mask = var_367_end_mask_0, x = attention_scores_5_cast_fp16)"
        )
        self.assertIn(marker, mil)
        text, report = rewrite_slice_layout(mil)
        self.assertIn("var_367_cast_fp16", report.rewritten)
        self.assertNotIn(
            "var_367_cast_fp16",
            [entry.output_name for entry in report.refused],
        )
        self.assertNotIn(marker, text)
        self.assertIn(
            "tensor<fp16, [1, 8, 749, 375]> var_367_cast_fp16 = concat(",
            text,
        )
        self.assertIn(
            "reshape(shape = var_368, x = var_367_cast_fp16)",
            text,
        )
        self.assertIn("matrix_bd_3_cast_fp16", report.rewritten)
        self.assertFalse(
            any(
                entry.output_name.startswith("matrix_bd_")
                and entry.disposition == "REFUSED"
                for entry in report.refused
            )
        )


if __name__ == "__main__":
    unittest.main()
