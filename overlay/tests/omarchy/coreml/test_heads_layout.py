# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the Q/K/V heads-layout rewrite."""

import unittest
from pathlib import Path

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.heads_layout import (
    HeadsLayoutError,
    numpy_heads_stack,
    numpy_naive_reshape,
    numpy_weight_permute_reshape,
    plan_heads_layout,
    require_exact,
    rewrite_heads_layout,
)

_B, _T, _H, _D, _C = 1, 375, 8, 128, 1024

_PREAMBLE = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 375, 1024]> hidden) {
    tensor<fp16, [1024, 1024]> q_w = const()[val = tensor<fp16, [1024, 1024]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(1000)))];
    tensor<fp16, [1024]> q_b = const()[val = tensor<fp16, [1024]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(2000)))];
    tensor<fp16, [1, 375, 1024]> linear_3 = linear(bias = q_b, weight = q_w, x = hidden)[name = tensor<string, []>("linear_3")];
    tensor<int32, [4]> var_331 = const()[name = tensor<string, []>("op_331"), val = tensor<int32, [4]>([1, 375, -1, 128])];
    tensor<fp16, [1, 375, 8, 128]> var_332 = reshape(shape = var_331, x = linear_3)[name = tensor<string, []>("op_332")];
    tensor<int32, [4]> query_states_1_perm_0 = const()[name = tensor<string, []>("query_states_1_perm_0"), val = tensor<int32, [4]>([0, 2, 1, 3])];
    tensor<fp16, [1, 8, 1, 128]> bias_q = const()[val = tensor<fp16, [1, 8, 1, 128]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(3000)))];
    tensor<fp16, [1, 8, 1, 128]> bias_v = const()[val = tensor<fp16, [1, 8, 1, 128]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(4000)))];
"""

_TWO_ADDS = """    tensor<fp16, [1, 8, 375, 128]> query_states_1_cast_fp16 = transpose(perm = query_states_1_perm_0, x = var_332)[name = tensor<string, []>("transpose_143")];
    tensor<fp16, [1, 8, 375, 128]> query_1 = add(x = query_states_1_cast_fp16, y = bias_q)[name = tensor<string, []>("query_1")];
    tensor<fp16, [1, 8, 375, 128]> query_states_with_bias_v_1 = add(x = query_states_1_cast_fp16, y = bias_v)[name = tensor<string, []>("query_states_with_bias_v_1")];
  } -> (query_1, query_states_with_bias_v_1);
}
"""

_NEG_PERM_KV = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 375, 1024]> hidden) {
    tensor<fp16, [1024, 1024]> k_w = const()[val = tensor<fp16, [1024, 1024]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(5000)))];
    tensor<fp16, [1024]> k_b = const()[val = tensor<fp16, [1024]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(6000)))];
    tensor<fp16, [1, 375, 1024]> linear_4 = linear(bias = k_b, weight = k_w, x = hidden)[name = tensor<string, []>("linear_4")];
    tensor<int32, [4]> var_336 = const()[name = tensor<string, []>("op_336"), val = tensor<int32, [4]>([1, 375, -1, 128])];
    tensor<fp16, [1, 375, 8, 128]> var_337 = reshape(shape = var_336, x = linear_4)[name = tensor<string, []>("op_337")];
    tensor<int32, [4]> hidden_states_23_perm_0 = const()[name = tensor<string, []>("hidden_states_23_perm_0"), val = tensor<int32, [4]>([0, 2, -3, -1])];
    tensor<bool, []> ty = const()[name = tensor<string, []>("ty"), val = tensor<bool, []>(true)];
    tensor<bool, []> tx = const()[name = tensor<string, []>("tx"), val = tensor<bool, []>(false)];
    tensor<fp16, [1, 8, 375, 128]> hidden_states_23_cast_fp16 = transpose(perm = hidden_states_23_perm_0, x = var_337)[name = tensor<string, []>("transpose_142")];
    tensor<fp16, [1, 8, 375, 375]> scores = matmul(transpose_x = tx, transpose_y = ty, x = hidden_states_23_cast_fp16, y = hidden_states_23_cast_fp16)[name = tensor<string, []>("scores")];
  } -> (scores);
}
"""


class NumpyProofTest(unittest.TestCase):
    def test_encoder_shaped_stack_is_bit_exact(self):
        rng = np.random.default_rng(20260913)
        cases = [
            rng.standard_normal((_B, _T, _H * _D)).astype(np.float16),
            np.zeros((_B, _T, _H * _D), np.float16),
            np.full((_B, _T, _H * _D), np.float16("-0")),
        ]
        cases[0][0, 0, 0] = np.float16("-0")
        for linear_out in cases:
            original, rewritten = numpy_heads_stack(linear_out, _H, _D)
            require_exact(original, rewritten, "heads stack")

    def test_naive_reshape_is_inexact(self):
        rng = np.random.default_rng(20260913)
        linear_out = rng.standard_normal((_B, _T, _H * _D)).astype(np.float16)
        original, rewritten = numpy_naive_reshape(linear_out, _H, _D)
        with self.assertRaises(HeadsLayoutError) as caught:
            require_exact(original, rewritten, "naive reshape [B,H,T,D]")
        self.assertIn("inexact naive reshape", str(caught.exception))

    def test_weight_row_permute_cannot_make_reshape_exact(self):
        rng = np.random.default_rng(20260913)
        x = rng.standard_normal((_B, 4, 8)).astype(np.float16)
        weight = rng.standard_normal((8, 8)).astype(np.float16)
        identity = np.arange(8)
        shuffled = rng.permutation(8)
        for perm in (identity, shuffled):
            original, rewritten = numpy_weight_permute_reshape(
                x, weight, perm, heads=2, head_dim=4
            )
            with self.assertRaises(HeadsLayoutError):
                require_exact(
                    original, rewritten, "weight-permute reshape [B,H,T,D]"
                )


class HeadsLayoutRewriteTest(unittest.TestCase):
    def test_rewrites_encoder_two_add_consumers(self):
        mil = _PREAMBLE + _TWO_ADDS
        plan = plan_heads_layout(mil)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].disposition, "REWRITTEN")
        self.assertEqual(plan[0].output_name, "query_states_1_cast_fp16")

        text, report = rewrite_heads_layout(mil)
        self.assertEqual(report.rewritten, ["query_states_1_cast_fp16"])
        self.assertEqual(report.refused, [])
        self.assertNotIn(
            "transpose(perm = query_states_1_perm_0", text
        )
        self.assertIn("concat(axis =", text)
        self.assertIn("x0 =", text)
        self.assertIn(
            "add(x = query_states_1_cast_fp16, y = bias_q)", text
        )
        self.assertIn(
            "add(x = query_states_1_cast_fp16, y = bias_v)", text
        )
        self.assertEqual(
            text.count("linear(bias ="), _H
        )
        self.assertNotIn("reshape(shape = var_331", text)
        self.assertEqual(
            report.to_dict()["schema"],
            "mlx-omarchy.heads-layout-rewrite.v1",
        )

    def test_rewrites_kv_negative_perm(self):
        text, report = rewrite_heads_layout(_NEG_PERM_KV)
        self.assertEqual(report.rewritten, ["hidden_states_23_cast_fp16"])
        self.assertNotIn(
            "transpose(perm = hidden_states_23_perm_0", text
        )
        self.assertIn(
            "matmul(transpose_x = tx, transpose_y = ty, "
            "x = hidden_states_23_cast_fp16",
            text,
        )

    def test_non_linear_producer_is_refused(self):
        mil = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 375, 8, 128]> var_332) {
    tensor<int32, [4]> query_states_1_perm_0 = const()[name = tensor<string, []>("query_states_1_perm_0"), val = tensor<int32, [4]>([0, 2, 1, 3])];
    tensor<fp16, [1, 8, 375, 128]> query_states_1_cast_fp16 = transpose(perm = query_states_1_perm_0, x = var_332)[name = tensor<string, []>("transpose_143")];
    tensor<fp16, [1, 8, 375, 128]> out = add(x = query_states_1_cast_fp16, y = query_states_1_cast_fp16)[name = tensor<string, []>("out")];
  } -> (out);
}
"""
        plan = plan_heads_layout(mil)
        self.assertEqual(plan[0].disposition, "REFUSED")
        self.assertIn("not a reshape", plan[0].reason)
        text, report = rewrite_heads_layout(mil)
        self.assertEqual(report.rewritten, [])
        self.assertIn("transpose(perm = query_states_1_perm_0", text)

    def test_real_encoder_mil_drops_query_states_transpose(self):
        mil_path = (
            Path(__file__).resolve().parents[4]
            / "receipts"
            / "2026-09-13-mask-layout-rewrite"
            / "model-fp16-nolayout.mil"
        )
        if not mil_path.is_file():
            self.skipTest("mask-layout encoder MIL receipt is not present")
        mil = mil_path.read_text()
        self.assertIn(
            "transpose(perm = query_states_1_perm_0, x = var_332_cast_fp16)",
            mil,
        )
        text, report = rewrite_heads_layout(mil)
        self.assertIn("query_states_1_cast_fp16", report.rewritten)
        self.assertNotIn(
            "query_states_1_cast_fp16",
            [entry.output_name for entry in report.refused],
        )
        self.assertNotIn(
            "transpose(perm = query_states_1_perm_0, x = var_332_cast_fp16)",
            text,
        )
        self.assertIn(
            "tensor<fp16, [1, 8, 375, 128]> query_states_1_cast_fp16 = concat(",
            text,
        )
        self.assertIn(
            "add(x = query_states_1_cast_fp16, y = var_345_to_fp16)", text
        )


if __name__ == "__main__":
    unittest.main()
