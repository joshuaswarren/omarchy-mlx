# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the encoder QK attention-layout rewrite."""

import unittest
from pathlib import Path

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.attention_layout import (
    AttentionLayoutError,
    STANDALONE_MIL,
    numpy_k_naive_reshape,
    numpy_k_token_stack,
    numpy_qk_contraction,
    plan_attention_layout,
    require_exact,
    rewrite_attention_layout,
)

_B, _H, _T, _D = 1, 8, 375, 128

_BORN = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 2, 8]> hidden, tensor<fp16, [1, 2, 2, 4]> query) {
    tensor<fp16, [4, 8]> k0_w = const()[val = tensor<fp16, [4, 8]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(0)))];
    tensor<fp16, [4]> k0_b = const()[val = tensor<fp16, [4]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(100)))];
    tensor<fp16, [1, 2, 4]> k0_lin = linear(bias = k0_b, weight = k0_w, x = hidden)[name = tensor<string, []>("k0_lin")];
    tensor<int32, [1]> k_axes = const()[name = tensor<string, []>("k_axes"), val = tensor<int32, [1]>(1)];
    tensor<fp16, [1, 1, 2, 4]> k0_exp = expand_dims(axes = k_axes, x = k0_lin)[name = tensor<string, []>("k0_exp")];
    tensor<fp16, [4, 8]> k1_w = const()[val = tensor<fp16, [4, 8]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(200)))];
    tensor<fp16, [4]> k1_b = const()[val = tensor<fp16, [4]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(300)))];
    tensor<fp16, [1, 2, 4]> k1_lin = linear(bias = k1_b, weight = k1_w, x = hidden)[name = tensor<string, []>("k1_lin")];
    tensor<fp16, [1, 1, 2, 4]> k1_exp = expand_dims(axes = k_axes, x = k1_lin)[name = tensor<string, []>("k1_exp")];
    tensor<int32, []> k_axis = const()[name = tensor<string, []>("k_axis"), val = tensor<int32, []>(1)];
    tensor<fp16, [1, 2, 2, 4]> keys = concat(axis = k_axis, x0 = k0_exp, x1 = k1_exp)[name = tensor<string, []>("keys_heads_concat")];
    tensor<bool, []> tx = const()[name = tensor<string, []>("tx"), val = tensor<bool, []>(false)];
    tensor<bool, []> ty = const()[name = tensor<string, []>("ty"), val = tensor<bool, []>(true)];
    tensor<fp16, [1, 2, 2, 2]> scores = matmul(transpose_x = tx, transpose_y = ty, x = query, y = keys)[name = tensor<string, []>("scores")];
  } -> (scores);
}
"""

_CONVERT = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 2, 3, 4]> query, tensor<fp16, [1, 2, 3, 4]> keys) {
    tensor<bool, []> tx = const()[name = tensor<string, []>("tx"), val = tensor<bool, []>(false)];
    tensor<bool, []> ty = const()[name = tensor<string, []>("ty"), val = tensor<bool, []>(true)];
    tensor<fp16, [1, 2, 3, 3]> scores = matmul(transpose_x = tx, transpose_y = ty, x = query, y = keys)[name = tensor<string, []>("scores")];
  } -> (scores);
}
"""

_PV = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 8, 375, 375]> weights, tensor<fp16, [1, 8, 375, 128]> value) {
    tensor<bool, []> tx = const()[name = tensor<string, []>("tx"), val = tensor<bool, []>(false)];
    tensor<bool, []> ty = const()[name = tensor<string, []>("ty"), val = tensor<bool, []>(false)];
    tensor<fp16, [1, 8, 375, 128]> out = matmul(transpose_x = tx, transpose_y = ty, x = weights, y = value)[name = tensor<string, []>("attn_output")];
  } -> (out);
}
"""


class NumpyProofTest(unittest.TestCase):
    def test_encoder_shaped_token_stack_is_bit_exact(self):
        rng = np.random.default_rng(20260913)
        cases = [
            rng.standard_normal((_B, _H, _T, _D)).astype(np.float16),
            np.zeros((_B, _H, _T, _D), np.float16),
            np.full((_B, _H, _T, _D), np.float16("-0")),
        ]
        cases[0][0, 0, 0, 0] = np.float16("-0")
        cases[0][0, 7, 374, 127] = np.float16("-0")
        for keys in cases:
            original, rewritten = numpy_k_token_stack(keys)
            require_exact(original, rewritten, "K token stack")
            self.assertEqual(original.shape, (_B, _H, _D, _T))

    def test_naive_reshape_is_inexact(self):
        rng = np.random.default_rng(20260913)
        keys = rng.standard_normal((_B, _H, _T, _D)).astype(np.float16)
        original, rewritten = numpy_k_naive_reshape(keys)
        with self.assertRaises(AttentionLayoutError) as caught:
            require_exact(original, rewritten, "naive reshape [B,H,D,T]")
        self.assertIn("inexact naive reshape", str(caught.exception))

    def test_qk_contraction_is_bit_exact(self):
        rng = np.random.default_rng(20260913)
        query = rng.standard_normal((1, 2, 3, 4)).astype(np.float16)
        keys = rng.standard_normal((1, 2, 3, 4)).astype(np.float16)
        query[0, 0, 0, 0] = np.float16("-0")
        original, rewritten = numpy_qk_contraction(query, keys)
        require_exact(original, rewritten, "QK contraction")
        self.assertEqual(original.shape, (1, 2, 3, 3))


class AttentionLayoutRewriteTest(unittest.TestCase):
    def test_births_k_from_per_head_linears(self):
        plan = plan_attention_layout(_BORN)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].disposition, "REWRITTEN")
        self.assertEqual(plan[0].output_name, "scores")

        text, report = rewrite_attention_layout(_BORN)
        self.assertEqual(report.rewritten, ["scores"])
        self.assertEqual(report.refused, [])
        self.assertIn("val = tensor<bool, []>(false)", text)
        self.assertIn(
            "tensor<fp16, [1, 2, 4, 2]> keys = concat(",
            text,
        )
        self.assertIn(
            "tensor<fp16, [1, 1, 4, 2]> k0_exp = expand_dims(",
            text,
        )
        self.assertNotIn(
            "tensor<fp16, [1, 1, 2, 4]> k0_exp = expand_dims(",
            text,
        )
        self.assertIn("slice_by_index(", text)
        self.assertIn("y = keys)", text)
        self.assertIn(
            "matmul(transpose_x = tx, transpose_y = scores_ty0, x = query, y = keys)",
            text,
        )
        self.assertEqual(
            report.to_dict()["schema"],
            "mlx-omarchy.attention-layout-rewrite.v1",
        )

    def test_converts_explicit_k_without_mutating_q(self):
        text, report = rewrite_attention_layout(_CONVERT)
        self.assertEqual(report.rewritten, ["scores"])
        self.assertIn("scores_k_dt", text)
        self.assertIn(
            "tensor<fp16, [1, 2, 4, 3]> scores_k_dt = concat(",
            text,
        )
        self.assertIn("y = scores_k_dt)", text)
        self.assertIn("x = query, y = scores_k_dt)", text)

    def test_pv_ty_false_is_not_a_candidate(self):
        plan = plan_attention_layout(_PV)
        self.assertEqual(plan, [])
        text, report = rewrite_attention_layout(_PV)
        self.assertEqual(report.rewritten, [])
        self.assertIn("transpose_y = ty, x = weights, y = value)", text)

    def test_standalone_mil_is_the_one_program_form(self):
        self.assertIn("[1, 8, 375, 128]> x", STANDALONE_MIL)
        self.assertIn("[1, 8, 128, 375]> w", STANDALONE_MIL)
        self.assertIn("val = bool(false)", STANDALONE_MIL)
        self.assertIn("transpose_x = tx, transpose_y = ty, x = x, y = w", STANDALONE_MIL)
        self.assertNotIn("val = bool(true)", STANDALONE_MIL)

    def test_real_encoder_mil_plans_score_matmuls(self):
        mil_path = (
            Path(__file__).resolve().parents[4]
            / "receipts"
            / "2026-09-13-slice-layout-rewrite"
            / "model-fp16-noslice.mil"
        )
        if not mil_path.is_file():
            self.skipTest("slice-layout encoder MIL receipt is not present")
        mil = mil_path.read_text()
        self.assertIn(
            "matmul(transpose_x = matmul_0_transpose_x_0, "
            "transpose_y = matmul_0_transpose_y_0, "
            "x = mul_0_cast_fp16, y = hidden_states_23_cast_fp16)",
            mil,
        )
        plan = plan_attention_layout(mil)
        rewritten = [entry.output_name for entry in plan if entry.disposition == "REWRITTEN"]
        refused = [entry.output_name for entry in plan if entry.disposition == "REFUSED"]
        self.assertEqual(refused, [])
        self.assertIn("matmul_0_cast_fp16", rewritten)
        self.assertEqual(len(rewritten), 24)


if __name__ == "__main__":
    unittest.main()
