# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the 1×1 conv-layout rewrite."""

import unittest
from pathlib import Path

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.conv_layout import (
    ConvLayoutError,
    STANDALONE_MIL,
    numpy_naive_nhwc_reshape,
    numpy_nchw_flatten_view,
    plan_conv_layout,
    require_exact,
    rewrite_conv_layout,
)

_N, _C, _H, _W = 1, 256, 6, 4

_ENCODER = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 256, 750, 32]> input_7_cast_fp16) {
    tensor<string, []> hidden_states_5_pad_type_0 = const()[name = tensor<string, []>("hidden_states_5_pad_type_0"), val = tensor<string, []>("valid")];
    tensor<int32, [2]> hidden_states_5_strides_0 = const()[name = tensor<string, []>("hidden_states_5_strides_0"), val = tensor<int32, [2]>([1, 1])];
    tensor<int32, [4]> hidden_states_5_pad_0 = const()[name = tensor<string, []>("hidden_states_5_pad_0"), val = tensor<int32, [4]>([0, 0, 0, 0])];
    tensor<int32, [2]> hidden_states_5_dilations_0 = const()[name = tensor<string, []>("hidden_states_5_dilations_0"), val = tensor<int32, [2]>([1, 1])];
    tensor<int32, []> hidden_states_5_groups_0 = const()[name = tensor<string, []>("hidden_states_5_groups_0"), val = tensor<int32, []>(1)];
    tensor<fp16, [256, 256, 1, 1]> encoder_subsampling_layers_3_weight_to_fp16 = const()[name = tensor<string, []>("encoder_subsampling_layers_3_weight_to_fp16"), val = tensor<fp16, [256, 256, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(15232)))];
    tensor<fp16, [256]> encoder_subsampling_layers_3_bias_to_fp16 = const()[name = tensor<string, []>("encoder_subsampling_layers_3_bias_to_fp16"), val = tensor<fp16, [256]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(146368)))];
    tensor<fp16, [1, 256, 750, 32]> hidden_states_5_cast_fp16 = conv(bias = encoder_subsampling_layers_3_bias_to_fp16, dilations = hidden_states_5_dilations_0, groups = hidden_states_5_groups_0, pad = hidden_states_5_pad_0, pad_type = hidden_states_5_pad_type_0, strides = hidden_states_5_strides_0, weight = encoder_subsampling_layers_3_weight_to_fp16, x = input_7_cast_fp16)[name = tensor<string, []>("hidden_states_5_cast_fp16")];
    tensor<string, []> hidden_states_1_pad_type_0 = const()[name = tensor<string, []>("hidden_states_1_pad_type_0"), val = tensor<string, []>("custom")];
    tensor<int32, [4]> hidden_states_1_pad_0 = const()[name = tensor<string, []>("hidden_states_1_pad_0"), val = tensor<int32, [4]>([1, 1, 1, 1])];
    tensor<int32, [2]> hidden_states_1_strides_0 = const()[name = tensor<string, []>("hidden_states_1_strides_0"), val = tensor<int32, [2]>([2, 2])];
    tensor<int32, [2]> hidden_states_1_dilations_0 = const()[name = tensor<string, []>("hidden_states_1_dilations_0"), val = tensor<int32, [2]>([1, 1])];
    tensor<int32, []> hidden_states_1_groups_0 = const()[name = tensor<string, []>("hidden_states_1_groups_0"), val = tensor<int32, []>(1)];
    tensor<fp16, [256, 1, 3, 3]> encoder_subsampling_layers_0_weight_to_fp16 = const()[name = tensor<string, []>("encoder_subsampling_layers_0_weight_to_fp16"), val = tensor<fp16, [256, 1, 3, 3]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(64)))];
    tensor<fp16, [256]> encoder_subsampling_layers_0_bias_to_fp16 = const()[name = tensor<string, []>("encoder_subsampling_layers_0_bias_to_fp16"), val = tensor<fp16, [256]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(4736)))];
    tensor<fp16, [1, 256, 375, 16]> hidden_states_1_cast_fp16 = conv(bias = encoder_subsampling_layers_0_bias_to_fp16, dilations = hidden_states_1_dilations_0, groups = hidden_states_1_groups_0, pad = hidden_states_1_pad_0, pad_type = hidden_states_1_pad_type_0, strides = hidden_states_1_strides_0, weight = encoder_subsampling_layers_0_weight_to_fp16, x = hidden_states_5_cast_fp16)[name = tensor<string, []>("hidden_states_1_cast_fp16")];
  } -> (hidden_states_5_cast_fp16);
}
"""

_SIBLING = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 256, 375, 16]> input_13_cast_fp16) {
    tensor<string, []> hidden_states_9_pad_type_0 = const()[name = tensor<string, []>("hidden_states_9_pad_type_0"), val = tensor<string, []>("valid")];
    tensor<int32, [2]> hidden_states_9_strides_0 = const()[name = tensor<string, []>("hidden_states_9_strides_0"), val = tensor<int32, [2]>([1, 1])];
    tensor<int32, [4]> hidden_states_9_pad_0 = const()[name = tensor<string, []>("hidden_states_9_pad_0"), val = tensor<int32, [4]>([0, 0, 0, 0])];
    tensor<int32, [2]> hidden_states_9_dilations_0 = const()[name = tensor<string, []>("hidden_states_9_dilations_0"), val = tensor<int32, [2]>([1, 1])];
    tensor<int32, []> hidden_states_9_groups_0 = const()[name = tensor<string, []>("hidden_states_9_groups_0"), val = tensor<int32, []>(1)];
    tensor<fp16, [256, 256, 1, 1]> encoder_subsampling_layers_6_weight_to_fp16 = const()[name = tensor<string, []>("encoder_subsampling_layers_6_weight_to_fp16"), val = tensor<fp16, [256, 256, 1, 1]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(153024)))];
    tensor<fp16, [256]> encoder_subsampling_layers_6_bias_to_fp16 = const()[name = tensor<string, []>("encoder_subsampling_layers_6_bias_to_fp16"), val = tensor<fp16, [256]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(284160)))];
    tensor<fp16, [1, 256, 375, 16]> hidden_states_9_cast_fp16 = conv(bias = encoder_subsampling_layers_6_bias_to_fp16, dilations = hidden_states_9_dilations_0, groups = hidden_states_9_groups_0, pad = hidden_states_9_pad_0, pad_type = hidden_states_9_pad_type_0, strides = hidden_states_9_strides_0, weight = encoder_subsampling_layers_6_weight_to_fp16, x = input_13_cast_fp16)[name = tensor<string, []>("hidden_states_9_cast_fp16")];
  } -> (hidden_states_9_cast_fp16);
}
"""

class NumpyProofTest(unittest.TestCase):
    def test_nchw_flatten_view_is_bit_exact(self):
        rng = np.random.default_rng(20260913)
        cases = [
            rng.standard_normal((_N, _C, _H, _W)).astype(np.float16),
            np.zeros((_N, _C, _H, _W), np.float16),
            np.full((_N, _C, _H, _W), np.float16("-0")),
        ]
        cases[0][0, 0, 0, 0] = np.float16("-0")
        for source in cases:
            original, rewritten = numpy_nchw_flatten_view(source)
            require_exact(original, rewritten, "NCHW flatten view")

    def test_naive_nhwc_reshape_is_inexact(self):
        rng = np.random.default_rng(20260913)
        source = rng.standard_normal((_N, _C, _H, _W)).astype(np.float16)
        weight = rng.standard_normal((_C, _C, 1, 1)).astype(np.float16)
        bias = rng.standard_normal((_C,)).astype(np.float16)
        original, rewritten = numpy_naive_nhwc_reshape(source, weight, bias)
        with self.assertRaises(ConvLayoutError) as caught:
            require_exact(original, rewritten, "naive nhwc reshape")
        self.assertIn("inexact naive nhwc reshape", str(caught.exception))


class ConvLayoutRewriteTest(unittest.TestCase):
    def test_rewrites_encoder_1x1_spelling(self):
        plan = plan_conv_layout(_ENCODER)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].disposition, "REWRITTEN")
        self.assertEqual(plan[0].output_name, "hidden_states_5_cast_fp16")

        text, report = rewrite_conv_layout(_ENCODER)
        self.assertEqual(report.rewritten, ["hidden_states_5_cast_fp16"])
        self.assertEqual(report.refused, [])
        self.assertIn(
            'string hidden_states_5_pad_type_0 = const()'
            '[name = string("hidden_states_5_pad_type_0"), '
            'val = string("valid")];',
            text,
        )
        self.assertIn(
            "int32 hidden_states_5_groups_0 = const()"
            '[name = string("hidden_states_5_groups_0"), val = int32(1)];',
            text,
        )
        self.assertNotIn(
            "tensor<int32, []> hidden_states_5_groups_0",
            text,
        )
        self.assertIn(
            "val = tensor<string, []>(\"custom\")",
            text,
        )
        self.assertIn(
            "tensor<int32, []> hidden_states_1_groups_0",
            text,
        )
        self.assertEqual(
            report.to_dict()["schema"],
            "mlx-omarchy.conv-layout-rewrite.v1",
        )
        text, report = rewrite_conv_layout(_SIBLING)
        self.assertEqual(report.rewritten, ["hidden_states_9_cast_fp16"])
        self.assertIn("val = string(\"valid\")", text)
        self.assertIn("val = int32(1)", text)

    def test_standalone_mil_is_rank4_k1_valid(self):
        self.assertIn("[1, 256, 32, 32]> x", STANDALONE_MIL)
        self.assertIn('val = string("valid")', STANDALONE_MIL)
        self.assertIn("val = int32(1)", STANDALONE_MIL)
        self.assertIn("[256, 256, 1, 1]", STANDALONE_MIL)
        self.assertNotIn("tensor<int32, []>", STANDALONE_MIL)
        self.assertNotIn("tensor<string, []>", STANDALONE_MIL)

    def test_real_encoder_mil_plans_both_1x1(self):
        mil_path = (
            Path(__file__).resolve().parents[4]
            / "receipts"
            / "2026-09-13-slice-layout-rewrite"
            / "model-fp16-noslice.mil"
        )
        if not mil_path.is_file():
            self.skipTest("slice-layout encoder MIL receipt is not present")
        mil = mil_path.read_text()
        plan = plan_conv_layout(mil)
        rewritten = [
            entry.output_name
            for entry in plan
            if entry.disposition == "REWRITTEN"
        ]
        refused = [
            entry.output_name
            for entry in plan
            if entry.disposition == "REFUSED"
        ]
        self.assertEqual(refused, [])
        self.assertIn("hidden_states_5_cast_fp16", rewritten)
        self.assertIn("hidden_states_9_cast_fp16", rewritten)
        self.assertEqual(len(rewritten), 2)
        text, report = rewrite_conv_layout(mil)
        self.assertEqual(set(report.rewritten), set(rewritten))
        self.assertIn(
            "int32 hidden_states_5_groups_0 = const()",
            text,
        )
        self.assertIn(
            "int32 hidden_states_9_groups_0 = const()",
            text,
        )


if __name__ == "__main__":
    unittest.main()
