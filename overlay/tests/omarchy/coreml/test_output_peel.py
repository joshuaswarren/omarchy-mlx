# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the fp16-output boundary peel."""

import unittest

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.output_peel import OutputPeelError, peel_fp16_outputs

_PREAMBLE = """program(1)
{
  func main<CoreML8>(tensor<fp32, [1, 4]> features) {
    tensor<string, []> to_fp32 = const()[name = tensor<string, []>("d0"), val = tensor<string, []>("fp32")];
    tensor<string, []> to_i32 = const()[name = tensor<string, []>("d1"), val = tensor<string, []>("int32")];
    tensor<fp16, [1, 4]> hidden_fp16 = add(x = features, y = features)[name = tensor<string, []>("op_add")];
    tensor<bool, [1, 4]> mask_bool = less(x = features, y = features)[name = tensor<string, []>("op_less")];
"""

_FOOTER_BOTH = """    tensor<fp32, [1, 4]> hidden = cast(dtype = to_fp32, x = hidden_fp16)[name = tensor<string, []>("cast_0")];
    tensor<int32, [1, 4]> mask = cast(dtype = to_i32, x = mask_bool)[name = tensor<string, []>("cast_1")];
  } -> (hidden, mask);
}
"""


class OutputPeelTest(unittest.TestCase):
    def test_peels_fp16_to_fp32_and_bool_to_int32(self):
        peeled, epilogue = peel_fp16_outputs(_PREAMBLE + _FOOTER_BOTH)
        self.assertNotIn("cast_0", peeled)
        self.assertNotIn("cast_1", peeled)
        self.assertIn("} -> (hidden_fp16, mask_bool);", peeled)
        # Every other line is byte-identical.
        self.assertEqual(
            len(peeled.splitlines()),
            len((_PREAMBLE + _FOOTER_BOTH).splitlines()) - 2,
        )
        for kept, original in zip(
            peeled.splitlines(), (_PREAMBLE + _FOOTER_BOTH).splitlines()
        ):
            if kept.startswith("  } -> ("):
                continue
            self.assertIn(kept, (_PREAMBLE + _FOOTER_BOTH).splitlines())
        self.assertEqual(epilogue["schema"], "mlx-omarchy.gpu-boundary-epilogue.v1")
        self.assertEqual(
            [cast["output"] for cast in epilogue["casts"]],
            ["hidden", "mask"],
        )
        self.assertEqual(
            [(cast["input"], cast["input_dtype"]) for cast in epilogue["casts"]],
            [("hidden_fp16", "fp16"), ("mask_bool", "bool")],
        )

    def test_refuses_narrowing_cast(self):
        mil = (
            _PREAMBLE
            + '    tensor<string, []> to_fp16 = const()[name = tensor<string, []>("d2"), val = tensor<string, []>("fp16")];\n'
            + '    tensor<fp16, [1, 4]> narrow = cast(dtype = to_fp16, x = features)[name = tensor<string, []>("cast_0")];\n'
            + "  } -> (narrow);\n}\n"
        )
        with self.assertRaisesRegex(OutputPeelError, "not an exact widening"):
            peel_fp16_outputs(mil)

    def test_non_cast_return_is_left_alone_but_still_reported(self):
        mil = (
            _PREAMBLE
            + "  } -> (hidden_fp16, mask_bool);\n}\n"
        )
        with self.assertRaisesRegex(OutputPeelError, "nothing to peel"):
            peel_fp16_outputs(mil)

    def test_missing_footer_is_named(self):
        with self.assertRaisesRegex(OutputPeelError, "no function footer"):
            peel_fp16_outputs(_PREAMBLE)


if __name__ == "__main__":
    unittest.main()
