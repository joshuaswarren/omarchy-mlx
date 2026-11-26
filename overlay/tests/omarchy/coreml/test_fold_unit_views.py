# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the unit-view fold (int32 shape aliases)."""

import unittest

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.fold_unit_views import (
    _broadcast,
    fold_unit_views,
    plan_unit_folds,
)

_PREAMBLE = """program(1)
{
  func main<CoreML8>(tensor<int32, [1]> length) {
    tensor<int32, [375]> arange = const()[name = tensor<string, []>("op_arange"), val = tensor<int32, [375]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64(100)))];
    tensor<int32, [1]> axes = const()[name = tensor<string, []>("op_axes"), val = tensor<int32, [1]>(1)];
"""

_FOOTER = """    tensor<bool, [1, 375]> mask = less(x = arange, y = expanded)[name = tensor<string, []>("op_less")];
  } -> (mask);
}
"""

_FIXTURE = (
    _PREAMBLE
    + """    tensor<int32, [1, 1]> expanded = expand_dims(axes = axes, x = length)[name = tensor<string, []>("op_expand")];
"""
    + _FOOTER
)


class BroadcastTest(unittest.TestCase):
    def test_numpy_rules(self):
        self.assertEqual(_broadcast([(375,), (1,)]), (375,))
        self.assertEqual(_broadcast([(375,), (1, 1)]), (1, 375))
        self.assertEqual(_broadcast([(1, 375), (1,)]), (1, 375))
        self.assertIsNone(_broadcast([(3,), (4,)]))
        self.assertEqual(_broadcast([(), (4,)]), (4,))


class UnitFoldTest(unittest.TestCase):
    def test_absorbs_unit_expand_into_less(self):
        plan = plan_unit_folds(_FIXTURE)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].disposition, "FOLDED")
        self.assertEqual(plan[0].output_name, "expanded")

        folded, report = fold_unit_views(_FIXTURE)
        self.assertEqual(report.folded, ["expanded"])
        self.assertEqual(report.refused, [])
        # The view is gone; the consumer reads the unexpanded tensor and
        # the duplicated const sibling (identical bytes).
        self.assertNotIn("op_expand", folded)
        self.assertIn("less(x = arange_fold_const, y = length)", folded)
        # The duplicated const sibling carries the unit axis; the
        # original const line is byte-identical.
        self.assertIn(
            "tensor<int32, [1, 375]> arange_fold_const = const()", folded
        )
        self.assertIn("offset = uint64(100)", folded)
        # Declared output shape of the consumer is unchanged.
        self.assertIn("tensor<bool, [1, 375]> mask = less(", folded)
        self.assertEqual(
            report.to_dict()["schema"], "mlx-omarchy.unit-view-fold.v1"
        )

    def test_value_semantics_match_numpy(self):
        rng = np.random.default_rng(375)
        arange = np.arange(375, dtype=np.int32)
        for length in (0, 1, 200, 374, 375):
            original = arange < np.array([[length]], np.int32)
            rewritten = arange.reshape(1, 375) < np.array(
                [length], np.int32
            )
            self.assertTrue(np.array_equal(original, rewritten))

    def test_const_fed_relabel_keeps_bytes(self):
        # The preamble already defines blob-backed arange [375]; the
        # view over it must become a const with identical bytes.
        mil = (
            _PREAMBLE
            + '    tensor<int32, [1, 375]> unit = expand_dims(axes = axes, x = arange)[name = tensor<string, []>("op_unit")];\n'
            + '    tensor<bool, [1, 375]> mask = less(x = unit, y = length)[name = tensor<string, []>("op_less")];\n'
            + "  } -> (mask);\n}\n"
        )
        folded, report = fold_unit_views(mil)
        self.assertIn("unit", report.folded)
        self.assertIn("tensor<int32, [1, 375]> unit = const()", folded)
        self.assertIn("offset = uint64(100)", folded)

    def test_matmul_consumer_is_refused_by_name(self):
        mil = (
            _PREAMBLE
            + """    tensor<int32, [1, 1]> expanded = expand_dims(axes = axes, x = length)[name = tensor<string, []>("op_expand")];
    tensor<int32, [1, 375]> out = matmul(x = expanded, y = expanded)[name = tensor<string, []>("op_mm")];
  } -> (out);
}
"""
        )
        plan = plan_unit_folds(mil)
        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].disposition, "REFUSED")
        self.assertIn("matmul", plan[0].reason)
        folded, report = fold_unit_views(mil)
        self.assertEqual(report.folded, [])
        self.assertIn("expanded", folded)  # untouched
        self.assertEqual(len(report.refused), 1)

    def test_fp16_and_bool_views_are_not_classified(self):
        mil = (
            _PREAMBLE
            + """    tensor<fp16, [1, 1]> wide = expand_dims(axes = axes, x = length)[name = tensor<string, []>("op_wide")];
    tensor<int32, [1, 1]> expanded = expand_dims(axes = axes, x = length)[name = tensor<string, []>("op_expand")];
"""
            + _FOOTER
        )
        plan = plan_unit_folds(mil)
        names = [entry.output_name for entry in plan]
        self.assertNotIn("wide", names)
        self.assertIn("expanded", names)

    def test_graph_return_view_is_refused(self):
        mil = (
            _PREAMBLE
            + """    tensor<int32, [1, 1]> expanded = expand_dims(axes = axes, x = length)[name = tensor<string, []>("op_expand")];
  } -> (expanded);
}
"""
        )
        plan = plan_unit_folds(mil)
        self.assertEqual(plan[0].disposition, "REFUSED")
        self.assertIn("graph return", plan[0].reason)


if __name__ == "__main__":
    unittest.main()
