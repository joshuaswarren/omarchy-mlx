# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for promoting encoder select fill a and fp16 cond."""

import unittest
from pathlib import Path

import numpy as np

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.select_runtime import (
    STANDALONE_MIL,
    SelectRuntimeError,
    materialize_bool_cond,
    materialize_fill,
    numpy_const_vs_runtime_fill,
    plan_select_runtime,
    require_exact,
    rewrite_select_runtime,
)

_SHAPE = (1, 8, 375, 375)

_ENCODER = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 8, 375, 375]> matrix_bd_5_cast_fp16, tensor<bool, [1, 1, 375, 375]> var_373) {
    tensor<fp16, []> var_8_to_fp16 = const()[name = tensor<string, []>("op_8_to_fp16"), val = tensor<fp16, []>(BLOBFILE(path = string("@model_path/weights/normalized.bin"), offset = uint64(31460800)))];
    tensor<fp16, [1, 8, 375, 375]> attention_mask_9_cast_fp16 = select(a = var_8_to_fp16, b = matrix_bd_5_cast_fp16, cond = var_373)[name = tensor<string, []>("attention_mask_9_cast_fp16")];
    tensor<fp16, []> var_13_to_fp16 = const()[name = tensor<string, []>("op_13_to_fp16"), val = tensor<fp16, []>(BLOBFILE(path = string("@model_path/weights/normalized.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 1024, 375]> input_41_cast_fp16 = select(a = var_13_to_fp16, b = matrix_bd_5_cast_fp16, cond = var_373)[name = tensor<string, []>("input_41_cast_fp16")];
  } -> (attention_mask_9_cast_fp16);
}
"""


class NumpyProofTest(unittest.TestCase):
    def test_const_ninf_equals_runtime_fill(self):
        rng = np.random.default_rng(20260913)
        cond = rng.integers(0, 2, size=_SHAPE).astype(bool)
        cond[0, 0, 0, 0] = False
        cond[0, 7, 374, 374] = True
        scores = rng.standard_normal(_SHAPE).astype(np.float16)
        scores[0, 0, 0, 0] = np.float16("-0")
        original, rewritten = numpy_const_vs_runtime_fill(cond, scores)
        require_exact(original, rewritten, "select const -inf vs runtime fill")
        self.assertEqual(int(original.view("<u2")[0, 0, 0, 0]), 0x8000)
        self.assertEqual(int(original.view("<u2")[0, 7, 374, 374]), 0xFC00)

    def test_broadcast_cond_matches_full_fill(self):
        rng = np.random.default_rng(20260913)
        cond = rng.integers(0, 2, size=(1, 1, 375, 375)).astype(bool)
        scores = rng.standard_normal(_SHAPE).astype(np.float16)
        original, rewritten = numpy_const_vs_runtime_fill(cond, scores)
        require_exact(original, rewritten, "broadcast cond runtime fill")
        fill = materialize_fill(_SHAPE)
        self.assertEqual(fill.shape, _SHAPE)
        self.assertTrue(np.all(fill.view("<u2") == 0xFC00))

    def test_fp16_zero_one_becomes_bool(self):
        mask = np.zeros(_SHAPE, dtype=np.float16)
        mask[0, 0, 0, 0] = np.float16(1.0)
        cond = materialize_bool_cond(mask)
        self.assertFalse(bool(cond[0, 1, 1, 1]))
        self.assertTrue(bool(cond[0, 0, 0, 0]))
        self.assertEqual(int(np.float16(1.0).view("<u2")), 0x3C00)
        scores = np.ones(_SHAPE, dtype=np.float16)
        original, rewritten = numpy_const_vs_runtime_fill(cond, scores)
        require_exact(original, rewritten, "bool cond from fp16 0/1")

    def test_non_zero_one_fp16_is_refused(self):
        mask = np.zeros((2,), dtype=np.float16)
        mask[1] = np.float16(0.5)
        with self.assertRaises(SelectRuntimeError) as ctx:
            materialize_bool_cond(mask)
        self.assertIn("not bit-exact 0/1", str(ctx.exception))


class SelectRuntimeRewriteTest(unittest.TestCase):
    def test_rewrites_encoder_ninf_select(self):
        plan = plan_select_runtime(_ENCODER)
        by_name = {entry.output_name: entry for entry in plan}
        self.assertEqual(by_name["attention_mask_9_cast_fp16"].disposition, "REWRITTEN")
        self.assertEqual(by_name["input_41_cast_fp16"].disposition, "REFUSED")
        self.assertIn("outside the runtime-a envelope", by_name["input_41_cast_fp16"].reason)

        text, report = rewrite_select_runtime(_ENCODER)
        self.assertEqual(report.rewritten, ["attention_mask_9_cast_fp16"])
        self.assertEqual(len(report.refused), 1)
        self.assertIn(
            "tensor<fp16, [1, 8, 375, 375]> var_8_to_fp16_rt",
            text,
        )
        self.assertIn(
            "select(a = var_8_to_fp16_rt, b = matrix_bd_5_cast_fp16, cond = var_373)",
            text,
        )
        self.assertNotIn("a = var_8_to_fp16,", text)
        self.assertNotIn("tensor<fp16, []> var_8_to_fp16 = const()", text)
        self.assertIn("select(a = var_13_to_fp16,", text)
        self.assertEqual(report.fills[0]["input"], "var_8_to_fp16_rt")
        self.assertEqual(report.fills[0]["shape"], [1, 8, 375, 375])
        self.assertEqual(report.fills[0]["storage"], "blob")
        self.assertEqual(
            report.to_dict()["schema"],
            "mlx-omarchy.select-runtime-rewrite.v1",
        )

    def test_standalone_mil_is_runtime_a(self):
        self.assertIn("tensor<fp16, [1, 8, 375, 375]> a", STANDALONE_MIL)
        self.assertIn("tensor<bool, [1, 8, 375, 375]> cond", STANDALONE_MIL)
        self.assertIn("select(a = a, b = b, cond = cond)", STANDALONE_MIL)
        self.assertNotIn("const()", STANDALONE_MIL)
        plan = plan_select_runtime(STANDALONE_MIL)
        self.assertEqual(plan, [])

    def test_fp16_cond_becomes_bool(self):
        mil = """program(1)
{
  func main<CoreML8>(tensor<fp16, [1, 8, 375, 375]> b, tensor<fp16, [1, 8, 375, 375]> cond) {
    tensor<fp16, []> ninf = const()[name = string("ninf"), val = tensor<fp16, []>(-inf)];
    tensor<fp16, [1, 8, 375, 375]> y = select(a = ninf, b = b, cond = cond)[name = string("y")];
  } -> (y);
}
"""
        plan = plan_select_runtime(mil)
        self.assertEqual(plan[0].disposition, "REWRITTEN")
        self.assertIn("materialize fp16 cond as bool", plan[0].reason)
        text, report = rewrite_select_runtime(mil)
        self.assertEqual(report.rewritten, ["y"])
        self.assertIn("tensor<bool, [1, 8, 375, 375]> cond", text)
        self.assertNotIn("tensor<fp16, [1, 8, 375, 375]> cond", text)
        self.assertIn("select(a = ninf_rt, b = b, cond = cond)", text)
        self.assertNotIn("a = ninf,", text)
        self.assertEqual(report.bool_conds[0]["storage"], "param")
        self.assertEqual(report.bool_conds[0]["input"], "cond")
        self.assertEqual(report.fills[0]["input"], "ninf_rt")

    def test_real_encoder_mil_plans_the_ninf_selects(self):
        mil_path = (
            Path(__file__).resolve().parents[4]
            / "receipts"
            / "2026-09-13-fp16-output-peel"
            / "model-fp16-hidden-only.mil"
        )
        if not mil_path.is_file():
            self.skipTest("peeled encoder MIL receipt is not present")
        mil = mil_path.read_text()
        plan = plan_select_runtime(mil)
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
        self.assertTrue(all(name.startswith("attention_mask_") for name in rewritten))
        self.assertEqual(len(rewritten), 24)
        self.assertEqual(len(refused), 24)
        text, report = rewrite_select_runtime(mil)
        self.assertEqual(report.rewritten, rewritten)
        self.assertEqual(len(report.fills), 1)
        self.assertIn("a = var_8_to_fp16_rt", text)
        self.assertNotIn("select(a = var_8_to_fp16,", text)
        self.assertIn("select(a = var_13_to_fp16,", text)
        fill = report.fills[0]["input"]
        self.assertIn(f"tensor<fp16, [1, 8, 375, 375]> {fill}", text)

    def test_noslice_fp16_cond_rewrites_24(self):
        mil_path = (
            Path(__file__).resolve().parents[4]
            / "receipts"
            / "2026-09-13-slice-layout-rewrite"
            / "model-fp16-noslice.mil"
        )
        if not mil_path.is_file():
            self.skipTest("noslice encoder MIL receipt is not present")
        mil = mil_path.read_text()
        plan = plan_select_runtime(mil)
        rewritten = [
            entry.output_name
            for entry in plan
            if entry.disposition == "REWRITTEN"
        ]
        self.assertTrue(all(name.startswith("attention_mask_") for name in rewritten))
        self.assertEqual(len(rewritten), 24)
        text, report = rewrite_select_runtime(mil)
        self.assertEqual(report.rewritten, rewritten)
        self.assertEqual(len(report.fills), 1)
        self.assertEqual(report.bool_conds[0]["source"], "var_373")
        self.assertEqual(report.bool_conds[0]["storage"], "promoted")
        self.assertEqual(report.bool_conds[0]["shape"], [1, 1, 375, 375])
        self.assertIn("cond = var_373_bool", text)
        self.assertIn("tensor<bool, [1, 1, 375, 375]> var_373_bool", text)
        self.assertIn("a = var_8_to_fp16_rt", text)
        self.assertIn("tensor<fp16, [1, 1, 375, 375]> var_373", text)


if __name__ == "__main__":
    unittest.main()
