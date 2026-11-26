# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host tests for the encoder-parity harness planner (sections 40-43)."""

import os
import unittest
from pathlib import Path

try:
    from _bootstrap import _TOOLS  # plain-script execution
except ImportError:  # package discovery (omarchy.coreml.*)
    from ._bootstrap import _TOOLS  # noqa: F401

from coreml.encoder_parity import (
    ANE_COUNTER_FIELDS,
    FP16_VALUE_EXACT,
    RELATIVE_L2,
    build_stage_plan,
    empty_counter_report,
    evaluate_invariants,
    frozen_tolerances,
    golden_anchors,
    load_encoder_spec,
)
from coreml.reference import ReferenceLock
from coreml.schema import Model_pb2

_HF_REVISION = "b650695c2322ee5281dff48d7345b2f3a58ff018"
_ENCODER = (
    "mweinbach1/parakeet-tdt-0.6b-v3-coreml" f"/{_HF_REVISION}"
    "/encoder.mlpackage"
)


def _cache() -> Path | None:
    base = os.environ.get("MLX_OMARCHY_CACHE_DIR")
    roots = []
    if base:
        roots.append(Path(base) / "parakeet-reference")
    roots.append(Path.home() / ".cache" / "mlx-omarchy" / "parakeet-reference")
    for root in roots:
        candidate = root / _ENCODER
        if candidate.is_dir():
            return candidate
    return None


def _wire(op, key, name):
    op.inputs[key].arguments.add().name = name


def _out(op, name):
    output = op.outputs.add()
    output.name = name


def _toy_spec():
    """prologue add -> layer matmul -> epilogue cast, with a shared const."""
    spec = Model_pb2.Model()
    spec.specificationVersion = 9
    block = spec.mlProgram.functions["main"].block_specializations["CoreML8"]

    shared = block.operations.add()
    shared.type = "const"
    _out(shared, "shared_const")

    add = block.operations.add()
    add.type = "add"
    _wire(add, "x", "input_features")
    _wire(add, "y", "shared_const")
    _out(add, "prologue_sum")

    matmul = block.operations.add()
    matmul.type = "matmul"
    _wire(matmul, "x", "prologue_sum")
    _wire(matmul, "y", "shared_const")
    _out(matmul, "encoder_layers_0_attention")

    cast = block.operations.add()
    cast.type = "cast"
    _wire(cast, "x", "encoder_layers_0_attention")
    _out(cast, "encoder_hidden")

    block.outputs.append("encoder_hidden")
    return spec


class EncoderParityPlanTest(unittest.TestCase):
    def test_toy_spec_stages_checkpoints_and_tolerances(self):
        plan = build_stage_plan(_toy_spec())
        self.assertEqual(plan.total_ops, 4)
        self.assertEqual(
            [stage.name for stage in plan.stages],
            ["prologue", "encoder_layers_00", "epilogue"],
        )
        self.assertEqual(
            [stage.op_count for stage in plan.stages], [2, 1, 1]
        )
        prologue = plan.stages[0]
        # The const is not a checkpoint; the consumed add result is.
        self.assertEqual(prologue.checkpoints, ("prologue_sum",))
        self.assertEqual(prologue.tolerance, FP16_VALUE_EXACT)
        self.assertEqual(plan.stages[1].tolerance, RELATIVE_L2)
        self.assertEqual(
            plan.stages[2].checkpoints, ("encoder_hidden",)
        )
        self.assertEqual(plan.op_tolerance["add"], FP16_VALUE_EXACT)
        self.assertEqual(plan.op_tolerance["matmul"], RELATIVE_L2)
        self.assertEqual(plan.op_tolerance["cast"], FP16_VALUE_EXACT)
        self.assertEqual(plan.op_tolerance["const"], FP16_VALUE_EXACT)

    def test_real_encoder_plan(self):
        cache = _cache()
        if cache is None:
            self.skipTest("parakeet reference cache not present")
        plan = build_stage_plan(load_encoder_spec(cache))
        self.assertEqual(plan.total_ops, 3351)
        self.assertEqual(len(plan.stages), 26)
        names = [stage.name for stage in plan.stages]
        self.assertEqual(names[0], "prologue")
        self.assertEqual(names[-1], "epilogue")
        self.assertEqual(
            names[1:3], ["encoder_layers_00", "encoder_layers_01"]
        )
        self.assertEqual(names[-2], "encoder_layers_23")
        layers = [s for s in plan.stages if s.name.startswith("encoder_layers")]
        self.assertEqual(len(layers), 24)
        self.assertEqual(layers[0].op_count, 146)
        self.assertEqual(
            [s.op_count for s in layers[1:23]], [132] * 22
        )
        # Boundary values the parity run must capture.
        self.assertEqual(
            plan.stages[-1].checkpoints,
            ("encoder_hidden", "encoder_mask", "linear_217_cast_fp16"),
        )
        self.assertIn("attention_mask_7", plan.stages[0].checkpoints)
        # Section 41 semantics per op class.
        self.assertEqual(plan.op_tolerance["add"], FP16_VALUE_EXACT)
        self.assertEqual(plan.op_tolerance["select"], FP16_VALUE_EXACT)
        self.assertEqual(
            plan.op_tolerance["logical_and"], FP16_VALUE_EXACT
        )
        for accumulating in ("linear", "matmul", "conv", "softmax",
                             "layer_norm"):
            self.assertEqual(plan.op_tolerance[accumulating], RELATIVE_L2)
        # Every op appears in exactly one stage and the histogram totals
        # reproduce the known encoder histogram.
        merged = {}
        for stage in plan.stages:
            for op, count in stage.histogram.items():
                merged[op] = merged.get(op, 0) + count
        self.assertEqual(merged["const"], 1783)
        self.assertEqual(merged["constexpr_lut_to_dense"], 194)
        self.assertEqual(merged["linear"], 194)
        self.assertEqual(merged["layer_norm"], 120)
        self.assertEqual(merged["conv"], 77)


class ToleranceContractTest(unittest.TestCase):
    def test_frozen_tolerances_come_from_the_lock(self):
        lock = ReferenceLock.load()
        tolerances = frozen_tolerances(lock)
        self.assertEqual(tolerances["encoder_max_abs_err"], 0.3)
        self.assertEqual(tolerances["encoder_mean_abs_err"], 0.02)
        self.assertEqual(tolerances["encoder_rel_l2_err"], 0.1)
        self.assertEqual(tolerances["nan_count_allowed"], 0)
        self.assertTrue(tolerances["token_ids_must_match_exactly"])
        self.assertIn("+/-0 equivalence", tolerances["fp16_value_semantics"])

    def test_golden_anchors_cover_the_encoder_boundary(self):
        lock = ReferenceLock.load()
        anchors = golden_anchors(lock)
        for logical in (
            "mel",
            "encoder_input_features",
            "encoder_input_mask",
            "encoder_hidden",
            "encoder_mask",
            "token_ids",
            "transcript",
        ):
            self.assertIn(logical, anchors)
            self.assertRegex(anchors[logical], r"^[0-9a-f]{64}$")


class TraceInvariantTest(unittest.TestCase):
    def test_clean_report_has_no_violations(self):
        report = empty_counter_report()
        report["ane_submissions"] = 26
        report["ane_worker_starts"] = 26
        report["expected_input_bytes"] = 0
        report["expected_output_bytes"] = 0
        self.assertEqual(evaluate_invariants(report), [])

    def test_timeout_cpu_fallback_and_byte_mismatch_are_violations(self):
        report = empty_counter_report()
        report["ane_timeouts"] = 1
        self.assertTrue(evaluate_invariants(report))

        report = empty_counter_report()
        report["cpu_tensor_events"] = 3
        violations = evaluate_invariants(report)
        self.assertTrue(any("section 43" in v for v in violations))

        report = empty_counter_report()
        report["expected_output_bytes"] = 128
        report["accounted_output_bytes"] = 64
        self.assertTrue(
            any("output byte accounting" in v for v in evaluate_invariants(report))
        )

        report = {field: 0 for field in ANE_COUNTER_FIELDS[:3]}
        self.assertTrue(
            any("missing counter" in v for v in evaluate_invariants(report))
        )

    def test_report_carries_all_nine_ane_counters(self):
        report = empty_counter_report()
        for field in ANE_COUNTER_FIELDS:
            self.assertIn(field, report)
        self.assertEqual(len(ANE_COUNTER_FIELDS), 9)


if __name__ == "__main__":
    unittest.main()
