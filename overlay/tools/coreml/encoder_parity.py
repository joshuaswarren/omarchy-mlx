# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Encoder-parity harness planner (plan sections 40-43, host only).

Builds the layer-by-layer acceptance plan for the pinned Parakeet
encoder from its parsed MIL program, extends the differential-harness
conventions (docs/differential-harness.md) with the Parakeet tolerance
policy, and defines the trace-counter report the parity run must emit.
The parity runs themselves wait on Phase 5 compiler closure; this module
is the skeleton they execute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .proto import load_model
from .reference import ReferenceLock

# Section 41, frozen. The value-exact class is fp16 equality with +/-0
# equivalence — the device semantics proven on m1-test-host (the H13 compiler
# models zero products as unsigned; mlx-omarchy-ane-worker --expect
# enforces the same rule). The relative-L2 class covers accumulated
# stages; its bound is the reference lock's frozen encoder contract and
# is never loosened after a failure.
FP16_VALUE_EXACT = "fp16_value_exact"
RELATIVE_L2 = "relative_l2"

# Ops whose outputs are pointwise/staging decisions of their inputs:
# value-exact when both sides run the same semantics. Everything else
# (reductions, matmuls, convolutions, softmax, layer_norm) accumulates
# and takes the relative-L2 class.
EXACT_OP_CLASSES = frozenset({
    "add", "sub", "mul", "cast", "select", "reshape", "transpose",
    "expand_dims", "squeeze", "split", "slice_by_index", "pad", "tile",
    "logical_and", "logical_not", "less", "floor", "floor_div",
    "reduce_min", "const", "constexpr_lut_to_dense",
})

# The nine ANE backend counters (overlay/mlx/backend/omarchy/trace.h).
ANE_COUNTER_FIELDS = (
    "ane_models_loaded",
    "ane_packages_compiled",
    "ane_package_cache_hits",
    "ane_worker_starts",
    "ane_submissions",
    "ane_timeouts",
    "ane_input_bytes",
    "ane_output_bytes",
    "ane_exec_ns",
)


class EncoderParityError(RuntimeError):
    """The encoder plan is malformed; the reason is named."""


@dataclass(frozen=True)
class StagePlan:
    """One acceptance stage of the encoder parity plan."""

    name: str
    first_op: int
    last_op: int  # inclusive
    op_count: int
    histogram: dict[str, int]
    tolerance: str
    checkpoints: tuple[str, ...]  # boundary tensors to capture and compare

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "first_op": self.first_op,
            "last_op": self.last_op,
            "op_count": self.op_count,
            "histogram": dict(sorted(self.histogram.items())),
            "tolerance": self.tolerance,
            "checkpoints": list(self.checkpoints),
        }


@dataclass(frozen=True)
class EncoderParityPlan:
    stages: tuple[StagePlan, ...]
    total_ops: int
    # Tolerance contract per op class for the first-divergence descent.
    op_tolerance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": "mlx-omarchy.parakeet-encoder-parity-plan.v1",
            "total_ops": self.total_ops,
            "stages": [stage.to_dict() for stage in self.stages],
            "op_tolerance": dict(sorted(self.op_tolerance.items())),
        }


def _stage_tolerance(histogram: dict[str, int]) -> str:
    if any(op not in EXACT_OP_CLASSES for op in histogram):
        return RELATIVE_L2
    return FP16_VALUE_EXACT


def build_stage_plan(spec) -> EncoderParityPlan:
    """Split the encoder program into acceptance stages.

    Stages are the prologue (input casts, subsampling, mask derivation),
    one stage per ``encoder_layers_N`` (N = 0..23 for the pinned
    encoder), and the epilogue (projector + output cast). Checkpoints
    are the tensors a stage produces that any LATER stage or the
    function outputs consume — exactly the boundary values the parity
    run captures.
    """
    blocks = list(_walk_blocks(spec))
    if len(blocks) != 1:
        raise EncoderParityError(
            f"expected one main block, found {len(blocks)}"
        )
    operations = list(blocks[0].operations)
    produced_by: dict[str, int] = {}
    consumers: dict[str, set[str]] = {}
    for index, op in enumerate(operations):
        for out in op.outputs:
            produced_by[out.name] = index
        for argument in op.inputs.values():
            for binding in argument.arguments:
                if binding.WhichOneof("binding") == "name":
                    consumers.setdefault(binding.name, set()).add(
                        op.outputs[0].name if op.outputs else "?"
                    )

    layer_re = re.compile(r"encoder_layers_(\d+)_")
    layer_indices = [
        index
        for index, op in enumerate(operations)
        if op.outputs and layer_re.search(op.outputs[0].name)
    ]
    if not layer_indices:
        first_layer_index = len(operations)
        last_layer_index = -1
    else:
        first_layer_index = layer_indices[0]
        last_layer_index = layer_indices[-1]

    def stage_of(index: int, previous: str | None) -> str:
        op = operations[index]
        for out in op.outputs:
            match = layer_re.search(out.name)
            if match:
                return f"encoder_layers_{int(match.group(1)):02d}"
        # Non-layer ops attach to the surrounding region: prologue
        # before the first layer op, epilogue after the last one, and
        # the current layer in between (shared constants and glue).
        if first_layer_index == len(operations):
            return "prologue"
        if index < first_layer_index:
            return "prologue"
        if index > last_layer_index:
            return "epilogue"
        return previous if previous not in (None, "prologue") else "prologue"

    boundaries: list[tuple[str, int]] = []
    previous = None
    for index in range(len(operations)):
        stage = stage_of(index, previous)
        if stage != previous:
            boundaries.append((stage, index))
            previous = stage

    function_outputs = {
        out.name for out in _function_named_outputs(spec)
    }

    stages: list[StagePlan] = []
    for position, (name, start) in enumerate(boundaries):
        end = (
            boundaries[position + 1][1] - 1
            if position + 1 < len(boundaries)
            else len(operations) - 1
        )
        histogram: dict[str, int] = {}
        for op in operations[start : end + 1]:
            histogram[op.type] = histogram.get(op.type, 0) + 1
        checkpoints = set()
        for index in range(start, end + 1):
            if operations[index].type == "const":
                continue  # constants are inputs, not computed checkpoints
            for out in operations[index].outputs:
                consumers_outside = any(
                    produced_by.get(consumer, -1) > end
                    or consumer in function_outputs
                    for consumer in consumers.get(out.name, ())
                    if consumer != out.name
                )
                if out.name in function_outputs or consumers_outside:
                    checkpoints.add(out.name)
        stages.append(
            StagePlan(
                name=name,
                first_op=start,
                last_op=end,
                op_count=end - start + 1,
                histogram=histogram,
                tolerance=_stage_tolerance(histogram),
                checkpoints=tuple(sorted(checkpoints)),
            )
        )

    op_tolerance = {}
    for stage in stages:
        for op in stage.histogram:
            # Section 41 semantics: pointwise/staging ops compare with
            # fp16 value equality (+/-0 equivalence); accumulating ops
            # take the relative-L2 class regardless of their stage.
            op_tolerance.setdefault(
                op,
                FP16_VALUE_EXACT if op in EXACT_OP_CLASSES else RELATIVE_L2,
            )

    total = sum(stage.op_count for stage in stages)
    if total != len(operations):
        raise EncoderParityError(
            f"stage plan covers {total} ops, program has {len(operations)}"
        )
    return EncoderParityPlan(
        stages=tuple(stages), total_ops=total, op_tolerance=op_tolerance
    )


def load_encoder_spec(encoder_package: Path):
    from .mlpackage import open_mlpackage

    package = open_mlpackage(Path(encoder_package))
    return load_model(package.model_path.read_bytes())


def frozen_tolerances(lock: ReferenceLock) -> dict:
    """The section-41 acceptance numbers, from the reference lock."""
    contract = lock.numerical_contract
    if contract is None:
        raise EncoderParityError(
            "reference lock has no frozen numerical contract"
        )
    return {
        "encoder_max_abs_err": contract.encoder_max_abs_err,
        "encoder_mean_abs_err": contract.encoder_mean_abs_err,
        "encoder_rel_l2_err": contract.encoder_rel_l2_err,
        "nan_count_allowed": contract.nan_count_allowed,
        "inf_count_allowed": contract.inf_count_allowed,
        "token_ids_must_match_exactly": contract.token_ids_must_match_exactly,
        "transcript_must_match_exactly": (
            contract.transcript_must_match_exactly
        ),
        "fp16_value_semantics": "+/-0 equivalence (m1-test-host-proven, "
        "unsigned-zero-product model)",
    }


def golden_anchors(lock: ReferenceLock) -> dict[str, str]:
    """Golden artifacts the parity run compares against, by logical
    name → sha256, from the lock's captured reference paths (keyed by
    file name)."""
    wanted = {
        "mel": "mel.npy",
        "mel_mask": "mel_mask.npy",
        "encoder_input_features": "encoder_input_features.npy",
        "encoder_input_mask": "encoder_input_mask.npy",
        "encoder_hidden": "encoder_hidden.npy",
        "encoder_mask": "encoder_mask.npy",
        "token_ids": "token_ids.json",
        "transcript": "transcript.txt",
        "waveform": "waveform.npy",
    }
    anchors = {}
    for logical, file_name in wanted.items():
        sha = lock.macos_reference_paths.get(file_name)
        if sha is None:
            raise EncoderParityError(
                f"reference lock is missing the golden '{file_name}' anchor"
            )
        anchors[logical] = sha
    return anchors


def empty_counter_report() -> dict:
    """The section-42 report skeleton the parity run must fill."""
    report = {field: 0 for field in ANE_COUNTER_FIELDS}
    report.update(
        {
            "gpu_primitive_dispatches": 0,
            "vk_compute_dispatches": 0,
            "cpu_tensor_events": 0,
            "accounted_input_bytes": 0,
            "accounted_output_bytes": 0,
            "expected_input_bytes": None,
            "expected_output_bytes": None,
        }
    )
    return report


def evaluate_invariants(report: dict) -> list[str]:
    """Section 42/43 invariants. Returns the list of violations (empty
    = clean). A clean parity run must account every compared byte on
    the ANE or Vulkan paths with zero CPU tensor events and zero
    timeouts."""
    violations = []
    for field in ANE_COUNTER_FIELDS:
        if field not in report:
            violations.append(f"missing counter '{field}'")
    if report.get("ane_timeouts", 0) != 0:
        violations.append(
            f"ane_timeouts={report['ane_timeouts']} (must be 0 on a "
            "clean parity run)"
        )
    completions = report.get("ane_submissions", 0) - report.get(
        "ane_timeouts", 0
    )
    if completions < 0:
        violations.append("more timeouts than submissions")
    if report.get("cpu_tensor_events", 0) != 0:
        violations.append(
            f"cpu_tensor_events={report['cpu_tensor_events']} (section 43: "
            "no CPU tensor fallback is allowed on the parity path)"
        )
    for side in ("input", "output"):
        expected = report.get(f"expected_{side}_bytes")
        accounted = report.get(f"accounted_{side}_bytes")
        if expected is not None and accounted != expected:
            violations.append(
                f"{side} byte accounting mismatch: accounted {accounted}, "
                f"expected {expected}"
            )
    return violations


def _walk_blocks(spec):
    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            yield block


def _function_named_outputs(spec):
    # MIL blocks declare their return values as plain name strings
    # (Block.outputs); the producing operation carries the ValueType.
    class _Named:
        def __init__(self, name):
            self.name = name

    for block in _walk_blocks(spec):
        for name in block.outputs:
            yield _Named(name)
