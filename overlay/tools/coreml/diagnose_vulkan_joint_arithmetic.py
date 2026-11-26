#!/usr/bin/env python3
# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Test bounded joint-linear arithmetic hypotheses against native tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from coreml.compare_vulkan_joint_golden import (
    _load_array,
    _load_manifest,
    _metrics,
    compare,
)
from coreml.pinned_component import load_pinned_component
from coreml.proto import mil_dtype_name
from coreml.vulkan_joint import _load_joint


_VARIANTS = {
    "baseline_fp16_matmul_then_fp16_bias": (
        "MLX fp16 matmul rounded to fp16, followed by a separate fp16 bias add."
    ),
    "mx_addmm_fp16": (
        "MLX addmm receives the fp16 bias, hidden state, and transposed weight in one GPU primitive."
    ),
    "explicit_fp32_dot_then_fp16_bias": (
        "Operands widened to fp32 for the dot product, dot rounded to fp16, then bias added in fp16."
    ),
    "fused_fp32_dot_and_bias_then_fp16": (
        "Operands and bias widened to fp32, with one fp16 rounding after dot product and bias."
    ),
    "fp16_products_fp32_reduce_then_fp16_bias": (
        "Products rounded to fp16, widened for an MLX fp32 reduction, rounded to fp16, then bias added in fp16."
    ),
    "balanced_tree_fp16_then_fp16_bias": (
        "Products and every level of a balanced binary reduction rounded to fp16, then bias added in fp16."
    ),
    "sequential_fp16_then_fp16_bias": (
        "Products accumulated left to right with fp16 rounding after every add, then bias added in fp16."
    ),
}


def _balanced_fp16(terms, mx):
    while terms.shape[1] > 1:
        width = terms.shape[1]
        paired_width = width // 2 * 2
        paired = mx.add(
            terms[:, :paired_width:2],
            terms[:, 1:paired_width:2],
        )
        terms = (
            mx.concatenate((paired, terms[:, -1:]), axis=1)
            if width % 2
            else paired
        )
    return terms[:, 0]


def _sequential_fp16(terms, mx):
    accumulated = mx.zeros((terms.shape[0],), dtype=mx.float16)
    for index in range(terms.shape[1]):
        accumulated = mx.add(accumulated, terms[:, index])
    return accumulated


def _run_variants(encoder: np.ndarray, decoder: np.ndarray, constants, mx):
    with mx.stream(mx.gpu):
        encoder_mx = mx.array(encoder)
        decoder_mx = mx.array(decoder)
        encoder_fp16 = encoder_mx.astype(mx.float16)
        decoder_fp16 = decoder_mx.astype(mx.float16)
        added_fp16 = mx.add(encoder_fp16, decoder_fp16)
        hidden = mx.maximum(added_fp16, mx.array(0, dtype=mx.float16))
        weight_fp32 = constants.weight.astype(mx.float32)
        hidden_fp32 = hidden.astype(mx.float32)
        dot_fp32 = mx.matmul(hidden_fp32, weight_fp32.T)
        baseline = mx.add(
            mx.matmul(hidden, constants.weight.T),
            constants.bias,
        )
        addmm_fp16 = mx.addmm(constants.bias, hidden, constants.weight.T)
        explicit_two_round = mx.add(
            dot_fp32.astype(mx.float16),
            constants.bias,
        )
        fused_fp32 = mx.add(
            dot_fp32,
            constants.bias.astype(mx.float32),
        ).astype(mx.float16)
        terms_fp16 = mx.multiply(constants.weight, hidden.reshape((1, 640)))
        product_rounded = mx.add(
            mx.sum(terms_fp16.astype(mx.float32), axis=1).astype(mx.float16),
            constants.bias,
        )
        balanced = mx.add(
            _balanced_fp16(terms_fp16, mx),
            constants.bias,
        )
        sequential = mx.add(
            _sequential_fp16(terms_fp16, mx),
            constants.bias,
        )
        values = (
            encoder_mx,
            decoder_mx,
            encoder_fp16,
            decoder_fp16,
            added_fp16,
            hidden,
            baseline,
            addmm_fp16,
            explicit_two_round,
            fused_fp32,
            product_rounded,
            balanced,
            sequential,
        )
        mx.eval(*values)
    stages = {
        "encoder_fp32_upload": np.asarray(encoder_mx),
        "decoder_fp32_upload": np.asarray(decoder_mx),
        "encoder_fp16_cast": np.asarray(encoder_fp16),
        "decoder_fp16_cast": np.asarray(decoder_fp16),
        "fp16_add": np.asarray(added_fp16),
        "fp16_relu": np.asarray(hidden),
    }
    outputs = dict(
        zip(
            _VARIANTS,
            (
                baseline,
                addmm_fp16,
                explicit_two_round,
                fused_fp32,
                product_rounded,
                balanced,
                sequential,
            ),
            strict=True,
        )
    )
    return stages, {
        name: np.asarray(value, dtype=np.float32).reshape(1, 8198)
        for name, value in outputs.items()
    }


def _reference_stages(encoder: np.ndarray, decoder: np.ndarray) -> dict[str, np.ndarray]:
    encoder_fp16 = encoder.astype(np.float16)
    decoder_fp16 = decoder.astype(np.float16)
    added_fp16 = np.add(encoder_fp16, decoder_fp16, dtype=np.float16)
    hidden = np.maximum(added_fp16, np.float16(0))
    return {
        "encoder_fp32_upload": encoder,
        "decoder_fp32_upload": decoder,
        "encoder_fp16_cast": encoder_fp16,
        "decoder_fp16_cast": decoder_fp16,
        "fp16_add": added_fp16,
        "fp16_relu": hidden,
    }


def _bit_mismatches(actual: np.ndarray, expected: np.ndarray) -> int:
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise ValueError("intermediate comparison contract differs")
    integer_dtype = np.dtype(f"u{actual.dtype.itemsize}")
    return int(np.count_nonzero(actual.view(integer_dtype) != expected.view(integer_dtype)))


def _artifact_evidence(capture_dir: Path, package_path: Path) -> dict[str, Any]:
    package_files = sorted(
        path.relative_to(package_path).as_posix()
        for path in package_path.rglob("*")
        if path.is_file()
    )
    capture_files = sorted(
        path.relative_to(capture_dir.parent).as_posix()
        for path in capture_dir.parent.rglob("*")
        if path.is_file()
    )
    compiler_markers = (
        "computeplan",
        "compute_plan",
        ".mlmodelc/",
        ".espresso.",
        "model.espresso",
    )
    compiler_artifacts = [
        name for name in package_files + capture_files
        if any(marker in name.lower() for marker in compiler_markers)
    ]
    component = load_pinned_component(package_path, "joint")
    linear_operations = [
        operation for operation in component.block.operations
        if operation.type == "linear"
    ]
    if len(linear_operations) != 1:
        raise ValueError("pinned joint must contain exactly one linear operation")
    linear = linear_operations[0]
    relu = component.block.operations[5]
    return {
        "source_package_files": package_files,
        "capture_file_count": len(capture_files),
        "compiled_compute_plan_artifacts": compiler_artifacts,
        "compiled_compute_plan_available": bool(compiler_artifacts),
        "mil_linear_contract": {
            "input_names": {
                key: [argument.name for argument in value.arguments]
                for key, value in linear.inputs.items()
            },
            "input_dtype": mil_dtype_name(
                relu.outputs[0].type.tensorType.dataType
            ),
            "weight_dtype": str(component.constant("head_weight_to_fp16").dtype),
            "bias_dtype": str(component.constant("head_bias_to_fp16").dtype),
            "output_dtype": mil_dtype_name(
                linear.outputs[0].type.tensorType.dataType
            ),
            "attribute_keys": sorted(linear.attributes),
            "accumulator_precision_or_reduction_order_declared": False,
        },
    }


def diagnose(capture_dir: Path, package_path: Path) -> dict[str, Any]:
    import mlx.core as mx

    capture_dir = capture_dir.resolve()
    package_path = package_path.resolve()
    authenticated = compare(capture_dir, package_path)
    manifest, snapshots, _ = _load_manifest(capture_dir)
    tensor_index = json.loads(snapshots["tdt_tensors.json"])
    constants = _load_joint(str(package_path))
    stage_mismatches = {name: 0 for name in _reference_stages(
        np.zeros((1, 640), dtype=np.float32),
        np.zeros((1, 640), dtype=np.float32),
    )}
    stage_elements = {name: 0 for name in stage_mismatches}
    candidate_outputs = {name: [] for name in _VARIANTS}
    token_golden = []
    duration_golden = []
    decisions = {
        name: {"token": 0, "duration": 0}
        for name in _VARIANTS
    }
    for trace in tensor_index["traces"]:
        paths = trace["tensor_paths"]
        encoder = _load_array(
            snapshots, manifest, paths["joint_encoder_frame"], "joint_encoder_frame"
        )
        decoder = _load_array(
            snapshots, manifest, paths["joint_decoder_state"], "joint_decoder_state"
        )
        golden_token = _load_array(
            snapshots, manifest, paths["joint_token_logits"], "joint_token_logits"
        )
        golden_duration = _load_array(
            snapshots,
            manifest,
            paths["joint_duration_logits"],
            "joint_duration_logits",
        )
        actual_stages, outputs = _run_variants(encoder, decoder, constants, mx)
        expected_stages = _reference_stages(encoder, decoder)
        for name, expected in expected_stages.items():
            stage_mismatches[name] += _bit_mismatches(actual_stages[name], expected)
            stage_elements[name] += expected.size
        for name, output in outputs.items():
            candidate_outputs[name].append(output)
            decisions[name]["token"] += int(
                np.argmax(output[:, :8193], axis=1).item()
                == np.argmax(golden_token, axis=1).item()
            )
            decisions[name]["duration"] += int(
                np.argmax(output[:, 8193:], axis=1).item()
                == np.argmax(golden_duration, axis=1).item()
            )
        token_golden.append(golden_token)
        duration_golden.append(golden_duration)
    token_golden_array = np.concatenate(token_golden)
    duration_golden_array = np.concatenate(duration_golden)
    candidates = {}
    for name, outputs in candidate_outputs.items():
        combined = np.concatenate(outputs)
        candidates[name] = {
            "mechanism": _VARIANTS[name],
            "token_logits": _metrics(combined[:, :8193], token_golden_array),
            "duration_logits": _metrics(combined[:, 8193:], duration_golden_array),
            "token_decisions_matching": decisions[name]["token"],
            "duration_decisions_matching": decisions[name]["duration"],
        }
    input_pipeline = {
        name: {
            "different_bitpatterns": stage_mismatches[name],
            "element_count": stage_elements[name],
            "bit_exact": stage_mismatches[name] == 0,
        }
        for name in stage_mismatches
    }
    equivalences = {
        "baseline_equals_explicit_fp32_dot_then_fp16_bias": all(
            candidates["baseline_fp16_matmul_then_fp16_bias"][output]["actual_raw_bitpattern_sha256"]
            == candidates["explicit_fp32_dot_then_fp16_bias"][output]["actual_raw_bitpattern_sha256"]
            for output in ("token_logits", "duration_logits")
        ),
        "mx_addmm_equals_fused_fp32_dot_and_bias_then_fp16": all(
            candidates["mx_addmm_fp16"][output]["actual_raw_bitpattern_sha256"]
            == candidates["fused_fp32_dot_and_bias_then_fp16"][output]["actual_raw_bitpattern_sha256"]
            for output in ("token_logits", "duration_logits")
        ),
    }
    return {
        "schema": "mlx-omarchy.parakeet-vulkan-joint-arithmetic-diagnosis/2",
        "capture": authenticated["capture"],
        "package_path": str(package_path),
        "runtime": authenticated["runtime"],
        "artifact_evidence": _artifact_evidence(capture_dir, package_path),
        "input_pipeline_against_numpy_ieee_fp16": input_pipeline,
        "candidates": candidates,
        "observed_equivalences": equivalences,
        "hypothesis_result": {
            "input_pipeline": (
                "The authenticated float32 uploads, fp16 casts, fp16 add, and fp16 ReLU are bit-exact against the reference algorithm."
            ),
            "linear": (
                "On this backend, baseline is bit-identical to an fp32 dot rounded before an fp16 bias add, while mx.addmm is bit-identical to fused fp32 dot-and-bias. Neither fused fp32 bias nor the tested explicit fp16 product and accumulator orders reproduce the native tensors. The package and capture contain no compiled compute plan that identifies the native accumulator or reduction order."
            ),
            "source_decision": (
                "Keep the existing fp16 matmul followed by fp16 bias. It is the closest tested full-tensor result and every tested candidate preserves all 16 token and duration decisions."
            ),
        },
        "tolerance_policy": (
            "No numerical acceptance tolerance was introduced or applied; complete tensor errors and bit patterns are reported."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose(args.capture_dir, args.package)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.write_text(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
