# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Vulkan execution of the exact pinned Parakeet joint MIL program."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from typing import NamedTuple

from .pinned_component import PinnedComponentError, load_pinned_component


class JointOutput(NamedTuple):
    token_logits: object
    duration_logits: object


class _JointConstants(NamedTuple):
    weight: object
    bias: object


def _mlx():
    import mlx.core as mx

    return mx


def _constant_scalar(component, name: str, expected):
    value = component.constant(name)
    if value.shape != () or value.item() != expected:
        raise PinnedComponentError(f"pinned joint constant {name!r} is not {expected!r}")


def _constant_vector(component, name: str, expected) -> None:
    value = component.constant(name)
    if value.shape != (len(expected),) or value.tolist() != list(expected):
        raise PinnedComponentError(f"pinned joint constant {name!r} differs")


@cache
def _load_joint(package_path: str) -> _JointConstants:
    mx = _mlx()
    component = load_pinned_component(Path(package_path), "joint")
    expected_operations = (
        "const", "const", "cast", "cast", "add", "relu", "const", "const",
        "linear", "const", "const", "const", "slice_by_index", "const",
        "const", "const", "const", "slice_by_index", "const", "cast", "cast",
    )
    if tuple(operation.type for operation in component.block.operations) != expected_operations:
        raise PinnedComponentError("pinned joint operation order differs")
    if list(component.block.outputs) != ["token_logits", "duration_logits"]:
        raise PinnedComponentError("pinned joint output order differs")
    _constant_scalar(component, "encoder_frame_to_fp16_dtype_0", "fp16")
    _constant_scalar(component, "decoder_state_to_fp16_dtype_0", "fp16")
    _constant_vector(component, "var_14_begin_0", (0, 0))
    _constant_vector(component, "var_14_end_0", (1, 8193))
    _constant_vector(component, "var_14_end_mask_0", (True, False))
    _constant_scalar(component, "var_14_cast_fp16_to_fp32_dtype_0", "fp32")
    _constant_vector(component, "var_19_begin_0", (0, 8193))
    _constant_vector(component, "var_19_end_0", (1, 8198))
    _constant_vector(component, "var_19_end_mask_0", (True, True))
    _constant_scalar(component, "var_19_cast_fp16_to_fp32_dtype_0", "fp32")
    weight = component.constant("head_weight_to_fp16")
    bias = component.constant("head_bias_to_fp16")
    if weight.shape != (8198, 640) or str(weight.dtype) != "float16":
        raise PinnedComponentError("pinned joint head weight contract differs")
    if bias.shape != (8198,) or str(bias.dtype) != "float16":
        raise PinnedComponentError("pinned joint head bias contract differs")
    return _JointConstants(mx.array(weight), mx.array(bias))

def _validate_input(value, name: str, mx) -> None:
    if not isinstance(value, mx.array):
        raise TypeError(f"{name} must be an mx.array")
    if value.dtype != mx.float32:
        raise ValueError(f"{name} must have float32 dtype")
    if tuple(value.shape) != (1, 640):
        raise ValueError(f"{name} must have shape (1, 640)")


def run_joint(encoder_frame, decoder_state, *, package_path: Path) -> JointOutput:
    """Return named token and duration logits from two float32 ``[1,640]`` arrays."""
    mx = _mlx()
    _validate_input(encoder_frame, "encoder_frame", mx)
    _validate_input(decoder_state, "decoder_state", mx)
    with mx.stream(mx.gpu):
        constants = _load_joint(str(Path(package_path).resolve()))
        decoder_fp16 = decoder_state.astype(mx.float16)
        encoder_fp16 = encoder_frame.astype(mx.float16)
        hidden = mx.maximum(
            mx.add(encoder_fp16, decoder_fp16),
            mx.array(0, dtype=mx.float16),
        )
        combined = mx.add(mx.matmul(hidden, constants.weight.T), constants.bias)
        token_fp16 = combined[:, :8193]
        duration_fp16 = combined[:, 8193:8198]
        duration_logits = duration_fp16.astype(mx.float32)
        token_logits = token_fp16.astype(mx.float32)
    return JointOutput(token_logits, duration_logits)
