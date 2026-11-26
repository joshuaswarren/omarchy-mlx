# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""MLX Vulkan checks for the pinned Parakeet joint package."""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "coreml"))

from coreml.pinned_component import load_pinned_component
from coreml.reference import ReferenceLock, default_cache_root
from coreml.vulkan_joint import run_joint
from coreml.vulkan_mel import trace_snapshot

LOCK = ReferenceLock.load(TOOLS / "coreml" / "parakeet-reference.lock")
CACHE = default_cache_root()
PACKAGE = CACHE / LOCK.model_repo / LOCK.model_revision / "joint.mlpackage"
ENCODER_CAPTURE = (
    CACHE
    / "captures"
    / "b650695c-75aec2a"
    / "20260912T154759Z-librispeech"
    / "ane"
    / "encoder_hidden.npy"
)


def _reference(encoder_frame, decoder_state):
    component = load_pinned_component(PACKAGE, "joint")
    weight = component.constant("head_weight_to_fp16")
    bias = component.constant("head_bias_to_fp16")
    hidden = np.maximum(
        np.add(
            encoder_frame.astype(np.float16),
            decoder_state.astype(np.float16),
            dtype=np.float16,
        ),
        np.float16(0),
    )
    logits = (hidden @ weight.T + bias).astype(np.float32)
    return logits[:, :8193], logits[:, 8193:8198]


@pytest.mark.parametrize(
    "encoder_frame,decoder_state,message",
    [
        (lambda: mx.zeros((1, 640), dtype=mx.float16), lambda: mx.zeros((1, 640), dtype=mx.float32), "encoder_frame.*float32"),
        (lambda: mx.zeros((640,), dtype=mx.float32), lambda: mx.zeros((1, 640), dtype=mx.float32), "encoder_frame.*\\(1, 640\\)"),
        (lambda: mx.zeros((1, 640), dtype=mx.float32), lambda: mx.zeros((1, 640), dtype=mx.float16), "decoder_state.*float32"),
        (lambda: mx.zeros((1, 640), dtype=mx.float32), lambda: mx.zeros((1, 1, 640), dtype=mx.float32), "decoder_state.*\\(1, 640\\)"),
    ],
)
def test_joint_rejects_wrong_named_input_contract(encoder_frame, decoder_state, message):
    with pytest.raises(ValueError, match=message):
        run_joint(encoder_frame(), decoder_state(), package_path=Path("missing"))


@pytest.mark.parametrize("source", ["deterministic", "licensed"])
def test_joint_matches_independent_fp_contract_on_vulkan(source):
    if not PACKAGE.is_dir():
        pytest.skip("pinned joint.mlpackage is not installed")
    if source == "licensed" and not ENCODER_CAPTURE.is_file():
        pytest.skip("licensed encoder capture is not installed")

    if source == "deterministic":
        encoder_frame = np.linspace(-0.75, 0.75, 640, dtype=np.float32)[None, :]
        decoder_state = np.linspace(0.25, -0.25, 640, dtype=np.float32)[None, :]
    else:
        encoder_frame = np.load(ENCODER_CAPTURE, allow_pickle=False)[:, 0, :]
        decoder_state = np.linspace(-0.125, 0.125, 640, dtype=np.float32)[None, :]
    expected_token, expected_duration = _reference(encoder_frame, decoder_state)

    before = trace_snapshot()
    result = run_joint(
        mx.array(encoder_frame), mx.array(decoder_state), package_path=PACKAGE
    )
    mx.eval(result.token_logits, result.duration_logits)
    after = trace_snapshot()
    actual_token = np.asarray(result.token_logits)
    actual_duration = np.asarray(result.duration_logits)

    assert actual_token.shape == (1, 8193)
    assert actual_duration.shape == (1, 5)
    assert actual_token.dtype == np.float32
    assert actual_duration.dtype == np.float32
    assert np.isfinite(actual_token).all()
    assert np.isfinite(actual_duration).all()
    assert actual_token.tobytes() == expected_token.tobytes()
    assert actual_duration.tobytes() == expected_duration.tobytes()
    assert after["gpu_primitive_dispatches"] > before["gpu_primitive_dispatches"]
    assert after["vk_compute_dispatches"] > before["vk_compute_dispatches"]


def test_joint_keeps_entire_graph_on_gpu_when_caller_defaults_to_cpu(tmp_path):
    if not PACKAGE.is_dir():
        pytest.skip("pinned joint.mlpackage is not installed")
    previous_default = mx.default_device()
    with mx.stream(mx.gpu):
        encoder_frame = mx.zeros((1, 640), dtype=mx.float32)
        decoder_state = mx.zeros((1, 640), dtype=mx.float32)
        mx.eval(encoder_frame, decoder_state)
    traces = []
    try:
        for label, device in (("gpu", mx.gpu), ("cpu", mx.cpu)):
            package = tmp_path / label / "joint.mlpackage"
            shutil.copytree(PACKAGE, package)
            mx.set_default_device(device)
            caller_default = mx.default_device()
            before = trace_snapshot()
            result = run_joint(
                encoder_frame,
                decoder_state,
                package_path=package,
            )
            mx.eval(result.token_logits, result.duration_logits)
            after = trace_snapshot()
            assert mx.default_device() == caller_default
            traces.append({
                key: after[key] - before[key]
                for key in (
                    "gpu_primitive_dispatches",
                    "vk_compute_dispatches",
                    "vk_submissions",
                    "commit_calls_with_work",
                )
            })
        assert traces[0]["gpu_primitive_dispatches"] > 0
        assert traces[1] == traces[0]
    finally:
        mx.set_default_device(previous_default)
    assert mx.default_device() == previous_default
