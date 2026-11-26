# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Host-side qualification checks for the pinned Vulkan mel workload."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

import vulkan_mel
from mel_stage_compare import STAGE_NAMES, verify_stage_evidence


FINAL_NAMES = ("mel", "mask", "encoder_features", "encoder_mask")


def _passing_inputs():
    comparisons = {
        name: {"bit_exact": True} for name in (*STAGE_NAMES, *FINAL_NAMES)
    }
    deltas = {
        "gpu_primitive_dispatches": 15,
        "vk_compute_dispatches": 7,
        "vk_submissions": 1,
    }
    return comparisons, deltas


def test_qualification_requires_the_exact_expected_stage_set():
    comparisons, deltas = _passing_inputs()
    missing = STAGE_NAMES[:-1]
    extra = (*STAGE_NAMES, "unexpected")

    missing_status = vulkan_mel._qualification_status(
        comparisons, missing, STAGE_NAMES, deltas
    )
    extra_status = vulkan_mel._qualification_status(
        comparisons, extra, STAGE_NAMES, deltas
    )

    assert missing_status["stage_set_exact"] is False
    assert missing_status["qualified"] is False
    assert extra_status["stage_set_exact"] is False
    assert extra_status["qualified"] is False


@pytest.mark.parametrize("dispatches", [1, 6, 8])
def test_qualification_requires_exactly_seven_custom_compute_dispatches(dispatches):
    comparisons, deltas = _passing_inputs()
    deltas["vk_compute_dispatches"] = dispatches

    status = vulkan_mel._qualification_status(
        comparisons, STAGE_NAMES, STAGE_NAMES, deltas
    )

    assert status["gpu_execution"]["vk_compute_dispatches"] is False
    assert status["qualified"] is False


def test_qualification_accepts_exact_stages_comparisons_and_dispatch_count():
    comparisons, deltas = _passing_inputs()

    status = vulkan_mel._qualification_status(
        comparisons, STAGE_NAMES, STAGE_NAMES, deltas
    )

    assert status == {
        "all_bit_exact": True,
        "stage_set_exact": True,
        "gpu_execution": {
            "gpu_primitive_dispatches": True,
            "vk_compute_dispatches": True,
            "vk_submissions": True,
        },
        "qualified": True,
    }


def test_comparison_stages_normalize_the_completed_api_mask_on_the_host():
    api_mask = np.ones((vulkan_mel.N_FRAMES,), dtype=np.int32)
    mel = object()
    result = SimpleNamespace(stages={"waveform": object()}, mask=api_mask, mel=mel)

    stages = vulkan_mel._comparison_stages(result, np)

    assert stages["mel_mask"].dtype == np.float32
    np.testing.assert_array_equal(stages["mel_mask"], api_mask)
    assert result.mask is api_mask
    assert result.mask.dtype == np.int32
    assert stages["mel_pinned"] is mel
    assert stages["mel_stepwise"] is mel


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.ones((vulkan_mel.N_FRAMES,), dtype=np.float32), "int32"),
        (np.ones((vulkan_mel.N_FRAMES, 1), dtype=np.int32), "shape"),
    ],
)
def test_comparison_stages_reject_invalid_api_masks(mask, message):
    result = SimpleNamespace(stages={}, mask=mask, mel=object())

    with pytest.raises(ValueError, match=message):
        vulkan_mel._comparison_stages(result, np)


def test_stage_evidence_authenticates_every_final_capture(tmp_path):
    capture_dir = tmp_path / "capture"
    dumps_dir = tmp_path / "stages"
    capture_dir.mkdir()
    dumps_dir.mkdir()
    paths = {}

    for name in (
        "waveform.npy",
        "mel.npy",
        "mel_mask.npy",
        "encoder_input_features.npy",
        "encoder_input_mask.npy",
    ):
        path = capture_dir / name
        path.write_bytes(name.encode())
        paths[name] = hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {}
    for name in STAGE_NAMES:
        path = dumps_dir / f"{name}.npy"
        path.write_bytes(name.encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest[name] = digest
        paths[f"mel_stage/{name}.npy"] = digest

    manifest_path = dumps_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    paths["mel_stage_manifest.json"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    lock = SimpleNamespace(macos_reference_paths=paths)

    verify_stage_evidence(capture_dir, dumps_dir, lock)
    (capture_dir / "encoder_input_mask.npy").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="encoder_input_mask.npy"):
        verify_stage_evidence(capture_dir, dumps_dir, lock)
