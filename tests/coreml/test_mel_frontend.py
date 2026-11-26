# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Exact pinned-audio frontend checks against the macOS Parakeet fixture."""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

import dft_geometry_probe
import mel_reference
from mel_reference import DivergenceReport, compare_capture, hann_window
from mel_stage_compare import stage_diff, verify_stage_evidence
from reference import default_cache_root

CAPTURE = (
    default_cache_root()
    / "captures"
    / "b650695c-75aec2a"
    / "20260912T154759Z-librispeech"
    / "ane"
)


def test_pinned_audio_to_mel_is_bit_exact():
    if not CAPTURE.is_dir():
        pytest.skip(f"pinned macOS capture is not installed at {CAPTURE}")

    report = compare_capture(CAPTURE)

    assert report.exact
    assert report.bit_exact_count == report.total == 3001 * 128
    assert all(report.structural.values())


def test_hann_window_rejects_unqualified_lengths():
    with pytest.raises(ValueError, match="pinned 400-sample Hann window"):
        hann_window(512)


def test_mel_cli_fails_when_a_structural_gate_fails(monkeypatch):
    report = DivergenceReport(
        exact=True,
        shape_expected=(1, 1),
        shape_actual=(1, 1),
        bit_exact_count=1,
        total=1,
        max_abs=0.0,
        mean_abs=0.0,
        first_divergence=None,
        structural={"mask_equals_golden": False},
    )
    monkeypatch.setattr(mel_reference, "compare_capture", lambda _: report)

    assert mel_reference.main(["mel_reference.py", "capture"]) == 1


def test_mel_cli_rejects_wrong_mask_dtype(monkeypatch):
    if not CAPTURE.is_dir():
        pytest.skip(f"pinned macOS capture is not installed at {CAPTURE}")
    extract = mel_reference.extract_chunk_features

    def boolean_mask(waveform, config):
        features, mask = extract(waveform, config)
        return features, mask.astype(np.bool_)

    monkeypatch.setattr(mel_reference, "extract_chunk_features", boolean_mask)
    assert mel_reference.main(["mel_reference.py", str(CAPTURE)]) == 1


@pytest.mark.parametrize(
    "computed,golden,diagnostic",
    [
        (
            np.arange(4, dtype=np.float32),
            np.arange(4, dtype=np.float32).reshape(2, 2),
            "shape mismatch",
        ),
        (
            np.arange(4, dtype=np.float32),
            np.arange(4, dtype=np.float64),
            "dtype mismatch",
        ),
    ],
)
def test_stage_diff_rejects_tensor_contract_mismatch(computed, golden, diagnostic):
    exact, message = stage_diff("stage", computed, golden)

    assert not exact
    assert diagnostic in message.lower()


def test_stage_diff_reports_signed_zero_bit_mismatch_without_crashing():
    exact, message = stage_diff(
        "stage",
        np.asarray([-0.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
    )

    assert not exact
    assert "first bitwise divergence" in message.lower()


def _finite_fma_boundary_frame() -> np.ndarray:
    def value(bits):
        return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]

    a = value(0x3FC00000)
    c = value(0x21800000)
    post = np.zeros(256, dtype=np.complex64)
    post.reshape(4, 16, 4)[0, 0, 1] = np.complex64(c)
    post.reshape(4, 16, 4)[1, 0, 1] = np.complex64(a)
    y = post.reshape(64, 4).astype(np.complex128)
    x = np.empty_like(y)
    x[:, 0] = (y[:, 0] + y[:, 1] + y[:, 2] + y[:, 3]) / 4
    x[:, 1] = (y[:, 0] + 1j * y[:, 1] - y[:, 2] - 1j * y[:, 3]) / 4
    x[:, 2] = (y[:, 0] - y[:, 1] + y[:, 2] - y[:, 3]) / 4
    x[:, 3] = (y[:, 0] - 1j * y[:, 1] - y[:, 2] + 1j * y[:, 3]) / 4
    values = x.T.reshape(1, 256).astype(np.complex64)
    frames = np.empty((1, 512), dtype=np.float32)
    frames[:, 0::2] = values.real
    frames[:, 1::2] = values.imag
    return frames


def test_candidate_spectrum_uses_single_rounded_float32_fma():
    real, _ = dft_geometry_probe.candidate_spectrum(
        _finite_fma_boundary_frame(), "vdsp-radix4-dif"
    )

    assert real.view(np.uint32)[0, 1] == 0x3ECB7A0F


def test_candidate_spectrum_preserves_negative_zero_bits():
    frames = np.zeros((1, 512), dtype=np.float32)
    frames[0, 0] = np.float32(-0.0)

    _, imaginary = dft_geometry_probe.candidate_spectrum(
        frames, "vdsp-radix4-dif"
    )

    assert imaginary.view(np.uint32)[0, 171] == 0x80000000


@pytest.mark.parametrize(
    "frames,message",
    [
        (np.zeros(512, dtype=np.float32), "two-dimensional"),
        (np.zeros((1, 2048), dtype=np.float32), "512"),
        (np.zeros((1, 512), dtype=np.float64), "float32"),
    ],
)
def test_candidate_spectrum_rejects_invalid_frame_contract(frames, message):
    with pytest.raises(ValueError, match=message):
        dft_geometry_probe.candidate_spectrum(frames, "vdsp-radix4-dif")


def test_stage_evidence_rejects_an_unpinned_capture(tmp_path):
    capture = tmp_path / "capture"
    dumps = tmp_path / "dumps"
    capture.mkdir()
    dumps.mkdir()
    (capture / "waveform.npy").write_bytes(b"not the pinned waveform")

    class Lock:
        macos_reference_paths = {"waveform.npy": "0" * 64}

    with pytest.raises(ValueError, match="waveform.npy"):
        verify_stage_evidence(capture, dumps, Lock())


def test_stage_evidence_rejects_an_unpinned_manifest(tmp_path):
    capture = tmp_path / "capture"
    dumps = tmp_path / "dumps"
    capture.mkdir()
    dumps.mkdir()
    contents = {
        "waveform.npy": b"waveform",
        "mel.npy": b"mel",
        "mel_mask.npy": b"mel_mask",
        "encoder_input_features.npy": b"features",
        "encoder_input_mask.npy": b"mask",
    }
    for name, blob in contents.items():
        (capture / name).write_bytes(blob)
    (dumps / "manifest.json").write_text("{}", encoding="utf-8")

    class Lock:
        macos_reference_paths = {
            name: hashlib.sha256(blob).hexdigest()
            for name, blob in contents.items()
        }
        macos_reference_paths["mel_stage_manifest.json"] = "0" * 64

    with pytest.raises(ValueError, match="stage manifest"):
        verify_stage_evidence(capture, dumps, Lock())
