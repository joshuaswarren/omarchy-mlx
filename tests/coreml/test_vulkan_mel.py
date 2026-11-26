# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""MLX Vulkan checks for the pinned Parakeet mel workload."""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

import vulkan_mel
from dft_geometry_probe import candidate_spectrum
from mel_reference import hann_window, mel_filterbank
from reference import ReferenceLock, default_cache_root

CAPTURE = (
    default_cache_root()
    / "captures"
    / "b650695c-75aec2a"
    / "20260912T154759Z-librispeech"
    / "ane"
)
STAGES = (
    "preemph",
    "hann",
    "mel_fb",
    "frames",
    "dft_real",
    "dft_imag",
    "power",
    "melproj",
    "logmel",
    "mean",
    "std",
)


@pytest.mark.parametrize(
    "waveform,message",
    [
        (lambda: mx.zeros((1,), dtype=mx.float16), "float32"),
        (lambda: mx.zeros((1, 1), dtype=mx.float32), "one-dimensional"),
        (lambda: mx.zeros((0,), dtype=mx.float32), "non-empty"),
        (lambda: mx.zeros((480001,), dtype=mx.float32), "480000"),
    ],
)
def test_vulkan_mel_rejects_invalid_waveform_contract(waveform, message):
    with pytest.raises(ValueError, match=message):
        vulkan_mel.extract_chunk_features(waveform())


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


def test_vulkan_dft_matches_certified_fma_bits():
    frames = _finite_fma_boundary_frame()
    expected_real, expected_imaginary = candidate_spectrum(
        frames, "vdsp-radix4-dif"
    )

    real, imaginary = vulkan_mel._dft_frames(mx.array(frames))
    mx.eval(real, imaginary)

    assert expected_real.view(np.uint32)[0, 1] == 0x3ECB7A0F
    assert np.asarray(real).tobytes() == expected_real.tobytes()
    assert np.asarray(imaginary).tobytes() == expected_imaginary.tobytes()

def test_vulkan_sqrt_matches_float32_reference_at_rounding_boundaries():
    bits = np.array(
        [0, 1, 2, 0x007FFFFF, 0x00800000, 0x01847B48,
         0x3F7FFFFF, 0x3F800000, 0x3F800001, 0x7F7FFFFF],
        dtype=np.uint32,
    )
    values = bits.view(np.float32)
    kernel = mx.fast.metal_kernel(
        name="parakeet_sqrt32_rounding_boundaries",
        input_names=["values"],
        output_names=["results"],
        header=vulkan_mel._SQRT_HEADER,
        source="""
            uint i = thread_position_in_grid.x;
            results[i] = sqrt32(values[i]);
        """,
        compile_options={"math_mode": "safe"},
    )
    actual = kernel(
        inputs=[mx.array(values)],
        output_shapes=[values.shape],
        output_dtypes=[mx.float32],
        grid=(values.size, 1, 1),
        threadgroup=(32, 1, 1),
        stream=mx.gpu,
    )[0]
    mx.eval(actual)
    expected = np.sqrt(values.astype(np.float64)).astype(np.float32)
    assert np.asarray(actual).tobytes() == expected.tobytes()


def test_vulkan_mel_constants_match_the_certified_reference():
    lock = ReferenceLock.load(TOOLS / "parakeet-reference.lock")
    constants = vulkan_mel._constant_arrays(mx)
    mx.eval(constants.hann, constants.filterbank)

    assert np.asarray(constants.hann).tobytes() == hann_window(400).tobytes()
    assert np.asarray(constants.filterbank).tobytes() == mel_filterbank(lock.mel).tobytes()


def test_pinned_waveform_all_vulkan_stages_and_encoder_inputs_are_bit_exact():
    stage_dir = (
        default_cache_root()
        / "captures"
        / "b650695c-75aec2a"
        / "mel-stage-probes"
        / "stage-capture"
    )
    if not CAPTURE.is_dir() or not stage_dir.is_dir():
        pytest.skip("pinned waveform or certified stage capture is not installed")

    waveform = mx.load(str(CAPTURE / "waveform.npy"))
    before = vulkan_mel.trace_snapshot()
    result = vulkan_mel.extract_chunk_features(waveform, capture_stages=True)
    mx.eval(result.mel, result.mask, result.encoder_features, result.encoder_mask)
    after = vulkan_mel.trace_snapshot()

    for name in STAGES:
        actual = np.asarray(result.stages[name])
        expected = np.load(stage_dir / f"{name}.npy", allow_pickle=False)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes(), name

    expected_mel = np.load(CAPTURE / "mel.npy", allow_pickle=False)
    expected_mask = np.load(CAPTURE / "mel_mask.npy", allow_pickle=False)
    expected_encoder = np.load(CAPTURE / "encoder_input_features.npy", allow_pickle=False)
    expected_encoder_mask = np.load(CAPTURE / "encoder_input_mask.npy", allow_pickle=False)
    assert np.asarray(result.mel).tobytes() == expected_mel.tobytes()
    assert np.asarray(result.mask).tobytes() == expected_mask.tobytes()
    assert np.asarray(result.encoder_features).tobytes() == expected_encoder.tobytes()
    assert np.asarray(result.encoder_mask).tobytes() == expected_encoder_mask.tobytes()
    assert after["gpu_primitive_dispatches"] > before["gpu_primitive_dispatches"]
    assert after["vk_compute_dispatches"] > before["vk_compute_dispatches"]


def _short_clip(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.arange(int(seconds * 16000), dtype=np.float32) / 16000
    tone = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 1210 * t)
    return (tone + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


def test_duration_aware_kernels_match_the_cpu_reference_on_short_audio():
    import mel_reference

    cfg = ReferenceLock.load(TOOLS / "parakeet-reference.lock").mel
    wave = _short_clip(1.5)
    result = vulkan_mel.extract_chunk_features(mx.array(wave), duration_aware=True)
    mx.eval(result.mel, result.mask)
    expected, expected_mask = mel_reference.extract_chunk_features(wave, cfg, duration_aware=True)

    assert result.valid_frames == int(expected_mask.sum()) == wave.size // 160 + 1
    assert np.asarray(result.mask).tobytes() == expected_mask.tobytes()
    assert not np.asarray(result.mel)[result.valid_frames :].any()
    assert np.asarray(result.mel).tobytes() == expected.tobytes()


def test_the_pinned_contract_keeps_all_frames_valid_on_short_audio():
    import mel_reference

    cfg = ReferenceLock.load(TOOLS / "parakeet-reference.lock").mel
    wave = _short_clip(1.5)
    result = vulkan_mel.extract_chunk_features(mx.array(wave))
    mx.eval(result.mel, result.mask)
    expected, expected_mask = mel_reference.extract_chunk_features(wave, cfg)

    assert result.valid_frames == vulkan_mel.N_FRAMES
    assert np.asarray(result.mask).tobytes() == expected_mask.tobytes()
    assert np.asarray(result.mel).tobytes() == expected.tobytes()
