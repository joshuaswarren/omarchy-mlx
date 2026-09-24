# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Full-range float32 arithmetic checks for the Vulkan mel workload."""

import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

import vulkan_mel
from dft_geometry_probe import candidate_spectrum
from dotpr import _fma32, mel_projection
from mel_reference import chunk_for_mel, stft_frames
from reference import ReferenceLock


def _gpu_fma(
    lhs, rhs, acc, *, header=vulkan_mel._FMA_HEADER, function="fma32"
):
    kernel = mx.fast.metal_kernel(
        name="parakeet_fma32_full_range",
        input_names=["lhs", "rhs", "acc"],
        output_names=["result"],
        header=header,
        source=f"""
            uint i = thread_position_in_grid.x;
            result[i] = {function}(lhs[i], rhs[i], acc[i]);
        """,
        compile_options={"math_mode": "safe"},
    )
    result = kernel(
        inputs=[mx.array(lhs), mx.array(rhs), mx.array(acc)],
        output_shapes=[lhs.shape],
        output_dtypes=[mx.float32],
        grid=(lhs.size, 1, 1),
        threadgroup=(256, 1, 1),
        stream=mx.gpu,
    )[0]
    mx.eval(result)
    return np.asarray(result)


def _gpu_bit_binary(lhs, rhs, function):
    kernel = mx.fast.metal_kernel(
        name=f"parakeet_{function}_full_range",
        input_names=["lhs", "rhs"],
        output_names=["result"],
        header=vulkan_mel._FMA_HEADER,
        source=f"""
            uint i = thread_position_in_grid.x;
            result[i] = {function}(lhs[i], rhs[i]);
        """,
        compile_options={"math_mode": "safe"},
    )
    result = kernel(
        inputs=[mx.array(lhs.view(np.uint32)), mx.array(rhs.view(np.uint32))],
        output_shapes=[lhs.shape],
        output_dtypes=[mx.uint32],
        grid=(lhs.size, 1, 1),
        threadgroup=(256, 1, 1),
        stream=mx.gpu,
    )[0]
    mx.eval(result)
    return np.asarray(result).view(np.float32)


def _finite_randoms(seed, size):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2**32, size=size, dtype=np.uint32)
    exponents = rng.integers(0, 255, size=size, dtype=np.uint32)
    bits = (bits & np.uint32(0x807FFFFF)) | (exponents << np.uint32(23))
    return bits.view(np.float32)


def _assert_same_bits(actual, expected):
    different = actual.view(np.uint32) != expected.view(np.uint32)
    count = int(different.sum())
    if count:
        first = tuple(int(index) for index in np.argwhere(different)[0])
        pytest.fail(
            f"{count} binary32 mismatches; first={first} "
            f"actual=0x{int(actual.view(np.uint32)[first]):08x} "
            f"expected=0x{int(expected.view(np.uint32)[first]):08x}"
        )


def test_fma32_matches_the_finite_binary32_contract_across_all_exponents():
    lhs = np.concatenate(
        (
            _finite_randoms(916, 8_192),
            np.array(
                [
                    0x00000000, 0x80000000, 0x00000001, 0x00000001,
                    0x00800000, 0x7F7FFFFF, 0x7F7FFFFF, 0x3F800000,
                    0xBF800000,
                ],
                dtype=np.uint32,
            ).view(np.float32),
        )
    )
    rhs = np.concatenate(
        (
            _finite_randoms(917, 8_192),
            np.array(
                [
                    0x80000000, 0x00000000, 0x3F800000, 0x3F000000,
                    0x00800000, 0x40000000, 0x40000000, 0x3F800000,
                    0x3F800000,
                ],
                dtype=np.uint32,
            ).view(np.float32),
        )
    )
    acc = np.concatenate(
        (
            _finite_randoms(918, 8_192),
            np.array(
                [
                    0x80000000, 0x80000000, 0x00000000, 0x00000000,
                    0x00000000, 0xFF7FFFFF, 0x00000000, 0xBF800000,
                    0x3F800000,
                ],
                dtype=np.uint32,
            ).view(np.float32),
        )
    )
    expected = _fma32(lhs, rhs, acc)

    actual = _gpu_fma(lhs, rhs, acc)

    _assert_same_bits(actual, expected)


def test_native_fma_does_not_meet_the_exact_custom_kernel_contract():
    lhs = np.array([0x5A58D667], dtype=np.uint32).view(np.float32)
    rhs = np.array([0x2C6C9263], dtype=np.uint32).view(np.float32)
    acc = np.array([0xBE6A1737], dtype=np.uint32).view(np.float32)

    expected = _fma32(lhs, rhs, acc)
    actual = _gpu_fma(lhs, rhs, acc, header="", function="fma")

    assert int(expected.view(np.uint32)[0]) == 0x4748616B
    assert int(actual.view(np.uint32)[0]) == 0x4748616A


def test_bit_arithmetic_preserves_finite_subnormals_and_signed_zero():
    lhs = np.array(
        [0x00000001, 0x007FFFFF, 0x00800000, 0x80000000, 0x00000000,
         0x7F7FFFFF, 0xFF7FFFFF],
        dtype=np.uint32,
    ).view(np.float32)
    rhs = np.array(
        [0x3F800000, 0x3F000000, 0x3F000000, 0x3F800000, 0xBF800000,
         0x40000000, 0x40000000],
        dtype=np.uint32,
    ).view(np.float32)
    with np.errstate(over="ignore"):
        expected_add = np.add(lhs, rhs, dtype=np.float32)
        expected_mul = np.multiply(lhs, rhs, dtype=np.float32)

    _assert_same_bits(_gpu_bit_binary(lhs, rhs, "add32_bits"), expected_add)
    _assert_same_bits(_gpu_bit_binary(lhs, rhs, "mul32_bits"), expected_mul)


def test_preemphasis_preserves_subnormal_results():
    lock = ReferenceLock.load(TOOLS / "parakeet-reference.lock")
    rng = np.random.default_rng(916)
    waveform = (
        rng.uniform(-1, 1, vulkan_mel.CHUNK_SAMPLES) * np.float32(2.0**-120)
    ).astype(np.float32)
    expected = np.empty_like(waveform)
    expected[0] = waveform[0]
    expected[1:] = (
        waveform[1:] - np.float32(lock.mel.preemphasis) * waveform[:-1]
    ).astype(np.float32)

    actual = vulkan_mel._preemphasize(mx.array(waveform))
    mx.eval(actual)

    _assert_same_bits(np.asarray(actual), expected)


def test_frame_windowing_preserves_subnormal_products():
    lock = ReferenceLock.load(TOOLS / "parakeet-reference.lock")
    rng = np.random.default_rng(917)
    waveform = (
        rng.uniform(-1, 1, vulkan_mel.CHUNK_SAMPLES) * np.float32(2.0**-120)
    ).astype(np.float32)
    preemphasis = np.empty_like(waveform)
    preemphasis[0] = waveform[0]
    preemphasis[1:] = (
        waveform[1:] - np.float32(lock.mel.preemphasis) * waveform[:-1]
    ).astype(np.float32)
    hann = np.asarray(vulkan_mel._constant_floats(), dtype=np.float32)[
        vulkan_mel.HANN_OFFSET:vulkan_mel.HANN_OFFSET + 400
    ]
    indices = (
        np.arange(vulkan_mel.N_FRAMES, dtype=np.int32)[:, None] * 160
        + np.arange(400, dtype=np.int32)[None, :]
        - 256
    )
    valid = (indices >= 0) & (indices < vulkan_mel.CHUNK_SAMPLES)
    windowed = np.zeros(indices.shape, dtype=np.float32)
    windowed[valid] = preemphasis[indices[valid]]
    expected = np.zeros((vulkan_mel.N_FRAMES, vulkan_mel.N_FFT), dtype=np.float32)
    expected[:, 56:456] = (windowed * hann[None, :]).astype(np.float32)

    actual = vulkan_mel._frames(mx.array(waveform), mx.array(hann))
    mx.eval(actual)

    _assert_same_bits(np.asarray(actual), expected)


def _reference_frames(scale):
    lock = ReferenceLock.load(TOOLS / "parakeet-reference.lock")
    rng = np.random.default_rng(916)
    waveform = (rng.uniform(-1, 1, 512) * np.float32(scale)).astype(np.float32)
    return stft_frames(waveform, lock.mel)


def test_dft_preserves_subnormal_fma_terms():
    frames = _reference_frames(2.0**-120)
    expected_real, expected_imaginary = candidate_spectrum(
        frames, "vdsp-radix4-dif"
    )

    actual_real, actual_imaginary = vulkan_mel._dft_frames(mx.array(frames))
    mx.eval(actual_real, actual_imaginary)

    _assert_same_bits(np.asarray(actual_real), expected_real)
    _assert_same_bits(np.asarray(actual_imaginary), expected_imaginary)


def test_low_scale_waveform_preserves_power_and_mel_subnormals():
    lock = ReferenceLock.load(TOOLS / "parakeet-reference.lock")
    rng = np.random.default_rng(916)
    waveform = (
        rng.uniform(-1, 1, 512) * np.float32(2.0**-70)
    ).astype(np.float32)
    frames = stft_frames(chunk_for_mel(waveform), lock.mel)
    expected_real, expected_imaginary = candidate_spectrum(
        frames, "vdsp-radix4-dif"
    )
    magnitude = np.sqrt(
        expected_real * expected_real + expected_imaginary * expected_imaginary
    ).astype(np.float32)
    expected_power = (magnitude * magnitude).astype(np.float32)
    positive_subnormal = (expected_power.view(np.uint32) > 0) & (
        expected_power.view(np.uint32) < 0x00800000
    )
    assert np.any(positive_subnormal)
    filterbank = np.asarray(vulkan_mel._constant_floats(), np.float32)[
        vulkan_mel.FILTERBANK_OFFSET:
    ].reshape(vulkan_mel.N_MELS, vulkan_mel.N_BINS)
    expected_mel_projection = mel_projection(filterbank, expected_power)

    constants = vulkan_mel._constant_arrays(mx)
    actual_frames = vulkan_mel._frames(mx.array(waveform), constants.hann)
    actual_real, actual_imaginary = vulkan_mel._dft_frames(actual_frames)
    actual_power = vulkan_mel._power(actual_real, actual_imaginary)
    actual_mel_projection, _ = vulkan_mel._mel_project(
        actual_power, constants.filterbank
    )
    mx.eval(actual_power, actual_mel_projection)

    _assert_same_bits(np.asarray(actual_power), expected_power)
    _assert_same_bits(np.asarray(actual_mel_projection), expected_mel_projection)


@pytest.mark.parametrize("scale", [2.0**-60, 2.0**53])
def test_mel_projection_handles_extreme_finite_waveform_scales(scale):
    frames = _reference_frames(scale)
    dft_real, dft_imaginary = candidate_spectrum(frames, "vdsp-radix4-dif")
    magnitude = np.sqrt(
        dft_real.astype(np.float64) ** 2
        + dft_imaginary.astype(np.float64) ** 2
    ).astype(np.float32)
    power = (magnitude * magnitude).astype(np.float32)
    filterbank = np.asarray(vulkan_mel._constant_floats(), np.float32)[
        vulkan_mel.FILTERBANK_OFFSET:
    ].reshape(vulkan_mel.N_MELS, vulkan_mel.N_BINS)
    expected = mel_projection(filterbank, power)

    actual, _ = vulkan_mel._mel_project(mx.array(power), mx.array(filterbank))
    mx.eval(actual)

    _assert_same_bits(np.asarray(actual), expected)
