# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Exact parity tests for the portable vDSP_dotpr reduction."""

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools" / "coreml"
sys.path.insert(0, str(TOOLS))

from dotpr import mel_projection, portable_dotpr
from generate_dotpr_probes import probe_cases


MACOS_RESULTS = {
    "fma-two-term": 0xBDE9787B,
    "double-rounding-boundary": 0x3FC00005,
    "overflow-midpoint-boundary": 0x7F7FFFFF,
    "vector-four": 0x423CABE1,
    "association-eight": 0x450C31FC,
    "vector-tail-seventeen": 0x44252851,
    "production-length-257": 0xC7D65DF8,
    "signed-zero": 0x00000000,
}


def floats(bits: list[int]) -> np.ndarray:
    return np.asarray(bits, dtype=np.uint32).view(np.float32)


@pytest.mark.parametrize("name,lhs_bits,rhs_bits", probe_cases())
def test_portable_dotpr_matches_macos_vdsp_bits(name, lhs_bits, rhs_bits):
    actual = portable_dotpr(floats(lhs_bits), floats(rhs_bits))
    assert actual.view(np.uint32).item() == MACOS_RESULTS[name]


def test_mel_projection_matches_independent_macos_vdsp_matrix_bits():
    filterbank = np.stack([
        floats([0x4115B804, 0x3DBE3ACE, 0xBD94FC9D, 0x42EED71E]),
        floats([0xBBA12958, 0x40C6ADFD, 0xC05058A1, 0x3EC6EFA3]),
    ])
    power = filterbank.copy()
    actual = mel_projection(filterbank, power)
    expected = np.asarray(
        [[0x466032FE, 0x423CABE1], [0x423CABE1, 0x42453040]],
        dtype=np.uint32,
    )
    assert np.array_equal(actual.view(np.uint32), expected)


@pytest.mark.parametrize(
    "lhs,rhs,message",
    [
        (np.ones(2, dtype=np.float64), np.ones(2, dtype=np.float32), "float32"),
        (np.ones((1, 2), dtype=np.float32), np.ones(2, dtype=np.float32), "one-dimensional"),
        (np.ones(2, dtype=np.float32), np.ones(3, dtype=np.float32), "same length"),
        (np.ones(0, dtype=np.float32), np.ones(0, dtype=np.float32), "non-empty"),
    ],
)
def test_portable_dotpr_rejects_invalid_vectors(lhs, rhs, message):
    with pytest.raises(ValueError, match=message):
        portable_dotpr(lhs, rhs)
