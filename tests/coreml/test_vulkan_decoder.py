# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Vulkan checks for the pinned Parakeet decoder package."""

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
from coreml.vulkan_decoder import (
    fused_lstm_numpy,
    load_decoder,
)
from coreml.vulkan_mel import trace_snapshot

LOCK = ReferenceLock.load(TOOLS / "coreml" / "parakeet-reference.lock")
PACKAGE = (
    default_cache_root()
    / LOCK.model_repo
    / LOCK.model_revision
    / "decoder.mlpackage"
)
MAX_ABS_ERROR = 5e-3
MEAN_ABS_ERROR = 2e-4


def _require_package():
    if not PACKAGE.is_dir():
        pytest.skip("pinned decoder.mlpackage is not installed")



def _mil_lstm(x, hidden, cell, weight_ih, weight_hh, bias):
    """Fused ``ios18.lstm`` contract: BNNS-blocked fp16 GEMV over the
    concatenated input, measured bit-indexed unaries, and the FMUL+FMA cell
    update (receipt 2026-09-15-bnns-lstm-re)."""
    return fused_lstm_numpy(x, hidden, cell, weight_ih, weight_hh, bias)

def _mil_reference(constants, input_ids, hidden, cell):
    indices = input_ids.astype(np.int16).astype(np.int32)
    indices = np.where(indices >= 0, indices, indices + 8193).astype(np.int16)
    sequence = constants["embedding_weight_to_fp16"][indices]
    hidden = hidden.astype(np.float16)
    cell = cell.astype(np.float16)

    sequence, hidden_0, cell_0 = _mil_lstm(
        sequence,
        hidden[0],
        cell[0],
        constants["concat_1_to_fp16"],
        constants["concat_2_to_fp16"],
        constants["concat_0_to_fp16"],
    )
    sequence, hidden_1, cell_1 = _mil_lstm(
        sequence,
        hidden[1],
        cell[1],
        constants["concat_4_to_fp16"],
        constants["concat_5_to_fp16"],
        constants["concat_3_to_fp16"],
    )
    sequence = sequence.transpose(1, 0, 2)
    decoder_hidden = (
        sequence.astype(np.float32)
        @ constants["projector_weight_to_fp16"].astype(np.float32).T
        + constants["projector_bias_to_fp16"].astype(np.float32)
    ).astype(np.float16).astype(np.float32)
    return (
        decoder_hidden,
        np.stack((hidden_0, hidden_1), axis=0).astype(np.float32),
        np.stack((cell_0, cell_1), axis=0).astype(np.float32),
    )


def _assert_close(actual, expected):
    delta = np.abs(actual - expected)
    assert float(delta.max()) <= MAX_ABS_ERROR
    assert float(delta.mean()) <= MEAN_ABS_ERROR


@pytest.mark.parametrize(
    ("which", "value", "message"),
    [
        ("input_ids", lambda: mx.zeros((1, 1), dtype=mx.int16), "int32"),
        ("input_ids", lambda: mx.zeros((1,), dtype=mx.int32), "shape"),
        ("hidden", lambda: mx.zeros((1, 1, 640), dtype=mx.float32), "shape"),
        ("hidden", lambda: mx.zeros((2, 1, 640), dtype=mx.float16), "float32"),
        ("cell", lambda: mx.zeros((2, 640), dtype=mx.float32), "shape"),
    ],
)
def test_decoder_rejects_invalid_input_contract(which, value, message):
    _require_package()
    decoder = load_decoder(PACKAGE)
    inputs = {
        "input_ids": mx.zeros((1, 1), dtype=mx.int32),
        "hidden": mx.zeros((2, 1, 640), dtype=mx.float32),
        "cell": mx.zeros((2, 1, 640), dtype=mx.float32),
    }
    inputs[which] = value()

    with pytest.raises(ValueError, match=message):
        decoder(**inputs)


def test_decoder_matches_official_mil_lstm_semantics_across_state_transitions():
    _require_package()
    component = load_pinned_component(PACKAGE, "decoder")
    names = (
        "embedding_weight_to_fp16",
        "concat_1_to_fp16",
        "concat_2_to_fp16",
        "concat_0_to_fp16",
        "concat_4_to_fp16",
        "concat_5_to_fp16",
        "concat_3_to_fp16",
        "projector_weight_to_fp16",
        "projector_bias_to_fp16",
    )
    constants = {name: component.constant(name) for name in names}
    decoder = load_decoder(PACKAGE)
    rng = np.random.default_rng(1847)
    cases = [
        (
            np.array([[0]], dtype=np.int32),
            np.zeros((2, 1, 640), dtype=np.float32),
            np.zeros((2, 1, 640), dtype=np.float32),
        ),
        (
            np.array([[8192]], dtype=np.int32),
            rng.normal(0, 0.1, (2, 1, 640)).astype(np.float32),
            rng.normal(0, 0.1, (2, 1, 640)).astype(np.float32),
        ),
    ]

    before = trace_snapshot()
    for input_ids, hidden, cell in cases:
        result = decoder(mx.array(input_ids), mx.array(hidden), mx.array(cell))
        mx.eval(*result)
        assert result.decoder_hidden.shape == (1, 1, 640)
        assert result.next_hidden.shape == (2, 1, 640)
        assert result.next_cell.shape == (2, 1, 640)
        assert all(value.dtype == mx.float32 for value in result)
        expected = _mil_reference(constants, input_ids, hidden, cell)
        for actual, reference in zip(result, expected):
            _assert_close(np.asarray(actual), reference)

    input_ids, hidden, cell = cases[0]
    first = decoder(mx.array(input_ids), mx.array(hidden), mx.array(cell))
    second = decoder(mx.array([[17]], dtype=mx.int32), first.next_hidden, first.next_cell)
    mx.eval(*second)
    expected_first = _mil_reference(constants, input_ids, hidden, cell)
    expected_second = _mil_reference(
        constants,
        np.array([[17]], dtype=np.int32),
        expected_first[1],
        expected_first[2],
    )
    for actual, reference in zip(second, expected_second):
        _assert_close(np.asarray(actual), reference)

    after = trace_snapshot()
    assert after["gpu_primitive_dispatches"] > before["gpu_primitive_dispatches"]
    assert after["vk_compute_dispatches"] > before["vk_compute_dispatches"]
    assert after["vk_submissions"] > before["vk_submissions"]


def test_decoder_preserves_mil_negative_embedding_index_semantics():
    _require_package()
    decoder = load_decoder(PACKAGE)
    hidden = mx.zeros((2, 1, 640), dtype=mx.float32)
    cell = mx.zeros((2, 1, 640), dtype=mx.float32)

    negative = decoder(mx.array([[-1]], dtype=mx.int32), hidden, cell)
    positive = decoder(mx.array([[8192]], dtype=mx.int32), hidden, cell)
    mx.eval(*negative, *positive)

    for negative_value, positive_value in zip(negative, positive):
        assert np.asarray(negative_value).tobytes() == np.asarray(positive_value).tobytes()
