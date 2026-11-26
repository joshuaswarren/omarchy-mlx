# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Correctly-rounded fp16 sigmoid/tanh versus macOS Core ML CPU bins."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))

from coreml.vulkan_decoder import fp16_sigmoid, fp16_tanh

ROOT = Path(__file__).resolve().parents[2]
MAC = ROOT / "receipts/2026-09-14-decoder-activation-capture-mac/mac"
ANE = ROOT / "receipts/2026-09-14-decoder-activations-on-ane/device"
F16 = np.dtype("<f2")


def _require_bins():
    if not (MAC / "y_cpu_only_sigmoid.bin").is_file():
        pytest.skip("Mac CPU activation bins are not present")
    if not (ANE / "x_sigmoid_0.bin").is_file():
        pytest.skip("decoder activation argument bins are not present")


def _load_x(op: str) -> np.ndarray:
    return np.concatenate(
        [np.fromfile(ANE / f"x_{op}_{p}.bin", F16) for p in range(3)]
    )


def _fp16_chain_sigmoid(x) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore"):
        exp = np.exp(-x64).astype(F16).astype(np.float64)
        return (1.0 / (1.0 + exp).astype(F16).astype(np.float64)).astype(F16)


def test_fp16_sigmoid_tanh_fixed_points():
    assert fp16_sigmoid(np.float16(0)) == np.float16(0.5)
    assert fp16_tanh(np.float16(0)) == np.float16(0)
    assert fp16_sigmoid(np.float16(20)) == np.float16(1)
    assert fp16_sigmoid(np.float16(-20)) == np.float16(0)
    assert fp16_tanh(np.float16(20)) == np.float16(1)
    assert fp16_tanh(np.float16(-20)) == np.float16(-1)
    x = np.array([-0.5, 0.25, 1.5], dtype=F16)
    assert np.array_equal(fp16_tanh(-x), -fp16_tanh(x))


@pytest.mark.parametrize("op,fn", [("sigmoid", fp16_sigmoid), ("tanh", fp16_tanh)])
def test_fp16_activations_match_mac_cpu_1280_and_1536(op, fn):
    _require_bins()
    layout = json.loads((ANE / "layout.json").read_text())
    x = _load_x(op)
    y = np.fromfile(MAC / f"y_cpu_only_{op}.bin", F16)
    lo, hi = layout[op]["args"]
    pred = fn(x)
    assert hi - lo == 1280
    assert int(np.count_nonzero(pred[lo:hi] == y[lo:hi])) == 1280
    assert int(np.count_nonzero(pred == y)) == 1536
    assert float(np.max(np.abs(pred[lo:hi].astype(np.float64) - y[lo:hi].astype(np.float64)))) == 0.0


def test_fp16_chain_and_h13_lut_are_not_the_mac_cpu_contract():
    _require_bins()
    layout = json.loads((ANE / "layout.json").read_text())
    lo, hi = layout["sigmoid"]["args"]
    x_s = _load_x("sigmoid")
    y_s = np.fromfile(MAC / "y_cpu_only_sigmoid.bin", F16)
    y_t = np.fromfile(MAC / "y_cpu_only_tanh.bin", F16)
    h13_s = np.concatenate(
        [np.fromfile(ANE / f"y_out_sigmoid_{p}.bin", F16) for p in range(3)]
    )
    h13_t = np.concatenate(
        [np.fromfile(ANE / f"y_out_tanh_{p}.bin", F16) for p in range(3)]
    )
    chain = _fp16_chain_sigmoid(x_s[lo:hi])
    assert int(np.count_nonzero(chain == y_s[lo:hi])) == 899
    assert int(np.count_nonzero(h13_s[lo:hi] == y_s[lo:hi])) == 116
    assert int(np.count_nonzero(h13_t[lo:hi] == y_t[lo:hi])) == 99
