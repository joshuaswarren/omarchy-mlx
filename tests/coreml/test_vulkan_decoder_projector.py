# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""The decoder projector kernel against native Core ML, lane for lane.

The projector's native arithmetic is established: an fp16 accumulator over
unrounded products, reduced in strictly ascending index order, with the fp16
bias added after the reduction. These checks hold the kernel to that contract
bit-exactly on every captured transition, show that the capture tells the
contract apart from its nearest neighbour, and hold the fp16 decode to the
format rather than to the device.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

TOOLS = Path(__file__).resolve().parents[2] / "overlay" / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS / "coreml"))

from coreml.reference import ReferenceLock, default_cache_root
from coreml.vulkan_decoder import (
    _HIDDEN_SIZE,
    _PROJECTOR_GROUP,
    _project,
    load_decoder,
)

LOCK = ReferenceLock.load(TOOLS / "coreml" / "parakeet-reference.lock")
PACKAGE = (
    default_cache_root() / LOCK.model_repo / LOCK.model_revision / "decoder.mlpackage"
)
# The authenticated capture that established the contract, pinned by content so
# an unrelated capture cannot quietly stand in for it.
CAPTURE_SHA256 = "6eaf2bfe801584af570cd66f42e8dc17efd67776fb181f691585d2bcab590e0a"
# An fp16 multiply followed by an fp16 add, scored on the same lanes. Measured
# in NumPy on the host and on the GPU, independently.
ROUNDED_PRODUCT_LANES = 4535


def _capture_dir():
    root = default_cache_root() / "captures"
    for candidate in sorted(root.glob("*/*/*/tdt_tensors.json")):
        if hashlib.sha256(candidate.read_bytes()).hexdigest() == CAPTURE_SHA256:
            return candidate.parent
    return None


def _require_transitions():
    """Captured projector input and native output, one row per transition."""
    _require_package()
    capture = _capture_dir()
    if capture is None:
        pytest.skip("the authenticated TDT tensor capture is not installed")
    index = json.loads((capture / "tdt_tensors.json").read_text())
    rows = []
    for trace in index["traces"]:
        paths = trace["tensor_paths"]
        if "decoder_next_hidden" not in paths or "decoder_output_hidden" not in paths:
            continue  # decoder-reuse transition: no recurrent step ran
        rows.append(
            (
                int(trace["index"]),
                mx.array(
                    np.load(capture / paths["decoder_next_hidden"])[1, 0]
                    .astype(np.float16)
                    .reshape(1, _HIDDEN_SIZE)
                ),
                np.load(capture / paths["decoder_output_hidden"])[0, 0],
            )
        )
    assert len(rows) == 15
    return rows


def _require_package():
    if not PACKAGE.is_dir():
        pytest.skip("pinned decoder.mlpackage is not installed")


class _Codes:
    """Just enough of the loaded weight tuple to drive ``_project``."""

    def __init__(self, weight, bias):
        self.projector_codes = mx.array(np.ascontiguousarray(weight.T).view(np.uint16))
        self.projector_bias_codes = mx.array(bias.view(np.uint16))


def test_projector_is_bit_exact_with_native_on_every_transition():
    """The shipped kernel, driven by the decoder's own loaded weights."""
    rows = _require_transitions()
    weights = load_decoder(PACKAGE)._weights

    for index, hidden, native in rows:
        projected = _project(hidden, weights, mx)
        mx.eval(projected)
        observed = np.asarray(projected)
        assert observed.shape == (1, 1, _HIDDEN_SIZE)
        assert observed.dtype == np.float32
        delta = observed.reshape(-1) - native
        assert int(np.count_nonzero(delta)) == 0, (
            f"transition {index}: {int(np.count_nonzero(delta))} of {_HIDDEN_SIZE} "
            f"lanes differ from native, worst {np.abs(delta).max()}"
        )


def test_the_capture_separates_the_contract_from_rounded_products():
    """Without this, the check above could pass on any fp16 reduction.

    Rounding each product before the add is the nearest neighbouring contract
    and the one a plain ``float16_t`` kernel would compile to. On these lanes it
    reproduces 4535 of 9600, so the bit-exact result above is a property of the
    arithmetic and not of the data.
    """
    rows = _require_transitions()
    from coreml.pinned_component import load_pinned_component

    component = load_pinned_component(PACKAGE, "decoder")
    weight = mx.array(
        np.ascontiguousarray(component.constant("projector_weight_to_fp16").T)
    )
    bias = mx.array(component.constant("projector_bias_to_fp16"))
    # `precise` keeps the compiler from contracting the multiply and the add
    # into the fused multiply-add that native actually uses.
    rounded = mx.fast.metal_kernel(
        name="parakeet_projector_rounded_products",
        input_names=["hidden", "weight", "bias"],
        output_names=["projected"],
        source="""
            uint column = thread_position_in_grid.x;
            float16_t accumulator = float16_t(0.0);
            for (uint k = 0u; k < 640u; ++k) {
                precise float16_t product = hidden[k]
                    * weight[k * 640u + column];
                precise float16_t total = accumulator + product;
                accumulator = total;
            }
            projected[column] = accumulator + bias[column];
        """,
        compile_options={"math_mode": "safe"},
    )

    exact = 0
    for _index, hidden, native in rows:
        projected = rounded(
            inputs=[hidden, weight, bias],
            output_shapes=[(1, 1, _HIDDEN_SIZE)],
            output_dtypes=[mx.float16],
            grid=(_HIDDEN_SIZE, 1, 1),
            threadgroup=(_PROJECTOR_GROUP, 1, 1),
            stream=mx.gpu,
        )[0]
        mx.eval(projected)
        exact += int(
            np.count_nonzero(
                np.asarray(projected).reshape(-1).astype(np.float64) == native
            )
        )
    assert exact == ROUNDED_PRODUCT_LANES


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("smallest subnormal", 2.0**-24),
        ("negative smallest subnormal", -(2.0**-24)),
        ("largest subnormal", 1023 * 2.0**-24),
        ("smallest normal", 2.0**-14),
        ("largest finite", 65504.0),
    ],
)
def test_the_kernel_decodes_fp16_edge_values_from_the_format(name, value):
    """One non-zero weight lane reduces the whole reduction to one decode.

    The pinned projector weight carries 437 fp16 denormals and Vulkan permits a
    driver to flush those, so the kernel decodes the bit pattern in integer
    arithmetic rather than converting. These are the values that would expose a
    flush or a wrong decode.
    """
    weight = np.zeros((_HIDDEN_SIZE, _HIDDEN_SIZE), dtype=np.float16)
    weight[0, 0] = np.float16(value)
    hidden = np.zeros(_HIDDEN_SIZE, dtype=np.float16)
    hidden[0] = np.float16(1.0)

    projected = _project(
        mx.array(hidden.reshape(1, _HIDDEN_SIZE)),
        _Codes(weight, np.zeros(_HIDDEN_SIZE, dtype=np.float16)),
        mx,
    )
    mx.eval(projected)

    observed = np.asarray(projected).reshape(-1)
    assert observed[0] == np.float32(value), name
    assert not observed[1:].any(), name
