# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Parakeet-specific MLX GPU execution for the pinned decoder package.

The two recurrent layers follow Apple's MIL ``lstm`` definition: input,
forget, output, cell (IFOC) gate packing; row-major ``[4H, I]`` and
``[4H, H]`` weights. The gate preacts, unaries and cell algebra reproduce the
BNNS-graph fused ``ios18.lstm`` CPU operator bit-exactly as measured on the
studio-host reference host (receipt 2026-09-15-bnns-lstm-re): one fp16
k-blocked GEMV (block 128, fp16 FMA chains, fp16 block folds) over the
concatenated ``[x; h]`` input and repacked ``[wi | wh]`` weight, an fp16 bias
add, bit-indexed sigmoid/tanh tables captured through the fused op, the cell
update ``rnd16(rnd16(f*c0) + i*g)`` (FMUL then FMA), and
``h = rnd16(o*tanh16(c1))``. The emulation was validated 640/640 lanes
bit-exact on both state outputs against the live operator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from functools import cache
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .pinned_component import (
    PinnedComponent,
    PinnedComponentError,
    load_pinned_component,
)
from .reference import ReferenceLock

_HIDDEN_SIZE = 640
_VOCAB_SIZE = 8193
_INPUT_IDS_SHAPE = (1, 1)
_STATE_SHAPE = (2, 1, _HIDDEN_SIZE)

_CONSTANT_SPECS = {
    "embedding_weight_to_fp16": ((_VOCAB_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "concat_1_to_fp16": ((4 * _HIDDEN_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "concat_2_to_fp16": ((4 * _HIDDEN_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "concat_0_to_fp16": ((4 * _HIDDEN_SIZE,), np.dtype("<f2")),
    "concat_4_to_fp16": ((4 * _HIDDEN_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "concat_5_to_fp16": ((4 * _HIDDEN_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "concat_3_to_fp16": ((4 * _HIDDEN_SIZE,), np.dtype("<f2")),
    "projector_weight_to_fp16": ((_HIDDEN_SIZE, _HIDDEN_SIZE), np.dtype("<f2")),
    "projector_bias_to_fp16": ((_HIDDEN_SIZE,), np.dtype("<f2")),
}

_EXPECTED_HISTOGRAM = {
    "add": 1,
    "cast": 8,
    "const": 44,
    "gather": 1,
    "greater_equal": 1,
    "linear": 1,
    "lstm": 2,
    "select": 1,
    "split": 2,
    "squeeze": 4,
    "stack": 2,
    "transpose": 2,
}

_F16 = np.dtype("<f2")


def fp16_sigmoid(x):
    """Correctly-rounded binary16 sigmoid: round ``1/(1+exp(-x))`` once."""
    x64 = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore"):
        y = 1.0 / (1.0 + np.exp(-x64))
    return np.asarray(y, dtype=_F16)


def fp16_tanh(x):
    """Correctly-rounded binary16 tanh: round ``tanh(x)`` once."""
    return np.asarray(np.tanh(np.asarray(x, dtype=np.float64)), dtype=_F16)


@cache
def _fused_luts():
    """Bit-indexed fused ``ios18.lstm`` sigmoid/tanh tables (studio-host reads).

    Each entry is indexed by the uint16 bit pattern of the fp16 argument and
    holds the operator's fp16 result. NaN/zero/saturation lanes follow the
    measured operator behaviour documented in receipt
    2026-09-15-bnns-lstm-re.
    """
    path = Path(__file__).resolve().parent / "fused_lut_2026-09-15.npz"
    with np.load(path) as z:
        return z["sigma_lut"], z["tanh_lut"]


def _lut_apply(lut, x):
    values = np.atleast_1d(np.asarray(x, dtype=_F16))
    return lut[values.view(np.uint16)]


def fused_sigmoid(x):
    """Fused-op sigmoid from the measured bit-indexed table."""
    return _lut_apply(_fused_luts()[0], x)


def fused_tanh(x):
    """Fused-op tanh from the measured bit-indexed table."""
    return _lut_apply(_fused_luts()[1], x)


def _add16(u, v):
    """fp16 addition with one rounding of the exact sum."""
    return (np.asarray(u, dtype=np.float64) + np.asarray(v, dtype=np.float64)).astype(_F16)


def _blocked_gemv(a, weight_t64, block=128):
    """BNNS fp16 GEMV: y[n] = sum_k a[k] * W[k][n] over ``weight_t64``.

    ``weight_t64`` is the [K, N] fp64 view of the fp16 weight (row-major over
    the reduction index). fp16 FMA chains (single rounding per term) over
    k-blocks of ``min(max(block, 8), K)``, each block folded into the fp16
    destination with one more rounding; the first block stores. fp64
    intermediate arithmetic reproduces each fp16 FMA exactly because fp16
    products are exact in fp64.
    """
    k_len, n_len = weight_t64.shape
    step = min(max(block, 8), k_len)
    a64 = np.asarray(a, dtype=np.float64)
    out = np.empty(n_len, dtype=_F16)
    for kb in range(0, k_len, step):
        acc = np.zeros(n_len, dtype=_F16)
        for k in range(kb, min(kb + step, k_len)):
            acc = (acc.astype(np.float64) + a64[k] * weight_t64[k]).astype(_F16)
        if kb == 0:
            out[:] = acc
        else:
            out[:] = (out.astype(np.float64) + acc.astype(np.float64)).astype(_F16)
    return out


def fused_lstm_layer(x, hidden, cell, weight_t64, bias):
    """One fused ``ios18.lstm`` timestep on the measured BNNS contract.

    ``x``, ``hidden``, ``cell`` are fp16-convertible arrays (640 lanes);
    ``weight_t64`` the [I+H, 4H] fp64 GEMV matrix of the repacked
    ``[wi | wh]`` weight; ``bias`` the fp16 bias (4H lanes). Returns
    ``(next_hidden, next_cell)`` as fp16 arrays of 640 lanes.
    """
    lanes = int(bias.size) // 4
    a = np.concatenate(
        [
            np.asarray(x, dtype=_F16).ravel(),
            np.asarray(hidden, dtype=_F16).ravel(),
        ]
    ).astype(_F16)
    preact = _add16(_blocked_gemv(a, weight_t64), np.asarray(bias, dtype=_F16).ravel())
    sig, tanh = _fused_luts()
    g_i = sig[preact[:lanes].view(np.uint16)]
    g_f = sig[preact[lanes : 2 * lanes].view(np.uint16)]
    g_o = sig[preact[2 * lanes : 3 * lanes].view(np.uint16)]
    g_g = tanh[preact[3 * lanes : 4 * lanes].view(np.uint16)]
    c0 = np.asarray(cell, dtype=_F16).ravel()
    forget_product = (g_f.astype(np.float64) * c0.astype(np.float64)).astype(_F16)
    next_cell = (
        forget_product.astype(np.float64) + g_i.astype(np.float64) * g_g.astype(np.float64)
    ).astype(_F16)
    tanh_cell = tanh[next_cell.view(np.uint16)]
    next_hidden = (g_o.astype(np.float64) * tanh_cell.astype(np.float64)).astype(_F16)
    return next_hidden, next_cell


def fused_lstm_numpy(x, hidden, cell, weight_ih, weight_hh, bias):
    """Contract entry point on raw fp16 weights; mirrors the pinned decode."""
    weight_t64 = np.ascontiguousarray(
        np.concatenate(
            [
                np.asarray(weight_ih, dtype=_F16),
                np.asarray(weight_hh, dtype=_F16),
            ],
            axis=1,
        ).T,
        dtype=np.float64,
    )
    next_hidden, next_cell = fused_lstm_layer(x, hidden, cell, weight_t64, bias)
    return (
        np.expand_dims(next_hidden.reshape(1, -1), axis=0),
        next_hidden.reshape(1, -1),
        next_cell.reshape(1, -1),
    )


def _validate_graph(component: PinnedComponent) -> None:
    histogram = dict(sorted(Counter(op.type for op in component.block.operations).items()))
    if histogram != _EXPECTED_HISTOGRAM:
        raise PinnedComponentError(
            f"pinned decoder operation inventory differs: {histogram}"
        )
    if tuple(component.block.outputs) != (
        "decoder_hidden",
        "next_hidden",
        "next_cell",
    ):
        raise PinnedComponentError("pinned decoder output bindings differ")


class DecoderResult(NamedTuple):
    decoder_hidden: object
    next_hidden: object
    next_cell: object


class _Weights(NamedTuple):
    embedding: object
    layer_0_ih: object
    layer_0_hh: object
    layer_0_bias: object
    layer_1_ih: object
    layer_1_hh: object
    layer_1_bias: object
    projector: object
    projector_bias: object


@cache
def _mlx():
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise RuntimeError("the pinned Parakeet decoder requires mlx-omarchy") from exc
    return mx


def _constant(component: PinnedComponent, name: str) -> np.ndarray:
    value = component.constant(name)
    shape, dtype = _CONSTANT_SPECS[name]
    if value.shape != shape or value.dtype != dtype:
        raise PinnedComponentError(
            f"pinned decoder constant {name!r} must be {dtype.name}{shape}, "
            f"got {value.dtype.name}{value.shape}"
        )
    return value


def _load_weights(component: PinnedComponent, mx) -> _Weights:
    values = [_constant(component, name) for name in _CONSTANT_SPECS]
    with mx.stream(mx.gpu):
        arrays = [mx.array(value) for value in values]
    mx.eval(*arrays)
    return _Weights(*arrays)


def _validate_array(name, value, shape, dtype, mx) -> None:
    if not isinstance(value, mx.array):
        raise TypeError(f"{name} must be an mlx.core.array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} must have {dtype} dtype, got {value.dtype}")


class VulkanDecoder:
    """Pinned decoder: GPU embedding/projector, measured BNNS LSTM contract."""

    __slots__ = ("_weights", "_layers")

    def __init__(self, component: PinnedComponent):
        _validate_graph(component)
        self._weights = _load_weights(component, _mlx())
        self._layers = tuple(
            (
                np.ascontiguousarray(
                    np.concatenate(
                        [
                            np.asarray(wih, dtype=_F16),
                            np.asarray(whh, dtype=_F16),
                        ],
                        axis=1,
                    ).T,
                    dtype=np.float64,
                ),
                np.asarray(bias, dtype=_F16).ravel().copy(),
            )
            for wih, whh, bias in (
                (
                    self._weights.layer_0_ih,
                    self._weights.layer_0_hh,
                    self._weights.layer_0_bias,
                ),
                (
                    self._weights.layer_1_ih,
                    self._weights.layer_1_hh,
                    self._weights.layer_1_bias,
                ),
            )
        )

    def __call__(self, input_ids, hidden, cell) -> DecoderResult:
        mx = _mlx()
        _validate_array("input_ids", input_ids, _INPUT_IDS_SHAPE, mx.int32, mx)
        _validate_array("hidden", hidden, _STATE_SHAPE, mx.float32, mx)
        _validate_array("cell", cell, _STATE_SHAPE, mx.float32, mx)
        weights = self._weights

        with mx.stream(mx.gpu):
            indices = input_ids.astype(mx.int16).astype(mx.int32)
            indices = mx.where(indices >= 0, indices, indices + _VOCAB_SIZE)
            indices = indices.astype(mx.int16)
            embedded = mx.take(weights.embedding, indices, axis=0)
            sequence = mx.transpose(embedded, (1, 0, 2))
            hidden_fp16 = hidden.astype(mx.float16)
            cell_fp16 = cell.astype(mx.float16)
            x_in = np.asarray(sequence[0], dtype=_F16).ravel().copy()
            states = []
            for (weight_t64, bias), h_state, c_state in (
                (self._layers[0], hidden_fp16[0], cell_fp16[0]),
                (self._layers[1], hidden_fp16[1], cell_fp16[1]),
            ):
                next_hidden, next_cell = fused_lstm_layer(
                    x_in, h_state, c_state, weight_t64, bias
                )
                states.append((next_hidden, next_cell))
                x_in = next_hidden
            sequence = mx.array(states[1][0].reshape(1, 1, -1))
            decoder_hidden = (
                mx.transpose(sequence, (1, 0, 2)) @ weights.projector.T
                + weights.projector_bias
            ).astype(mx.float32)
            next_hidden = mx.stack(
                tuple(mx.array(state[0].reshape(1, -1)) for state in states),
                axis=0,
            ).astype(mx.float32)
            next_cell = mx.stack(
                tuple(mx.array(state[1].reshape(1, -1)) for state in states),
                axis=0,
            ).astype(mx.float32)
        return DecoderResult(decoder_hidden, next_hidden, next_cell)


def load_decoder(package_path: Path) -> VulkanDecoder:
    """Hash-validate and load the exact pinned ``decoder.mlpackage``."""
    return VulkanDecoder(load_pinned_component(package_path, "decoder"))


def _tensor_summary(value) -> dict:
    materialized = np.asarray(value)
    return {
        "shape": list(materialized.shape),
        "dtype": str(materialized.dtype),
        "finite": bool(np.isfinite(materialized).all()),
        "sha256": hashlib.sha256(materialized.tobytes()).hexdigest(),
    }


def smoke(package_path: Path) -> dict:
    """Run two dependent decoder transitions and return a machine-readable receipt."""
    mx = _mlx()
    decoder = load_decoder(package_path)
    with mx.stream(mx.gpu):
        hidden = mx.zeros(_STATE_SHAPE, dtype=mx.float32)
        cell = mx.zeros(_STATE_SHAPE, dtype=mx.float32)
        first = decoder(mx.array([[0]], dtype=mx.int32), hidden, cell)
        second = decoder(
            mx.array([[8192]], dtype=mx.int32),
            first.next_hidden,
            first.next_cell,
        )
    mx.eval(*first, *second)
    summaries = {
        "first": {name: _tensor_summary(value) for name, value in first._asdict().items()},
        "second": {name: _tensor_summary(value) for name, value in second._asdict().items()},
    }
    if not all(item["finite"] for step in summaries.values() for item in step.values()):
        raise RuntimeError("pinned decoder smoke produced a non-finite output")
    lock = ReferenceLock.load()
    return {
        "schema": "mlx-omarchy.parakeet-vulkan-decoder-smoke/1",
        "component": "decoder.mlpackage",
        "model_revision": lock.model_revision,
        "device": str(mx.default_device()),
        "tokens": [0, 8192],
        "state_reused_on_gpu": True,
        "outputs": summaries,
        "native_coreml_parity": False,
        "native_capture_required": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(smoke(args.package), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
