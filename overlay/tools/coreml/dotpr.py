# Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
# SPDX-License-Identifier: MIT
"""Portable float32 reduction matching contiguous ``vDSP_dotpr`` on arm64."""

from __future__ import annotations

import numpy as np


def _fma32(lhs: np.ndarray, rhs: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """Round an exact float32 product-plus-accumulator once to float32."""
    lhs64 = lhs.astype(np.float64)
    rhs64 = rhs.astype(np.float64)
    acc64 = acc.astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        product = lhs64 * rhs64
        total = product + acc64
        back = total - product
        residual = (product - (total - back)) + (acc64 - back)
        even = np.bitwise_and(total.view(np.uint64), np.uint64(1)) == 0
        needs_nudge = np.isfinite(total) & (residual != 0) & even
        direction = np.where(residual > 0, np.inf, -np.inf)
        sticky = np.where(needs_nudge, np.nextafter(total, direction), total)
        return sticky.astype(np.float32)


def _add32(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.add(lhs, rhs, dtype=np.float32)


def _validate_rows(lhs: np.ndarray, rhs: np.ndarray) -> None:
    if lhs.dtype != np.float32 or rhs.dtype != np.float32:
        raise ValueError("dot product inputs must have float32 dtype")
    if lhs.ndim != 2 or rhs.ndim != 2:
        raise ValueError("dot product row inputs must be two-dimensional")
    if lhs.shape[1] != rhs.shape[1]:
        raise ValueError("dot product inputs must have the same length")
    if lhs.shape[1] == 0:
        raise ValueError("dot product inputs must be non-empty")


def _dotpr_rows(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    _validate_rows(lhs, rhs)
    length = lhs.shape[1]
    if length < 4:
        result = np.zeros((rhs.shape[0], lhs.shape[0]), dtype=np.float32)
        for offset in range(length):
            result = _fma32(
                lhs[None, :, offset], rhs[:, None, offset], result
            )
        return result

    accumulators = np.zeros((8, rhs.shape[0], lhs.shape[0], 4), dtype=np.float32)
    groups = (8, 12, 16, 20, 24, 28, 4, 0)
    offset = 0

    while length - offset >= 32:
        for accumulator, relative in enumerate(groups):
            indices = slice(offset + relative, offset + relative + 4)
            accumulators[accumulator] = _fma32(
                lhs[None, :, indices],
                rhs[:, None, indices],
                accumulators[accumulator],
            )
        offset += 32

    first = _add32(accumulators[0], accumulators[1])
    second = _add32(accumulators[2], accumulators[3])
    third = _add32(accumulators[4], accumulators[5])
    fourth = _add32(accumulators[6], accumulators[7])
    first = _add32(first, second)
    second = _add32(third, fourth)

    while length - offset >= 8:
        first = _fma32(
            lhs[None, :, offset : offset + 4],
            rhs[:, None, offset : offset + 4],
            first,
        )
        second = _fma32(
            lhs[None, :, offset + 4 : offset + 8],
            rhs[:, None, offset + 4 : offset + 8],
            second,
        )
        offset += 8

    first = _add32(first, second)
    if length - offset >= 4:
        first = _fma32(
            lhs[None, :, offset : offset + 4],
            rhs[:, None, offset : offset + 4],
            first,
        )
        offset += 4

    pairs = _add32(first[..., (0, 2)], first[..., (1, 3)])
    if length - offset >= 2:
        pairs = _fma32(
            lhs[None, :, offset : offset + 2],
            rhs[:, None, offset : offset + 2],
            pairs,
        )
        offset += 2

    result = _add32(pairs[..., 0], pairs[..., 1])
    if offset < length:
        result = _fma32(
            lhs[None, :, offset],
            rhs[:, None, offset],
            result,
        )
    return result


def portable_dotpr(lhs: np.ndarray, rhs: np.ndarray) -> np.float32:
    """Return one contiguous float32 dot product in arm64 vDSP order."""
    lhs = np.asarray(lhs)
    rhs = np.asarray(rhs)
    if lhs.ndim != 1 or rhs.ndim != 1:
        raise ValueError("dot product inputs must be one-dimensional")
    return _dotpr_rows(lhs[None, :], rhs[None, :])[0, 0]


def mel_projection(filterbank: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Project power frames through mel rows in arm64 vDSP order."""
    return _dotpr_rows(np.asarray(filterbank), np.asarray(power))
