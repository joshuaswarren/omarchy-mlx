// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
#pragma once

// Dense<->tile element placement shared by the libane device backend.
//
// Hardware-proven layout (a9f14124 smoke runtime): linear element index
// NCHW[6] = [N, C, H, W, plane_stride, row_stride] (same layout
// contract as the validated a9f14124 smoke runtime). element_size is 2
// for fp16 surfaces; the 2026-09-13 ABI extension adds 1-byte bool
// surfaces via the existing per-binding dtype field (compiler yield:
// no separate elementDtype field): logical_bytes stay bytes — a bool
// surface of N elements is N bytes — and only element-derived
// arithmetic widens from the fp16 assumption.
#include "mlx/backend/omarchy/ane/manifest.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mlx::core::omarchy::ane {

inline size_t ane_element_size(const AneProgramBinding& binding) {
  return binding.dtype == "bool" ? 1 : 2;
}

inline size_t ane_packed_offset(
    const AneProgramBinding& binding, size_t element) {
  const uint64_t height = binding.nchw[2];
  const uint64_t width = binding.nchw[3];
  const uint64_t plane_stride = binding.nchw[4];
  const uint64_t row_stride = binding.nchw[5];
  const uint64_t plane_elements = height * width;
  const uint64_t plane = element / plane_elements;
  const uint64_t within = element % plane_elements;
  const uint64_t row = within / width;
  const uint64_t column = within % width;
  return static_cast<size_t>(
      plane * plane_stride + row * row_stride +
      column * ane_element_size(binding));
}

// Row-wise dense<->tile transfer. A binding's tile places each row at
// plane * plane_stride + row * row_stride and stores width contiguous
// elements, and the dense source is row-major, so a full row is one
// memcpy of width * element_size bytes. The per-element reference loops
// this replaces measured ~50-80 ms per encoder island submit on a 13-inch M1;
// row memcpy is the same byte placement without the per-element call.
//
// `elements` caps the transfer below the full tensor when a caller
// stages fewer elements than NCHW describes; a partial trailing row (a
// case no validated bundle exercises) falls back to per-element
// placement so the byte placement stays identical to the reference.

inline void ane_pack_rows(
    const AneProgramBinding& binding,
    const uint8_t* dense,
    uint8_t* tile,
    size_t elements) {
  const size_t element_size = ane_element_size(binding);
  const uint64_t height = binding.nchw[2];
  const uint64_t width = binding.nchw[3];
  const uint64_t plane_stride = binding.nchw[4];
  const uint64_t row_stride = binding.nchw[5];
  const size_t row_bytes = static_cast<size_t>(width) * element_size;
  const size_t rows = static_cast<size_t>(binding.nchw[0] * binding.nchw[1] * height);
  const size_t full_rows = std::min(elements / width, rows);
  const bool dense_tile = row_stride == row_bytes &&
      plane_stride == static_cast<uint64_t>(height) * row_bytes;
  if (dense_tile && full_rows == rows) {
    std::memcpy(tile, dense, rows * row_bytes);
  } else {
    for (size_t row = 0; row < full_rows; ++row) {
      std::memcpy(
          tile + (row / height) * plane_stride + (row % height) * row_stride,
          dense + row * row_bytes,
          row_bytes);
    }
  }
  for (size_t element = full_rows * width; element < elements; ++element) {
    const size_t plane = static_cast<size_t>(element / (height * width));
    const size_t within = element % (height * width);
    std::memcpy(
        tile + plane * plane_stride + (within / width) * row_stride +
            (within % width) * element_size,
        dense + element * element_size,
        element_size);
  }
}

inline void ane_unpack_rows(
    const AneProgramBinding& binding,
    const uint8_t* tile,
    uint8_t* dense,
    size_t elements) {
  const size_t element_size = ane_element_size(binding);
  const uint64_t height = binding.nchw[2];
  const uint64_t width = binding.nchw[3];
  const uint64_t plane_stride = binding.nchw[4];
  const uint64_t row_stride = binding.nchw[5];
  const size_t row_bytes = static_cast<size_t>(width) * element_size;
  const size_t rows = static_cast<size_t>(binding.nchw[0] * binding.nchw[1] * height);
  const size_t full_rows = std::min(elements / width, rows);
  const bool dense_tile = row_stride == row_bytes &&
      plane_stride == static_cast<uint64_t>(height) * row_bytes;
  if (dense_tile && full_rows == rows) {
    std::memcpy(dense, tile, rows * row_bytes);
  } else {
    for (size_t row = 0; row < full_rows; ++row) {
      std::memcpy(
          dense + row * row_bytes,
          tile + (row / height) * plane_stride + (row % height) * row_stride,
          row_bytes);
    }
  }
  for (size_t element = full_rows * width; element < elements; ++element) {
    const size_t plane = static_cast<size_t>(element / (height * width));
    const size_t within = element % (height * width);
    std::memcpy(
        dense + element * element_size,
        tile + plane * plane_stride + (within / width) * row_stride +
            (within % width) * element_size,
        element_size);
  }
}


} // namespace mlx::core::omarchy::ane
