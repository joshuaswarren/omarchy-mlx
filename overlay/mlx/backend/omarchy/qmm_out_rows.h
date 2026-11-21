// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Output-row splitting for QuantizedMatmul when the out storage-buffer
// binding would exceed maxStorageBufferRange (issue #560, the lm_head
// logits matmul at vocab 248320 over context 8192 on Honeykrisp's
// INT32_MAX limit). The encoder's pre-dispatch refusal stays the back
// stop; these helpers let the m>1 routes emit one dispatch per out-row
// chunk instead, each binding a disjoint, offset-aligned byte range of
// the same buffer (VkDescriptorBufferInfo {buffer, offset, range}).
//
// The split must keep every chunk boundary element-exact: shaders read
// x and write out through vector paths that require the output's
// element offset to carry the same alignment the unchunked dispatch
// had, so a chunk binds at exactly out_start + r0 * row_bytes and the
// shader's output_offset resets to 0 within that bound range. That
// works only when every chunk-start byte offset is a multiple of the
// device's minStorageBufferOffsetAlignment: the out base must already
// be aligned, and rows_per_chunk must be a multiple of
// align / gcd(align, row_bytes).
//
// Pure arithmetic only, so a host test exercises the exact math the
// dispatch path uses.

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>

namespace mlx::core::omarchy {

// Rows per output chunk, or 0 when the out binding already fits the
// range limit or no aligned whole-row split exists. Row chunks only;
// column tiles cannot work because a column's writes are strided
// through the whole buffer, so a bound range would still have to span
// it.
//
// |matrix_m| is the dispatch's out row count, |row_bytes| its bytes
// per out row (matrix_n * itemsize at the call site), |out_byte_start|
// the byte offset of the out data inside its buffer (binding.offset +
// output_offset * itemsize), |range_limit| the device's
// maxStorageBufferRange and |align| its
// minStorageBufferOffsetAlignment (at least 16 for the vector paths).
inline uint32_t qmm_out_row_split(
    uint32_t matrix_m,
    uint64_t row_bytes,
    uint64_t out_byte_start,
    uint64_t range_limit,
    uint64_t align) {
  if (align == 0u) {
    align = 1u;
  }
  if (matrix_m <= 1u || row_bytes == 0u ||
      out_byte_start + uint64_t(matrix_m) * row_bytes <= range_limit) {
    return 0u;
  }
  // Every chunk-start byte offset must stay aligned: the out base and
  // each whole number of rows.
  if (out_byte_start % align != 0u) {
    return 0u;
  }
  const uint64_t row_mult = align / std::gcd(align, row_bytes % align);
  const uint64_t rows_fit = range_limit / row_bytes;
  const uint64_t rows_per = (rows_fit / row_mult) * row_mult;
  if (rows_per == 0u || rows_per >= matrix_m) {
    return 0u;
  }
  return static_cast<uint32_t>(rows_per);
}

// The per-chunk dispatch facts for the chunk beginning at out row
// |r0|. Everything the caller patches is here: the binding's new byte
// offset and range, the zeroed output_offset, the lhs_offset advance
// and the reduced matrix_m / count.
struct QmmOutChunk {
  uint64_t binding_offset; // out_start + r0 * row_bytes (align-aligned)
  uint64_t binding_range;  // bytes bound for this chunk, <= range_limit
  uint32_t rows;
  uint32_t lhs_offset_add; // r0 * matrix_k elements
  uint32_t count;          // rows * matrix_n output elements
};

inline QmmOutChunk qmm_out_chunk(
    uint32_t r0,
    uint32_t rows,
    uint64_t row_bytes,
    uint64_t out_byte_start,
    uint32_t matrix_n,
    uint32_t matrix_k,
    uint64_t range_limit) {
  QmmOutChunk c;
  c.binding_offset = out_byte_start + uint64_t(r0) * row_bytes;
  c.binding_range = std::min<uint64_t>(
      (uint64_t(rows) * row_bytes + 3ull) & ~3ull, range_limit);
  c.rows = rows;
  c.lhs_offset_add = static_cast<uint32_t>(uint64_t(r0) * matrix_k);
  c.count = static_cast<uint32_t>(uint64_t(rows) * matrix_n);
  return c;
}

} // namespace mlx::core::omarchy
