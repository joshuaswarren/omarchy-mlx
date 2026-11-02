// SPDX-License-Identifier: MIT
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace calibration {
inline constexpr char arm[] = "calibration_batch4_4_plus_tail";
struct Cell { uint32_t k, n; };
inline constexpr std::array<Cell, 4> cells{{{448, 130}, {2048, 2048},
                                         {2048, 6144}, {2048, 16384}}};
struct Block { uint32_t rows, x_offset, y_offset, groups; };
inline bool two_word(uint32_t k, uint32_t n) {
  return k % 512 == 0 && n % 8 == 0;
}
inline std::vector<Block> plan(uint32_t k, uint32_t n, uint32_t t,
                               uint32_t x_offset, uint32_t y_offset) {
  if (k == 0 || k % 64 || n == 0 || t < 5 || t > 8 ||
      uint64_t(x_offset) + uint64_t(t) * k > UINT32_MAX ||
      uint64_t(y_offset) + uint64_t(t) * n > UINT32_MAX) {
    throw std::runtime_error("invalid calibration shape or span");
  }
  std::vector<Block> blocks;
  for (uint32_t row = 0; row < t; row += 4) {
    blocks.push_back({std::min(4u, t - row), x_offset + row * k,
                      y_offset + row * n, (n + 7u) / 8u});
  }
  return blocks;
}
template <class Params>
Params push_constants(Block block, uint32_t k, uint32_t n, uint32_t t) {
  Params params{};
  params.count = block.rows * n;
  params.operation = 4; params.reduce_size = 64; params.output_size = t * n;
  params.matrix_m = 1; params.matrix_n = n; params.matrix_k = k;
  params.shape[0] = block.rows;
  params.lhs_offset = block.x_offset; params.output_offset = block.y_offset;
  return params;
}
inline void require_batch4_trace(const std::vector<int>& trace, int batch4) {
  if (trace != std::vector<int>{batch4, batch4}) {
    throw std::runtime_error("calibration requires two actual Batch4 dispatches; not production T8");
  }
}
} // namespace calibration
