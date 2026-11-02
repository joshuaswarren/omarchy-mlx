// SPDX-License-Identifier: MIT
#include "contract.h"
#include <cassert>
#include <iostream>
struct Params {
  uint32_t count{}, operation{}, reduce_size{}, output_size{};
  uint32_t matrix_m{}, matrix_n{}, matrix_k{}, shape[4]{};
  uint32_t lhs_offset{}, output_offset{}, rhs_offset{}, aux_offset{}, aux_size{}, flags{};
};

int main() {
  using namespace calibration;
  for (auto cell : cells) {
    for (unsigned t = 5; t <= 8; ++t) {
      auto blocks = plan(cell.k, cell.n, t, 8, 8);
      assert(blocks.size() == 2);
      unsigned covered = 0;
      for (auto b : blocks) {
        assert(b.rows >= 1 && b.rows <= 4);
        assert(b.x_offset == 8 + covered * cell.k);
        assert(b.y_offset == 8 + covered * cell.n);
        assert(b.x_offset + b.rows * cell.k <= 8 + t * cell.k);
        assert(b.y_offset + b.rows * cell.n <= 8 + t * cell.n);
        assert(b.groups == (cell.n + 7) / 8);
        const auto p = push_constants<Params>(b, cell.k, cell.n, t);
        assert(p.count == b.rows * cell.n && p.output_size == t * cell.n);
        assert(p.matrix_m == 1 && p.matrix_n == cell.n && p.matrix_k == cell.k);
        assert(p.shape[0] == b.rows && p.operation == 4 && p.reduce_size == 64);
        assert(p.lhs_offset == b.x_offset && p.output_offset == b.y_offset);
        assert(p.rhs_offset == 0 && p.aux_offset == 0 && p.aux_size == 0 && p.flags == 0);
        covered += b.rows;
      }
      assert(covered == t && blocks[1].rows == t - 4);
      require_batch4_trace({547, 547}, 547);
    }
  }
  assert(two_word(2048, 16384));
  assert(!two_word(448, 130));
  assert(!two_word(2048, 130));
  for (auto trace : {std::vector<int>{}, {449}, {547}, {547, 410}, {547, 547, 547}}) {
    bool refused = false;
    try { require_batch4_trace(trace, 547); }
    catch (const std::runtime_error&) { refused = true; }
    assert(refused);
  }
  for (unsigned t : {0u, 1u, 4u, 9u, 16u}) {
    bool refused = false;
    try { plan(2048, 6144, t, 8, 8); }
    catch (const std::runtime_error&) { refused = true; }
    assert(refused);
  }
  for (auto shape : {Cell{0, 8}, Cell{65, 8}, Cell{2048, 0}}) {
    bool refused = false;
    try { plan(shape.k, shape.n, 8, 8, 8); }
    catch (const std::runtime_error&) { refused = true; }
    assert(refused);
  }
  assert(std::string(arm) == "calibration_batch4_4_plus_tail");
  std::cout << "16 shape/row plans, both word walks, offsets/tails, stale-route refusal PASS\n";
}
