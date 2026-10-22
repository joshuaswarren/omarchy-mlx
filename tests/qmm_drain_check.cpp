#include <array>
#include <cassert>
#include <iostream>
#include <vector>

using uint = unsigned int;
#include "../overlay/mlx/backend/omarchy/shaders/qmm_drain.h"

int main() {
  uint cases = 0;
  for (uint tile_rows : {16u, 32u}) {
    const uint blocks = tile_rows / 16u;
    for (uint m : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 65u}) {
      for (uint n : {1u, 7u, 8u, 9u, 31u, 32u, 33u, 63u, 65u}) {
        std::vector<uint> output(m * n, 0u);
        std::vector<uint> writes(m * n, 0u);
        for (uint row_base = 0; row_base < m; row_base += tile_rows) {
          for (uint col_base = 0; col_base < n; col_base += 32u) {
            for (uint block = 0; block < blocks; ++block) {
              std::array<uint, 512> scratch{};
              std::array<bool, 512> occupied{};
              for (uint subgroup = 0; subgroup < 2u; ++subgroup) {
                const uint row_block = subgroup * (tile_rows / 2u) + block * 8u;
                for (uint cb = 0; cb < 4u; ++cb) {
                  for (uint r = 0; r < 8u; ++r) {
                    for (uint c = 0; c < 8u; ++c) {
                      const uint slot = qmm_drain_slot(subgroup, cb, r, c);
                      assert(slot < scratch.size());
                      assert(!occupied[slot]);
                      occupied[slot] = true;
                      scratch[slot] = 1u + (row_base + row_block + r) * n + col_base + cb * 8u + c;
                    }
                  }
                }
              }
              for (uint subgroup = 0; subgroup < 2u; ++subgroup) {
                const uint row_block = qmm_drain_row(subgroup, block, blocks);
                for (uint lane = 0; lane < 32u; ++lane) {
                  for (uint r = 0; r < 8u; ++r) {
                    const uint row = row_base + row_block + r;
                    const uint col = col_base + lane;
                    if (row < m && col < n) {
                      const uint dst = row * n + col;
                      const uint slot = qmm_drain_slot(subgroup, lane / 8u, r, lane % 8u);
                      assert(occupied[slot]);
                      output[dst] = scratch[slot];
                      ++writes[dst];
                    }
                  }
                }
              }
            }
          }
        }
        for (uint i = 0; i < output.size(); ++i) {
          assert(output[i] == i + 1u);
          assert(writes[i] == 1u);
        }
        ++cases;
      }
    }
  }
  std::cout << cases << " drain layouts: exact output, one write per element, disjoint subgroup scratch\n";
}
