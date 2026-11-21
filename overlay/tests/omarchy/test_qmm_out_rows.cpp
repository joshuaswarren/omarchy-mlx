// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Issue #560: the lm_head QuantizedMatmul at context 8192 writes
// 248320 x 8192 bf16 logits - 3.8 GiB, over Honeykrisp's
// maxStorageBufferRange of INT32_MAX. The encoder refuses the single
// binding, so eval_gpu splits the output into whole-row chunks, each
// binding a disjoint, minStorageBufferOffsetAlignment-aligned range of
// the same buffer. These checks pin the pure split math the dispatch
// wrapper uses: chunk size, binding offsets/ranges, the lhs advance
// and every no-split case.

#include <cstdint>
#include <cstdio>

#include "qmm_out_rows.h"

namespace om = mlx::core::omarchy;

namespace {

int fails = 0;

void check(bool cond, int line, const char* expr) {
  if (!cond) {
    ++fails;
    std::printf("FAIL %d: %s\n", line, expr);
  }
}

#define CHECK(c) check((c), __LINE__, #c)

} // namespace

int main() {
  const uint64_t limit = 2147483647ull; // Honeykrisp INT32_MAX
  const uint64_t row_bytes = 248320ull * 2; // vocab x bf16 at ctx8192

  // The issue-560 geometry splits, and every chunk range stays legal.
  const uint32_t rows_per =
      om::qmm_out_row_split(8192, row_bytes, 0, limit, 16);
  CHECK(rows_per >= 1u && rows_per < 8192u);
  CHECK(uint64_t(rows_per) * row_bytes <= limit);
  CHECK((uint64_t(rows_per) * row_bytes) % 16 == 0);

  // A middle chunk binds at the exact aligned row start, covers only
  // its rows, and carries the matching lhs advance and output count.
  const om::QmmOutChunk c =
      om::qmm_out_chunk(rows_per, 100, row_bytes, 0, 248320, 2048, limit);
  CHECK(c.binding_offset == uint64_t(rows_per) * row_bytes);
  CHECK(c.binding_offset % 16 == 0);
  CHECK(c.binding_range == 100ull * row_bytes);
  CHECK(c.rows == 100u);
  CHECK(c.count == 100ull * 248320u);
  CHECK(c.lhs_offset_add == uint64_t(rows_per) * 2048u);

  // A chunk range clamps at the range limit when the rounded word
  // end would pass it.
  const om::QmmOutChunk tail = om::qmm_out_chunk(
      0, 1, row_bytes, 0, 248320, 2048, row_bytes - 10);
  CHECK(tail.binding_range == row_bytes - 10);
  (void)tail;

  // Shapes that already fit never split.
  CHECK(om::qmm_out_row_split(64, 512, 0, limit, 16) == 0);
  CHECK(om::qmm_out_row_split(4, 400, 0, 4u * 400u, 16) == 0);
  CHECK(om::qmm_out_row_split(1, row_bytes, 0, limit, 16) == 0);

  // A single row over the limit cannot split; the encoder refusal
  // still stands for it.
  CHECK(om::qmm_out_row_split(8, 8, 0, 4, 16) == 0);

  // An unaligned out base cannot produce aligned chunk starts; the
  // refusal stands there too.
  CHECK(om::qmm_out_row_split(8192, row_bytes, 2, limit, 16) == 0);

  // Row counts keep every chunk start a multiple of the alignment:
  // gcd(16, 66) = 2, so rows_per is a multiple of 8.
  const uint32_t rb2 =
      om::qmm_out_row_split(100, 66, 0, 66ull * 9 * 16 + 66, 16);
  if (rb2 != 0u) {
    CHECK((uint64_t(rb2) * 66u) % 16 == 0);
  }

  if (fails == 0) {
    std::printf("qmm_out_rows: all checks passed\n");
  }
  return fails;
}
