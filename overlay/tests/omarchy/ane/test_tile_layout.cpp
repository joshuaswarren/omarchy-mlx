// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Host tests for the dense<->tile element placement across element
// sizes (fp16 stride 2, bool stride 1). No device, no libane.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "mlx/backend/omarchy/ane/tile_layout.h"

using namespace mlx::core::omarchy::ane;

namespace {

AneProgramBinding binding_of(
    const std::string& dtype,
    std::array<uint64_t, 6> nchw) {
  AneProgramBinding binding;
  binding.tensor = "t";
  binding.dtype = dtype;
  binding.nchw = nchw;
  return binding;
}

} // namespace

TEST_CASE("fp16 offsets are unchanged from the hardware-proven formula") {
  // The encoder skew pattern: [N=1, C=8, H=375, W=749] with the proven
  // plane/row strides from the a9f14124 runtime (64/64 bytes).
  auto b = binding_of("float16", {1, 8, 375, 749, 64, 64});
  CHECK(ane_element_size(b) == 2);
  // element 0 -> plane 0, row 0, column 0
  CHECK(ane_packed_offset(b, 0) == 0);
  // element 1 -> column 1 -> 2 bytes
  CHECK(ane_packed_offset(b, 1) == 2);
  // last column of the first row
  CHECK(ane_packed_offset(b, 748) == 748 * 2);
  // first column of the second row -> row_stride
  CHECK(ane_packed_offset(b, 749) == 64);
  // the add-mul fixture geometry: NCHW [1,64,1,1,64,64]
  auto addmul = binding_of("float16", {1, 64, 1, 1, 64, 64});
  for (size_t element = 0; element < 64; ++element) {
    CHECK(ane_packed_offset(addmul, element) == element * 64);
  }
}

TEST_CASE("bool surfaces use one-byte columns, byte strides unchanged") {
  auto b = binding_of("bool", {1, 8, 375, 749, 64, 64});
  CHECK(ane_element_size(b) == 1);
  CHECK(ane_packed_offset(b, 0) == 0);
  CHECK(ane_packed_offset(b, 1) == 1);
  CHECK(ane_packed_offset(b, 748) == 748);
  // row stride is byte-addressed and identical to fp16
  CHECK(ane_packed_offset(b, 749) == 64);
  CHECK(ane_packed_offset(b, 749 + 5) == 64 + 5);
  // plane stride unchanged too
  auto deep = binding_of("bool", {1, 8, 375, 749, 3000, 64});
  CHECK(ane_packed_offset(deep, 375 * 749) == 3000);
}

TEST_CASE("byte-accounting rule: N bool elements are N bytes") {
  // The ABI contract: logical_bytes stay bytes, so a bool surface of N
  // logical elements has logical_bytes == N. The staging loops derive
  // the element count as logical_bytes / element_size, which the worker
  // tests exercise end to end with the mock device.
  auto b = binding_of("bool", {1, 2, 3, 4, 64, 64});
  const size_t elements = 1 * 2 * 3 * 4;
  const size_t logical_bytes = elements * ane_element_size(b);
  CHECK(logical_bytes == elements);
  auto f = binding_of("float16", {1, 2, 3, 4, 64, 64});
  CHECK(24 * ane_element_size(f) == 48);
}
