// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// A masked_scatter result feeding a concatenate must read back exact on
// every device. MaskedScatter bound only its own byte count
// (min(out_bytes, buffer size)); a later concatenate copy that lands in
// the recycled allocation writes past that declared range, and conformant
// drivers drop writes outside a descriptor's range. The second half then
// read back as the recycled allocation's stale bytes on discrete GPUs.
// The fix binds the full allocation, so the recycle path must be exact.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <iostream>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

// First mismatching unit index or npos when the evaluated array's raw
// storage equals the expected bits.
template <typename T>
size_t first_mismatch(const T* actual, const std::vector<T>& expected) {
  for (size_t i = 0; i < expected.size(); ++i) {
    if (actual[i] != expected[i]) {
      return i;
    }
  }
  return static_cast<size_t>(-1);
}

} // namespace

TEST_CASE("masked scatter half of a concatenate is exact") {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device.");
    return;
  }
  Stream s = gpu_stream();
  auto& encoder = omarchy::get_command_encoder(s);

  const Shape shape{1, 8, 256, 128};
  const int64_t per_half = 8 * 256 * 128;

  for (Dtype dtype : {float32, bfloat16}) {
    CAPTURE(dtype == float32 ? "float32" : "bfloat16");
    array source_f = reshape(
        arange(0, static_cast<float>(per_half), 1.0f, float32, s), shape, s);
    array source = astype(source_f, dtype, s);
    source.eval(s);
    encoder.synchronize();

    const size_t word_bytes =
        dtype == float32 ? sizeof(uint32_t) : sizeof(uint16_t);
    const size_t units =
        static_cast<size_t>(per_half) * source.itemsize() / word_bytes;

    // First half of the concatenate is a masked_scatter result that is
    // never force-evaluated: its buffer is only materialized inside the
    // concatenate's own eval, so the scatter writes and the concatenate
    // second-copy write can land in one submission on a recycled
    // allocation.
    array mask = ones({per_half}, bool_, s);
    array scatter_src = astype(
        reshape(arange(0, static_cast<float>(per_half), 1.0f, float32, s),
                {per_half}, s),
        dtype, s);
    array scattered =
        masked_scatter(zeros({per_half}, dtype, s), mask, scatter_src, s);
    scattered.reshape(shape, s);

    array lazy = concatenate(
        {scattered, zeros(shape, dtype, s)}, 2, s);
    lazy.eval(s);
    encoder.synchronize();

    std::vector<uint16_t> want16(units, 0);
    std::vector<uint32_t> want32(units, 0);
    std::vector<uint16_t> src16;
    std::vector<uint32_t> src32;
    if (dtype == float32) {
      const auto* p = scattered.data<uint32_t>();
      src32.assign(p, p + units);
    } else {
      const auto* p = scattered.data<uint16_t>();
      src16.assign(p, p + units);
    }
    if (dtype == float32) {
      const auto* bits = lazy.data<uint32_t>();
      CHECK(first_mismatch(bits, src32) == static_cast<size_t>(-1));
      CHECK(first_mismatch(bits + units, want32) == static_cast<size_t>(-1));
    } else {
      const auto* bits = lazy.data<uint16_t>();
      CHECK(first_mismatch(bits, src16) == static_cast<size_t>(-1));
      CHECK(first_mismatch(bits + units, want16) == static_cast<size_t>(-1));
    }

    // Control: evaluate the scatter before the concatenate; the ordered
    // graph must also be exact.
    array ordered = concatenate({scattered, zeros(shape, dtype, s)}, 2, s);
    ordered.eval(s);
    encoder.synchronize();
    if (dtype == float32) {
      const auto* bits = ordered.data<uint32_t>();
      CHECK(first_mismatch(bits + units, want32) == static_cast<size_t>(-1));
    } else {
      const auto* bits = ordered.data<uint16_t>();
      CHECK(first_mismatch(bits + units, want16) == static_cast<size_t>(-1));
    }
  }
}
