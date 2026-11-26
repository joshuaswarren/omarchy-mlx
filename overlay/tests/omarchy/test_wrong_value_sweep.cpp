// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wrong-value sweep regression tests (2026-09-11):
//
// 1. Float-to-bool casts decide from encoded bits. A float compare
//    flushes subnormals on drivers without DenormPreserve (the
//    Honeykrisp fork), which turned f32 and bf16 subnormals into False;
//    the f16 path survived only because f16 subnormals decode to normal
//    f32 values.
//
// 2. Padded pools and same-mode 1-D convolves keep exact boundary
//    values under allocator churn. Both route their zero padding through
//    the host scalar fill; when the fill's drain was removed
//    (5facd59a) the boundary bytes kept the recycled block's previous
//    contents and only the interior stayed correct.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/random.h"
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

bool compute_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

// Builds a float32 array from raw IEEE bits (the C++ twin of the
// upstream test's uint32 .view(float32) trick); an astype from uint32
// would integer-convert instead of reinterpreting.
array f32_from_bits(const std::vector<uint32_t>& bits, Stream s) {
  std::vector<float> values(bits.size());
  memcpy(values.data(), bits.data(), bits.size() * sizeof(float));
  return array(values.begin(), Shape{int(values.size())}, float32);
}

// Reads an evaluated array's host-visible storage.
std::vector<float> read_values(const array& a) {
  const_cast<array&>(a).eval();
  std::vector<float> host(a.size());
  memcpy(host.data(), a.data<float>(), host.size() * sizeof(float));
  return host;
}

void expect_values_conv(const std::vector<float>& host,
                        const std::vector<float>& want) {
  REQUIRE(host.size() == want.size());
  for (size_t i = 0; i < host.size(); ++i) {
    if (std::fabs(host[i] - want[i]) > 1e-5f) {
      FAIL("index ", i, ": got ", host[i], " want ", want[i]);
    }
  }
}

// Fill freed blocks with random normal bytes worth of values so the array
// under test recycles a dirty block. Fresh zeroed pages hide the recycled
// fill bug this file pins.
void churn(Stream stream) {
  std::vector<array> garbage;
  for (int i = 0; i < 4; ++i) {
    garbage.push_back(random::normal(Shape{1 << 18}, float32, std::nullopt, stream));
  }
  for (auto& g : garbage) {
    g.eval();
  }
}

std::vector<float> conv_same_reference(
    const std::vector<float>& in,
    const std::vector<float>& k) {
  // np.convolve 'same' with N=4 == pad (2, 1) then a valid window pass.
  // conv1d is a cross-correlation: out[o] = sum_t padded[o+t] * k[t].
  std::vector<float> padded(in.size() + 3, 0.0f);
  for (size_t i = 0; i < in.size(); ++i) {
    padded[i + 2] = in[i];
  }
  std::vector<float> out(in.size(), 0.0f);
  for (size_t o = 0; o < in.size(); ++o) {
    double acc = 0.0;
    for (size_t t = 0; t < k.size(); ++t) {
      acc += static_cast<double>(padded[o + t]) * k[t];
    }
    out[o] = static_cast<float>(acc);
  }
  return out;
}

} // namespace

TEST_CASE("float to bool casts decide from encoded bits") {
  if (!compute_available()) {
    return;
  }
  Stream s = gpu_stream();
  // 0x00000001 is the smallest positive f32 subnormal. Both float subnormals
  // and bfloat16 subnormals ride the f32 subnormal encoding: a flush-to-zero
  // float compare reads them as zero. Every listed encoding except the two
  // zeros and (by contract) nothing else casts false; NaN casts true.
  std::vector<uint32_t> f32_bits{
      0x00000001, // smallest subnormal -> true
      0x007FFFFF, // largest subnormal -> true
      0x80000001, // negative subnormal -> true
      0x7FC00000, // NaN -> true
      0x7F800000, // +inf -> true
      0xFF800000, // -inf -> true
      0x00000000, // +0 -> false
      0x80000000, // -0 -> false
      0x3F800000, // 1.0 -> true
  };
  array f32_in = f32_from_bits(f32_bits, s);
  array f32_bool = astype(f32_in, bool_, s);
  f32_bool.eval();
  const bool* f32_out = f32_bool.data<bool>();
  CHECK(f32_out[0]);
  CHECK(f32_out[1]);
  CHECK(f32_out[2]);
  CHECK(f32_out[3]);
  CHECK(f32_out[4]);
  CHECK(f32_out[5]);
  CHECK_FALSE(f32_out[6]);
  CHECK_FALSE(f32_out[7]);
  CHECK(f32_out[8]);

  // bf16 encodings lifted into their exact f32 positions: astype
  // f32 -> bfloat16 then re-creates the target bits losslessly.
  std::vector<uint32_t> bf16_f32_bits{
      0x00010000, // smallest bf16 subnormal -> true
      0x007F0000, // bf16 subnormal -> true
      0x80010000, // negative bf16 subnormal -> true
      0x00000000, // +0 -> false
      0x80000000, // -0 -> false
      0x3F800000, // 1.0 -> true
  };
  array bf16_in = astype(
      f32_from_bits(bf16_f32_bits, s), bfloat16, s);
  array bf16_bool = astype(bf16_in, bool_, s);
  bf16_bool.eval();
  const bool* bf16_out = bf16_bool.data<bool>();
  CHECK(bf16_out[0]);
  CHECK(bf16_out[1]);
  CHECK(bf16_out[2]);
  CHECK_FALSE(bf16_out[3]);
  CHECK_FALSE(bf16_out[4]);
  CHECK(bf16_out[5]);

  // f16 subnormals decode to normal f32 values and were never broken;
  // kept as the control the failing cluster was measured against.
  // f16 subnormals decode to normal f32 values, so astype round-trips
  // them exactly.
  std::vector<uint32_t> f16_f32_bits{
      0x35800000, // 0x0001: smallest positive f16 subnormal
      0x387FE000, // 0x03FF: largest f16 subnormal
      0x00000000, // +0 -> false
      0x80000000, // -0 -> false
  };
  array f16_in = astype(f32_from_bits(f16_f32_bits, s), float16, s);
  array f16_bool = astype(f16_in, bool_, s);
  f16_bool.eval();
  const bool* f16_out = f16_bool.data<bool>();
  CHECK(f16_out[0]);
  CHECK(f16_out[1]);
  CHECK_FALSE(f16_out[2]);
  CHECK_FALSE(f16_out[3]);
}

TEST_CASE("same-mode convolve and padded pool survive allocator churn") {
  if (!compute_available()) {
    return;
  }
  Stream s = gpu_stream();

  std::vector<float> a_data(24);
  for (int i = 0; i < 24; ++i) {
    a_data[i] = 0.1f + 0.03f * i;
  }
  std::vector<float> v_data{0.5f, 0.25f, 0.125f, 0.0625f};
  auto want = conv_same_reference(a_data, v_data);

  // The upstream failing signature: mx.convolve(a 24, v 4, mode='same')
  // pads (2, 1), so outputs 0, 1, and 23 read scalar-filled bytes. With
  // the drain missing, churned runs return the recycled block's bytes at
  // exactly those boundary outputs while the clean-state run is exact.
  auto run_conv = [&]() {
    array a(a_data.begin(), Shape{24}, float32);
    array v(v_data.begin(), Shape{4}, float32);
    array padded_flat = pad(a, std::vector<int>{0}, Shape{2}, Shape{1});
    array padded = reshape(padded_flat, Shape{1, 27, 1}, s);
    array w3 = reshape(v, Shape{1, 4, 1}, s);
    array out = reshape(conv1d(padded, w3, 1, 0, 1, 1, s), Shape{24}, s);
    return read_values(out);
  };
  expect_values_conv(run_conv(), want);
  for (int rep = 0; rep < 8; ++rep) {
    churn(s);
    expect_values_conv(run_conv(), want);
  }

  // MaxPool1d k2 s2 p1 over (2, 4, 3): the upstream test_pooling case.
  // Pool = max over a (1, 1)-padded, stride-2 view of axis -2.
  std::vector<float> x_data;
  for (int n = 0; n < 2; ++n) {
    for (int r = 0; r < 4; ++r) {
      for (int c = 0; c < 3; ++c) {
        x_data.push_back(static_cast<float>(n * 12 + r * 3 + c));
      }
    }
  }
  for (int rep = 0; rep < 4; ++rep) {
    churn(s);
    array x(x_data.begin(), Shape{2, 4, 3}, float32);
    // (2, 4, 3) -> pad rows -> (2, 6, 3), windows of 2 rows stride 2.
    std::vector<float> padded(2 * 6 * 3,
                              std::numeric_limits<float>::lowest());
    for (int n = 0; n < 2; ++n) {
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
          padded[(n * 6 + r + 1) * 3 + c] = x_data[(n * 4 + r) * 3 + c];
        }
      }
    }
    array flat_padded(padded.begin(), Shape{2 * 6 * 3}, float32);
    // rows: windows at padded row offsets 0 and 2 (stride 2), 3 columns.
    array windows = as_strided(flat_padded, {2, 3, 2, 3}, {18, 6, 3, 1}, 0, s);
    array pooled = max(windows, 2, false, s);
    std::vector<float> pool_want;
    for (int n = 0; n < 2; ++n) {
      for (int o = 0; o < 3; ++o) {
        for (int c = 0; c < 3; ++c) {
          float m = std::numeric_limits<float>::lowest();
          for (int t = 0; t < 2; ++t) {
            int r = o * 2 + t - 1;
            if (r >= 0 && r < 4) {
              m = std::max(m, x_data[(n * 4 + r) * 3 + c]);
            }
          }
          pool_want.push_back(m);
        }
      }
    }
    expect_values_conv(read_values(pooled), pool_want);
  }
}
