// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <array>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>
#include <functional>
#include <vector>
#include <numeric>
#include <random>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/compile.h"
#include "mlx/device.h"
#include "mlx/linalg.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

void check_values(
    array value,
    const std::vector<float>& expected,
    const Stream& stream,
    double epsilon = 1e-5) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  const float* values = value.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(values[index] == doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

void check_int32_values(
    array value,
    const std::vector<int32_t>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  const int32_t* values = value.data<int32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_uint32_values(
    array value,
    const std::vector<uint32_t>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  const uint32_t* values = value.data<uint32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_uint16_values(
    array value,
    const std::vector<uint16_t>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  const uint16_t* values = value.data<uint16_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_int64_values(
    array value,
    const std::vector<int64_t>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  const int64_t* values = value.data<int64_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_uint64_values(
    array value,
    const std::vector<uint64_t>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  const uint64_t* values = value.data<uint64_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}


// Checks one batch matrix of a rank-5 output against a host vector.
void check_values(
    array value,
    size_t b1,
    size_t b2,
    const std::vector<float>& expected,
    const Stream& stream,
    double epsilon = 1e-5) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE(value.size() >= (b1 * 2 + b2 + 1) * expected.size());
  const float* values = value.data<float>();
  size_t base = (b1 * 2 + b2) * expected.size();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(
        values[base + index] ==
        doctest::Approx(expected[index]).epsilon(epsilon));
  }
}

std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

// Host threefry2x32 copied from the upstream CPU reference
// mlx/backend/cpu/threefry.cpp: the same rotation constants and 5-round
// key schedule, so GPU words must match bit for bit.
std::pair<uint32_t, uint32_t> host_threefry(
    std::pair<uint32_t, uint32_t> key,
    std::pair<uint32_t, uint32_t> count) {
  constexpr static uint32_t rotations[2][4] = {
      {13, 15, 26, 6}, {17, 29, 16, 24}};

  uint32_t ks[3] = {key.first, key.second, key.first ^ key.second ^ 0x1BD11BDA};

  count.first += ks[0];
  count.second += ks[1];

  for (int i = 0; i < 5; ++i) {
    for (auto r : rotations[i % 2]) {
      count.first += count.second;
      count.second = (count.second << r) | (count.second >> (32 - r));
      count.second ^= count.first;
    }
    count.first += ks[(i + 1) % 3];
    count.second += ks[(i + 2) % 3] + i + 1;
  }

  return count;
}

// Word j of one key's region under the width-4 RandomBits layout, per
// upstream RandomBits::eval_cpu: counters walk (first, second) pairs and
// an odd word count leaves a middle word fed by counter (half, 0).
uint32_t host_random_word(
    std::pair<uint32_t, uint32_t> key,
    uint32_t words,
    uint32_t word) {
  uint32_t half = words / 2;
  bool even = words % 2 == 0;
  std::pair<uint32_t, uint32_t> counter;
  if (word < half) {
    counter = {word, word + half + (even ? 0u : 1u)};
  } else if (even) {
    counter = {word - half, word};
  } else if (word == half) {
    counter = {half, 0u};
  } else {
    counter = {word - half - 1u, word};
  }
  auto bits = host_threefry(key, counter);
  return (word < half || (!even && word == half)) ? bits.first : bits.second;
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

} // namespace

TEST_CASE("FP32 elementwise primitives dispatch through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  auto& counters = omarchy::trace::counters();
  uint64_t compute_before = counters.vk_compute_dispatches.load();
  uint64_t primitives_before = counters.gpu_primitive_dispatches.load();
  array x({1.0f, 2.0f, 4.0f, 8.0f}, float32);
  array y({2.0f, 4.0f, 2.0f, 4.0f}, float32);

  check_values(add(x, y, stream), {3.0f, 6.0f, 6.0f, 12.0f}, stream);
  check_values(multiply(x, y, stream), {2.0f, 8.0f, 8.0f, 32.0f}, stream);
  check_values(divide(x, y, stream), {0.5f, 0.5f, 2.0f, 2.0f}, stream);
  check_values(maximum(x, y, stream), {2.0f, 4.0f, 4.0f, 8.0f}, stream);
  check_values(add(x, array(1.0f), stream), {2.0f, 3.0f, 5.0f, 9.0f}, stream);
  check_values(subtract(x, y, stream), {-1.0f, -2.0f, 2.0f, 4.0f}, stream);
  check_values(
      subtract(x, array(0.5f), stream), {0.5f, 1.5f, 3.5f, 7.5f}, stream);
  check_values(
      subtract(array(1.0f), y, stream), {-1.0f, -3.0f, -1.0f, -3.0f}, stream);

  CHECK(counters.vk_compute_dispatches.load() >= compute_before + 5);
  CHECK(counters.gpu_primitive_dispatches.load() >= primitives_before + 5);
}

TEST_CASE("FP32 unary primitives dispatch through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array x({0.0f, 1.0f, 2.0f}, float32);
  array positive({1.0f, 4.0f, 16.0f}, float32);

  check_values(
      exp(x, stream), {1.0f, std::exp(1.0f), std::exp(2.0f)}, stream, 1e-4);
  check_values(
      sigmoid(x, stream),
      {0.5f, 1.0f / (1.0f + std::exp(-1.0f)),
       1.0f / (1.0f + std::exp(-2.0f))},
      stream,
      1e-4);
  check_values(square(positive, stream), {1.0f, 16.0f, 256.0f}, stream);
  check_values(sqrt(positive, stream), {1.0f, 2.0f, 4.0f}, stream);
  check_values(rsqrt(positive, stream), {1.0f, 0.5f, 0.25f}, stream);
  check_values(negative(x, stream), {0.0f, -1.0f, -2.0f}, stream);
}

TEST_CASE("Log bases match host references at several magnitudes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Magnitudes spanning 1e-3 to 1e5 so a natural-log alias for either
  // base cannot pass. Found by upstream ops_tests.cpp:1709 and :1721:
  // log2/log10 kernels omitted the 1/ln(base) scaling.
  std::vector<float> values{
      1.0e-3f, 0.5f, 1.0f, 2.0f, 10.0f, 1024.0f, 1000.0f, 123456.0f};
  array x(values.begin(), Shape{static_cast<int>(values.size())}, float32);

  std::vector<float> log_expected;
  std::vector<float> log2_expected;
  std::vector<float> log10_expected;
  for (float value : values) {
    log_expected.push_back(std::log(value));
    log2_expected.push_back(std::log2(value));
    log10_expected.push_back(std::log10(value));
  }
  check_values(log(x, stream), log_expected, stream, 1e-6);
  check_values(log2(x, stream), log2_expected, stream, 1e-6);
  check_values(log10(x, stream), log10_expected, stream, 1e-6);

  // Upstream anchors: log2(1024) is exactly 10, log10(1000) is exactly 3.
  check_values(
      log2(array(1024.0f), stream), {10.0f}, stream, 1e-6);
  check_values(
      log10(array(1000.0f), stream), {3.0f}, stream, 1e-6);
}

TEST_CASE("FP16 casts and elementwise primitives use Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }
  array source({0.5f, -2.0f, 7.25f}, float32);
  array half = astype(source, float16, stream);
  array round_trip = astype(half, float32, stream);

  CHECK_EQ(half.dtype(), float16);
  check_values(round_trip, {0.5f, -2.0f, 7.25f}, stream, 1e-3);
  check_values(
      astype(add(half, half, stream), float32, stream),
      {1.0f, -4.0f, 14.5f},
      stream,
      1e-3);
  check_values(
      astype(sum(half, std::vector<int>{0}, false, stream), float32, stream),
      {5.75f},
      stream,
      1e-3);
}

TEST_CASE("BF16 primitives use emulated conversion through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.storage_buffer_16bit_access ||
      !capabilities.shader_int16) {
    skip("Vulkan device lacks required BF16 storage and shader features.");
    return;
  }

  array source({0.5f, -2.0f, 7.25f}, float32);
  array wide = astype(source, bfloat16, stream);
  CHECK_EQ(wide.dtype(), bfloat16);
  check_values(astype(wide, float32, stream), {0.5f, -2.0f, 7.25f}, stream, 8e-3);

  array x({1.5f, 2.5f, -3.25f}, float32);
  array y({0.5f, 1.25f, 0.75f}, float32);
  check_values(
      astype(add(astype(x, bfloat16, stream), astype(y, bfloat16, stream),
                 stream),
             float32, stream),
      {2.0f, 3.75f, -2.5f},
      stream,
      8e-3);

  array grid(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2, 2, 2},
      float32);
  check_values(
      astype(
          sum(astype(grid, bfloat16, stream), std::vector<int>{1, 2}, false,
              stream),
          float32,
          stream),
      {10.0f, 26.0f},
      stream,
      8e-3);

  array a(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
       12.0f},
      {3, 4},
      float32);
  array b({0.5f, 1.0f, 2.0f, 1.0f, 0.5f, 1.0f, 2.0f, 1.0f}, {4, 2}, float32);
  check_values(
      astype(
          matmul(astype(a, bfloat16, stream), astype(b, bfloat16, stream),
                 stream),
          float32,
          stream),
      {14.0f, 10.0f, 34.0f, 26.0f, 54.0f, 42.0f},
      stream,
      8e-3);

  if (!capabilities.shader_float16) {
    skip("Vulkan device lacks shaderFloat16; skipping the BF16/FP16 cast.");
    return;
  }
  array half = astype(source, float16, stream);
  array half_round_trip = astype(
      astype(astype(half, bfloat16, stream), float16, stream), float32,
      stream);
  check_values(half_round_trip, {0.5f, -2.0f, 7.25f}, stream, 8e-3);
}

TEST_CASE("suffix Sum and Max reductions use Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array x(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2, 2, 2},
      float32);

  check_values(sum(x, std::vector<int>{1, 2}, false, stream), {10.0f, 26.0f}, stream);
  check_values(
      max(x, std::vector<int>{2}, false, stream),
      {2.0f, 4.0f, 6.0f, 8.0f},
      stream);
}

TEST_CASE("elementwise on broadcast-expanded views matches host values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Upstream test_reduce.py::test_expand_sums shapes: (5,1,5,1,5,1)
  // broadcast with axes {1,3,5} grown to 5. Broadcast views keep
  // contiguous=true while data_size < size, so every dense consumer must
  // stride-walk the view and write dense output storage.
  std::vector<float> base_values(125);
  std::iota(base_values.begin(), base_values.end(), 1.0f);
  array base(base_values.begin(), Shape{5, 1, 5, 1, 5, 1}, float32);
  array view = broadcast_to(base, Shape{5, 5, 5, 1, 5, 1}, stream);

  std::vector<float> scaled_expected;
  for (int a = 0; a < 5; ++a) {
    for (int b = 0; b < 5; ++b) {
      for (int c = 0; c < 5; ++c) {
        for (int e = 0; e < 5; ++e) {
          scaled_expected.push_back(base_values[a * 25 + c * 5 + e] / 1000.0f);
        }
      }
    }
  }
  // The exact upstream failing expression: sum over the expanded-axis
  // size-1 axes is a graph no-op returning the view, and the divide
  // consumed it as dense (470/625 elements wrong before the fix).
  check_values(
      divide(sum(view, std::vector<int>{3}, false, stream),
             array(1000.0f),
             stream),
      scaled_expected,
      stream,
      1e-6);
  check_values(
      divide(sum(view, std::vector<int>{5}, false, stream),
             array(1000.0f),
             stream),
      scaled_expected,
      stream,
      1e-6);
  check_values(
      divide(sum(view, std::vector<int>{3, 5}, false, stream),
             array(1000.0f),
             stream),
      scaled_expected,
      stream,
      1e-6);
  // The view straight into elementwise, no reduction on top.
  check_values(divide(view, array(1000.0f), stream), scaled_expected, stream, 1e-6);

  // A scalar broadcast view through a unary op: output must be dense,
  // not a mirror of the one-element source buffer.
  array scalar_view =
      broadcast_to(array(2.0f), Shape{4}, stream);
  check_values(
      exp(scalar_view, stream),
      {std::exp(2.0f),
       std::exp(2.0f),
       std::exp(2.0f),
       std::exp(2.0f)},
      stream,
      1e-5);

  // The int elementwise twin shares the output-data setup.
  std::vector<int32_t> int_base_values(5);
  std::iota(int_base_values.begin(), int_base_values.end(), 1);
  array int_base(int_base_values.begin(), Shape{5, 1}, int32);
  array int_view = broadcast_to(int_base, Shape{5, 5}, stream);
  check_int32_values(
      subtract(int_view, array(1, int32), stream),
      {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4,
       4, 4},
      stream);

  // e163a56 lifted the rank cap: a real reduce over the rank-5 view now
  // computes, so pin the axis-0 sum against the host reference instead
  // of a named error. Output shape (5,5,1,5,1): element (b,c,e) sums
  // base over the expanded axis 0.
  std::vector<float> sum_expected;
  for (int b = 0; b < 5; ++b) {
    for (int c = 0; c < 5; ++c) {
      for (int e = 0; e < 5; ++e) {
        float total = 0.0f;
        for (int a = 0; a < 5; ++a) {
          total += base_values[a * 25 + c * 5 + e];
        }
        sum_expected.push_back(total);
      }
    }
  }
  check_values(
      sum(view, std::vector<int>{0}, false, stream),
      sum_expected,
      stream,
      1e-6);
}

TEST_CASE("col-contiguous view Add forces General output storage") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Two gapless transposes whose result layout is col-contiguous hit the
  // VectorVector branch of get_binary_op_type. Donating a donor's
  // transposed strides to a linearly written output put correct values
  // at scrambled positions (sum preserved). The Qwen3-TTS codec
  // quantizer decode summed exactly this shape, (1, 512, 41), and the
  // M2 voice output degraded to a hum. The donation guard must force
  // the General path: dense output storage, stride-aware reads.
  std::vector<float> base1_values(41 * 512);
  std::iota(base1_values.begin(), base1_values.end(), 0.0f);
  for (float& value : base1_values) {
    value = value * 0.25f - 5000.0f;
  }
  std::vector<float> base2_values(41 * 512);
  std::iota(base2_values.begin(), base2_values.end(), 0.0f);
  for (float& value : base2_values) {
    value = value * 0.125f - 2500.0f;
  }
  array base1(base1_values.begin(), Shape{41, 512}, float32);
  array base2(base2_values.begin(), Shape{41, 512}, float32);
  array view1 = transpose(base1, {1, 0}, stream);
  array view2 = transpose(base2, {1, 0}, stream);

  std::vector<float> expected(41 * 512);
  for (int col = 0; col < 41; ++col) {
    for (int row = 0; row < 512; ++row) {
      expected[row * 41 + col] =
          base1_values[col * 512 + row] + base2_values[col * 512 + row];
    }
  }
  check_values(add(view1, view2, stream), expected, stream, 1e-4);

  // The 3-D quantizer shape: two (1, 41, 512) views transposed to
  // (1, 512, 41) are col-contiguous in the same way.
  std::vector<float> base3_values(1 * 41 * 512);
  std::iota(base3_values.begin(), base3_values.end(), 0.0f);
  for (float& value : base3_values) {
    value = std::fmod(value * 0.031f, 7.0f);
  }
  std::vector<float> base4_values(1 * 41 * 512);
  std::iota(base4_values.begin(), base4_values.end(), 0.0f);
  for (float& value : base4_values) {
    value = std::fmod(value * 0.017f, 5.0f);
  }
  array base3(base3_values.begin(), Shape{1, 41, 512}, float32);
  array base4(base4_values.begin(), Shape{1, 41, 512}, float32);
  array view3 = transpose(base3, {0, 2, 1}, stream);
  array view4 = transpose(base4, {0, 2, 1}, stream);
  std::vector<float> expected3(1 * 41 * 512);
  for (int col = 0; col < 41; ++col) {
    for (int row = 0; row < 512; ++row) {
      expected3[row * 41 + col] =
          base3_values[col * 512 + row] + base4_values[col * 512 + row];
    }
  }
  check_values(add(view3, view4, stream), expected3, stream, 1e-5);

  // The int elementwise family shares the guarded helper.
  std::vector<int32_t> int1_values(64);
  std::vector<int32_t> int2_values(64);
  std::iota(int1_values.begin(), int1_values.end(), 0);
  std::iota(int2_values.rbegin(), int2_values.rend(), 0);
  array int1(int1_values.begin(), Shape{8, 8}, int32);
  array int2(int2_values.begin(), Shape{8, 8}, int32);
  std::vector<int32_t> int_expected(64);
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 8; ++col) {
      int_expected[row * 8 + col] =
          int1_values[col * 8 + row] + int2_values[col * 8 + row];
    }
  }
  check_int32_values(
      add(
          transpose(int1, {1, 0}, stream),
          transpose(int2, {1, 0}, stream),
          stream),
      int_expected,
      stream);
}

TEST_CASE("compute indexing stays inside Vulkan and uint32 limits") {
  constexpr uint32_t max_u32 = std::numeric_limits<uint32_t>::max();

  CHECK_EQ(omarchy::compute_dispatch_group_count(0), 0);
  CHECK_EQ(omarchy::compute_dispatch_group_count(256), 1);
  CHECK_EQ(omarchy::compute_dispatch_group_count(257), 2);
  CHECK_EQ(omarchy::compute_dispatch_group_count(max_u32), 65535);
  CHECK(omarchy::compute_index_span_fits(0, max_u32));
  CHECK(omarchy::compute_index_span_fits(max_u32, 1));
  CHECK_FALSE(omarchy::compute_index_span_fits(max_u32, 2));
  CHECK(omarchy::compute_index_span_fits(1, max_u32));
  CHECK_FALSE(omarchy::compute_index_span_fits(2, max_u32));
}

TEST_CASE("compute shaders handle offsets, multiple workgroups, and NaN") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> values(300);
  std::vector<float> expected(300);
  for (size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<float>(index);
    expected[index] = values[index] + 1.0f;
  }
  array many(values.begin(), Shape{300}, float32);
  check_values(add(many, array(1.0f), stream), expected, stream);

  array source = add(many, array(0.0f), stream);
  source.eval();
  auto& encoder = omarchy::get_command_encoder(stream);
  encoder.synchronize();
  array output = multiply(source, array(0.0f), stream);
  output.eval();
  encoder.synchronize();
  auto binding = [](const array& value) {
    auto* buffer =
        static_cast<const omarchy::VulkanBuffer*>(value.buffer().ptr());
    return omarchy::ComputeBinding{buffer->buffer, 0, buffer->size};
  };
  omarchy::ComputeParams params;
  params.count = 300;
  params.operation = 0;
  params.lhs_size = 300;
  params.rhs_size = 300;
  params.output_size = 300;
  std::array<omarchy::ComputeBinding, 3> bindings{
      binding(source), binding(source), binding(output)};
  encoder.dispatch_compute(
      omarchy::ComputeKernel::ElementwiseF32, bindings, params, 1);
  encoder.synchronize();
  CHECK_EQ(output.data<float>()[0], 0.0f);
  CHECK_EQ(output.data<float>()[256], 512.0f);
  CHECK_EQ(output.data<float>()[299], 598.0f);

  array base({-9.0f, 0.0f, 1.0f, 2.0f, 3.0f}, float32);
  array offset = slice(base, {1}, {5}, {1}, stream);
  check_values(square(offset, stream), {0.0f, 1.0f, 4.0f, 9.0f}, stream);

  float nan = std::numeric_limits<float>::quiet_NaN();
  array nan_result = max(
      array({1.0f, nan, 3.0f}, float32),
      std::vector<int>{0},
      false,
      stream);
  nan_result.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK(std::isnan(nan_result.data<float>()[0]));
}

TEST_CASE("FP32 and FP16 Matmul support dense and transposed weights") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  array b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}, float32);
  array weights(
      {7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f}, {2, 3}, float32);

  check_values(matmul(a, b, stream), {58.0f, 64.0f, 139.0f, 154.0f}, stream);
  check_values(
      matmul(a, transpose(weights, {1, 0}, stream), stream),
      {58.0f, 64.0f, 139.0f, 154.0f},
      stream);

  array stored({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2}, float32);
  check_values(
      matmul(
          transpose(stored, {1, 0}, stream),
          array({7.0f, 10.0f, 8.0f, 11.0f, 9.0f, 12.0f}, {3, 2}, float32),
          stream),
      {76.0f, 103.0f, 100.0f, 136.0f},
      stream);

  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    check_values(
        astype(
            matmul(
                astype(a, float16, stream),
                astype(b, float16, stream),
                stream),
            float32,
            stream),
        {58.0f, 64.0f, 139.0f, 154.0f},
        stream,
        1e-3);
    check_values(
        astype(
            addmm(
                astype(array({1.0f, 2.0f}, float32), float16, stream),
                astype(a, float16, stream),
                astype(b, float16, stream),
                2.0f,
                0.5f,
                stream),
            float32,
            stream),
        {116.5f, 129.0f, 278.5f, 309.0f},
        stream,
        1e-3);
  }
}

TEST_CASE("Matmul flattens leading batches and AddMM broadcasts bias") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array a(
      {1.0f,
       2.0f,
       3.0f,
       4.0f,
       5.0f,
       6.0f,
       7.0f,
       8.0f,
       9.0f,
       10.0f,
       11.0f,
       12.0f},
      {2, 2, 3},
      float32);
  array b({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, {3, 2}, float32);
  array product = matmul(a, b, stream);
  check_values(
      product,
      {58.0f, 64.0f, 139.0f, 154.0f, 220.0f, 244.0f, 301.0f, 334.0f},
      stream);
  check_values(
      add(product, array({1.0f, 2.0f}, float32), stream),
      {59.0f, 66.0f, 140.0f, 156.0f, 221.0f, 246.0f, 302.0f, 336.0f},
      stream);
  check_values(
      addmm(array({1.0f, 2.0f}, float32), a, b, 2.0f, 0.5f, stream),
      {116.5f, 129.0f, 278.5f, 309.0f, 440.5f, 489.0f, 602.5f, 669.0f},
      stream);
}

TEST_CASE("Matmul handles tiled edges, offsets, and empty K") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr size_t m = 17;
  constexpr size_t k = 18;
  constexpr size_t n = 19;
  std::vector<float> a_values(m * k);
  std::vector<float> b_values(k * n);
  std::vector<float> expected(m * n, 0.0f);
  for (size_t index = 0; index < a_values.size(); ++index) {
    a_values[index] = static_cast<float>(static_cast<int>(index % 7) - 3);
  }
  for (size_t index = 0; index < b_values.size(); ++index) {
    b_values[index] = static_cast<float>(static_cast<int>(index % 5) - 2);
  }
  for (size_t row = 0; row < m; ++row) {
    for (size_t column = 0; column < n; ++column) {
      for (size_t inner = 0; inner < k; ++inner) {
        expected[row * n + column] +=
            a_values[row * k + inner] * b_values[inner * n + column];
      }
    }
  }
  check_values(
      matmul(
          array(a_values.begin(), Shape{m, k}, float32),
          array(b_values.begin(), Shape{k, n}, float32),
          stream),
      expected,
      stream);

  std::vector<float> transposed_expected(m * n, 0.0f);
  for (size_t row = 0; row < m; ++row) {
    for (size_t column = 0; column < n; ++column) {
      for (size_t inner = 0; inner < k; ++inner) {
        transposed_expected[row * n + column] +=
            a_values[inner * m + row] * b_values[inner * n + column];
      }
    }
  }
  check_values(
      matmul(
          transpose(array(a_values.begin(), Shape{k, m}, float32), {1, 0},
                    stream),
          array(b_values.begin(), Shape{k, n}, float32),
          stream),
      transposed_expected,
      stream);

  array a_base(
      {99.0f, 99.0f, 99.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
      {3, 3},
      float32);
  array b_base(
      {99.0f, 99.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
      {4, 2},
      float32);
  array a_offset = slice(a_base, {1, 0}, {3, 3}, {1, 1}, stream);
  array b_offset = slice(b_base, {1, 0}, {4, 2}, {1, 1}, stream);
  check_values(
      matmul(a_offset, b_offset, stream),
      {58.0f, 64.0f, 139.0f, 154.0f},
      stream);

  std::vector<float> empty;
  array empty_a(empty.begin(), Shape{2, 0}, float32);
  array empty_b(empty.begin(), Shape{0, 3}, float32);
  check_values(
      matmul(empty_a, empty_b, stream),
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
      stream);
  check_values(
      addmm(
          array({1.0f, 2.0f, 3.0f}, float32),
          empty_a,
          empty_b,
          1.0f,
          2.0f,
          stream),
      {2.0f, 4.0f, 6.0f, 2.0f, 4.0f, 6.0f},
      stream);
}

TEST_CASE("non-zero scalar fills dispatch through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  check_values(
      full({2, 3}, 1.5f, float32, stream),
      std::vector<float>(6, 1.5f),
      stream);
  check_values(full({2}, 0.0f, float32, stream), {0.0f, 0.0f}, stream);

  // IntegerScalarFills: int32 and uint32 fill through the raw-word
  // path bit-exactly; int64 rides the 64-bit fill behind shaderInt64.
  check_int32_values(full({2}, 5, int32, stream), {5, 5}, stream);
  check_uint32_values(
      full({2}, 4294967295u, uint32, stream),
      {4294967295u, 4294967295u},
      stream);
  array wide_i64 = full({2}, 5, int64, stream);
  wide_i64.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(wide_i64.data<int64_t>()[0], int64_t(5));
  CHECK_EQ(wide_i64.data<int64_t>()[1], int64_t(5));

  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    array half_out = zeros({2, 2}, float16, stream);
    half_out.eval();
    array half_scalar = array(1.5f, float16);
    copy_gpu_inplace(
        half_scalar,
        half_out,
        half_out.shape(),
        half_scalar.strides(),
        half_out.strides(),
        0,
        0,
        CopyType::Scalar,
        stream);
    check_values(
        astype(half_out, float32, stream),
        std::vector<float>(4, 1.5f),
        stream,
        1e-3);
  }
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array bf_out = zeros({3}, bfloat16, stream);
    bf_out.eval();
    array bf_scalar = array(2.5f, bfloat16);
    copy_gpu_inplace(
        bf_scalar,
        bf_out,
        bf_out.shape(),
        bf_scalar.strides(),
        bf_out.strides(),
        0,
        0,
        CopyType::Scalar,
        stream);
    check_values(
        astype(bf_out, float32, stream), {2.5f, 2.5f, 2.5f}, stream, 8e-3);
  }
}

TEST_CASE("general strided copies materialize through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array base({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, {2, 3}, float32);
  check_values(
      contiguous(transpose(base, stream), false, stream),
      {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f},
      stream);

  array wide(
      {0.0f,
       1.0f,
       2.0f,
       3.0f,
       4.0f,
       5.0f,
       6.0f,
       7.0f,
       8.0f,
       9.0f,
       10.0f,
       11.0f},
      {3, 4},
      float32);
  check_values(
      contiguous(slice(wide, {0, 1}, {3, 4}, {2, 1}, stream), false, stream),
      {1.0f, 2.0f, 3.0f, 9.0f, 10.0f, 11.0f},
      stream);

  array grid(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2, 2, 2},
      float32);
  check_values(
      contiguous(transpose(grid, {2, 0, 1}, stream), false, stream),
      {1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f},
      stream);

  array ints({0, 1, 2, 3}, {2, 2}, int32);
  check_int32_values(
      contiguous(transpose(ints, stream), false, stream),
      {0, 2, 1, 3},
      stream);
}

TEST_CASE("value_and_grad computes matmul and subtract gradients on device") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array x({0.0f, 0.1f, 0.2f, 0.05f, 0.15f, 0.25f}, {2, 3}, float32);
  array w({0.1f, -0.2f, 0.3f, 0.05f, -0.1f, 0.2f}, {3, 2}, float32);

  auto fun = [&](const std::vector<array>& inputs) {
    return sum(exp(matmul(inputs[0], inputs[1], stream), stream), stream);
  };
  auto [value, grads] = value_and_grad(fun, std::vector<int>{0, 1})({x, w});

  std::vector<float> xv = {0.0f, 0.1f, 0.2f, 0.05f, 0.15f, 0.25f};
  std::vector<float> wv = {0.1f, -0.2f, 0.3f, 0.05f, -0.1f, 0.2f};
  float expected_value = 0.0f;
  float e[2][2];
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 2; ++column) {
      float y = 0.0f;
      for (int inner = 0; inner < 3; ++inner) {
        y += xv[row * 3 + inner] * wv[inner * 2 + column];
      }
      e[row][column] = std::exp(y);
      expected_value += e[row][column];
    }
  }
  std::vector<float> expected_dx(6);
  for (int row = 0; row < 2; ++row) {
    for (int inner = 0; inner < 3; ++inner) {
      float grad = 0.0f;
      for (int column = 0; column < 2; ++column) {
        grad += e[row][column] * wv[inner * 2 + column];
      }
      expected_dx[row * 3 + inner] = grad;
    }
  }
  std::vector<float> expected_dw(6);
  for (int inner = 0; inner < 3; ++inner) {
    for (int column = 0; column < 2; ++column) {
      float grad = 0.0f;
      for (int row = 0; row < 2; ++row) {
        grad += xv[row * 3 + inner] * e[row][column];
      }
      expected_dw[inner * 2 + column] = grad;
    }
  }
  check_values(value, {expected_value}, stream, 1e-4);
  check_values(grads.at(0), expected_dx, stream, 1e-4);
  check_values(grads.at(1), expected_dw, stream, 1e-4);

  auto sub_fun = [&](const std::vector<array>& inputs) {
    return sum(subtract(inputs[0], inputs[1], stream), stream);
  };
  auto [sub_value, sub_grads] = value_and_grad(sub_fun, std::vector<int>{0, 1})(
      {x, reshape(w, {2, 3}, stream)});
  check_values(sub_value, {0.4f}, stream, 1e-5);
  // The positive cotangent is a broadcast view of the scalar seed, so
  // materialize it on device before reading linearly.
  check_values(
      multiply(sub_grads.at(0), array(1.0f), stream),
      std::vector<float>(6, 1.0f),
      stream);
  check_values(sub_grads.at(1), std::vector<float>(6, -1.0f), stream);
}

TEST_CASE("jvp computes forward-mode tangents through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array x({0.0f, 0.1f, 0.2f}, float32);
  array tangent({1.0f, -2.0f, 3.0f}, float32);

  auto exp_fun = [&](const array& input) {
    return sum(exp(input, stream), stream);
  };
  auto [exp_value, exp_tangent] = jvp(exp_fun, x, tangent);
  std::vector<float> ex = {
      std::exp(0.0f), std::exp(0.1f), std::exp(0.2f)};
  float expected_value = ex[0] + ex[1] + ex[2];
  float expected_jvp = ex[0] - 2.0f * ex[1] + 3.0f * ex[2];
  check_values(exp_value, {expected_value}, stream, 1e-4);
  check_values(exp_tangent, {expected_jvp}, stream, 1e-4);

  std::vector<float> av = {0.0f, 0.1f, 0.2f, 0.05f, 0.15f, 0.25f};
  std::vector<float> wv = {0.1f, -0.2f, 0.3f, 0.05f, -0.1f, 0.2f};
  std::vector<float> tv = {1.0f, 0.5f, -1.0f, 2.0f, 0.25f, -0.5f};
  array a(av.begin(), Shape{2, 3}, float32);
  array w(wv.begin(), Shape{3, 2}, float32);
  array a_tangent(tv.begin(), Shape{2, 3}, float32);
  auto matmul_fun = [&](const array& input) {
    return matmul(input, w, stream);
  };
  auto [mm_value, mm_tangent] = jvp(matmul_fun, a, a_tangent);
  std::vector<float> expected_product(4);
  std::vector<float> expected_mm_jvp(4);
  for (int row = 0; row < 2; ++row) {
    for (int column = 0; column < 2; ++column) {
      for (int inner = 0; inner < 3; ++inner) {
        expected_product[row * 2 + column] +=
            av[row * 3 + inner] * wv[inner * 2 + column];
        expected_mm_jvp[row * 2 + column] +=
            tv[row * 3 + inner] * wv[inner * 2 + column];
      }
    }
  }
  check_values(mm_value, expected_product, stream, 1e-4);
  check_values(mm_tangent, expected_mm_jvp, stream, 1e-4);
}

TEST_CASE("vmap batches closures over the leading axis through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  array x(xv.begin(), Shape{2, 3}, float32);

  auto batched_exp = vmap(
      [&](const array& input) { return exp(input, stream); });
  check_values(
      batched_exp(x),
      {std::exp(xv[0]),
       std::exp(xv[1]),
       std::exp(xv[2]),
       std::exp(xv[3]),
       std::exp(xv[4]),
       std::exp(xv[5])},
      stream,
      1e-4);

  std::vector<float> yv = {1.0f, -0.5f, 2.0f, 0.25f, -1.0f, 0.75f};
  array y(yv.begin(), Shape{2, 3}, float32);
  auto batched_add = vmap(
      [&](const array& lhs, const array& rhs) {
        return add(lhs, rhs, stream);
      });
  std::vector<float> expected_add(6);
  for (size_t index = 0; index < expected_add.size(); ++index) {
    expected_add[index] = xv[index] + yv[index];
  }
  check_values(batched_add(x, y), expected_add, stream);

  std::vector<float> av = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> bv = {0.5f, 1.0f, 2.0f, 1.0f, 0.5f, 1.0f, 2.0f, 1.0f};
  array batched_a(av.begin(), Shape{2, 2, 2}, float32);
  array batched_b(bv.begin(), Shape{2, 2, 2}, float32);
  auto batched_matmul = vmap(
      [&](const array& lhs, const array& rhs) {
        return matmul(lhs, rhs, stream);
      });
  // Equal leading batch dims dispatch as one batched product now.
  check_values(
      batched_matmul(batched_a, batched_b),
      {4.5f, 3.0f, 9.5f, 7.0f, 14.5f, 11.0f, 19.5f, 15.0f},
      stream);
}

TEST_CASE("mx.compile evaluates the elementwise tape and no_fuse still matches") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.1f, 0.2f, 0.3f};
  std::vector<float> yv = {1.0f, 0.5f, 2.0f, 0.25f};
  array x(xv.begin(), Shape{4}, float32);
  array y(yv.begin(), Shape{4}, float32);
  std::vector<float> expected(4);
  for (size_t index = 0; index < expected.size(); ++index) {
    expected[index] = std::exp(xv[index]) * yv[index];
  }
  using VectorFn = std::function<std::vector<array>(const std::vector<array>&)>;

  // The fused tape interprets on the GPU and matches the host reference.
  set_compile_mode(CompileMode::enabled);
  VectorFn fused_fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        multiply(exp(inputs[0], stream), inputs[1], stream)};
  };
  auto fused = compile(fused_fun);
  check_values(fused({x, y})[0], expected, stream, 1e-5);

  // no_fuse keeps the tape unfused and must stay green.
  set_compile_mode(CompileMode::no_fuse);
  VectorFn unfused_fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        multiply(exp(inputs[0], stream), inputs[1], stream)};
  };
  auto unfused = compile(unfused_fun);
  check_values(unfused({x, y})[0], expected, stream, 1e-5);
  set_compile_mode(CompileMode::enabled);
}

TEST_CASE("unsupported compute shapes and dtypes refuse by name") {
  if (!compute_available()) {
    return;
  }
  if (!is_available(Device::cpu)) {
    // No CPU backend in this build: the CPU device does not exist at all.
    CHECK_FALSE(is_available(Device::cpu));
    CHECK_EQ(device_count(Device::cpu), 0);
  }
  Stream stream = gpu_stream();

  // Integer Add, Multiply, and Square computed in this wave, so this no
  // longer pins a refusal. Like the complex64 case below it stays as an
  // exact check, here against independent host uint32 arithmetic: the
  // operands sit on every wrap boundary, so a float round-trip or a
  // wrong wrap convention cannot pass.
  std::vector<uint32_t> au_v = {0u, 1u, 2147483647u, 4294967290u};
  std::vector<uint32_t> bu_v = {0u, 2u, 1u, 12u};
  array au(au_v.begin(), Shape{4}, uint32);
  array bu(bu_v.begin(), Shape{4}, uint32);
  std::vector<uint32_t> u_sum(au_v.size());
  std::vector<uint32_t> u_prod(au_v.size());
  std::vector<uint32_t> u_square(au_v.size());
  for (size_t i = 0; i < au_v.size(); ++i) {
    u_sum[i] = au_v[i] + bu_v[i];
    u_prod[i] = au_v[i] * bu_v[i];
    u_square[i] = au_v[i] * au_v[i];
  }
  check_uint32_values(add(au, bu, stream), u_sum, stream);
  check_uint32_values(multiply(au, bu, stream), u_prod, stream);
  check_uint32_values(square(au, stream), u_square, stream);

  std::vector<int32_t> ai_v = {2147483647, -2147483648, -1, 12345};
  std::vector<int32_t> bi_v = {1, -1, 2147483647, -54321};
  array ai(ai_v.begin(), Shape{4}, int32);
  array bi(bi_v.begin(), Shape{4}, int32);
  std::vector<int32_t> i_sum(ai_v.size());
  std::vector<int32_t> i_prod(ai_v.size());
  std::vector<int32_t> i_square(ai_v.size());
  for (size_t i = 0; i < ai_v.size(); ++i) {
    // Signed wrap computed in unsigned host arithmetic, then cast
    // back, so the reference itself never overflows a signed type.
    i_sum[i] = static_cast<int32_t>(
        static_cast<uint32_t>(ai_v[i]) + static_cast<uint32_t>(bi_v[i]));
    i_prod[i] = static_cast<int32_t>(
        static_cast<uint32_t>(ai_v[i]) * static_cast<uint32_t>(bi_v[i]));
    i_square[i] = static_cast<int32_t>(
        static_cast<uint32_t>(ai_v[i]) * static_cast<uint32_t>(ai_v[i]));
  }
  check_int32_values(add(ai, bi, stream), i_sum, stream);
  check_int32_values(multiply(ai, bi, stream), i_prod, stream);
  check_int32_values(square(ai, stream), i_square, stream);

  // Slice views with gaps materialize at eval, so elementwise work over
  // them runs. A transpose view keeps its strides and pins the named
  // layout error.
  array base({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  check_values(
      exp(slice(base, {0, 1}, {2, 3}, {1, 2}, stream), stream),
      {std::exp(2.0f), std::exp(5.0f)},
      stream,
      1e-4);
  // A transposed view is gapless and keeps its strides, so the
  // elementwise stride path reads it directly with no copy.
  check_values(
      exp(transpose(base, stream), stream),
      {std::exp(1.0f),
       std::exp(4.0f),
       std::exp(2.0f),
       std::exp(5.0f),
       std::exp(3.0f),
       std::exp(6.0f)},
      stream,
      1e-4);

  // Ten-axis float and integer broadcasts use external axis metadata
  // rather than a fixed-rank push-constant table.
  const Shape base_shape{2, 1, 2, 1, 2, 1, 2, 1, 2, 1};
  const Shape broadcast_shape{2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<float> lhs_values(32);
  std::iota(lhs_values.begin(), lhs_values.end(), 1.0f);
  array lhs(lhs_values.begin(), base_shape, float32);
  std::vector<float> wide_values(1024, 1.0f);
  array rhs(wide_values.begin(), broadcast_shape, float32);
  std::vector<float> broadcast_expected(1024);
  std::vector<size_t> broadcast_sources(1024);
  for (size_t flat = 0; flat < broadcast_expected.size(); ++flat) {
    size_t rem = flat;
    size_t source = 0;
    size_t source_stride = 16;
    for (size_t axis = 0; axis < 10; ++axis) {
      size_t coord = rem >> (9 - axis);
      rem &= (size_t{1} << (9 - axis)) - 1;
      if (axis % 2 == 0) {
        source += coord * source_stride;
        source_stride >>= 1;
      }
    }
    broadcast_sources[flat] = source;
    broadcast_expected[flat] = lhs_values[source] + 1.0f;
  }
  check_values(add(lhs, rhs, stream), broadcast_expected, stream);

  std::vector<int32_t> int_lhs_values(32);
  std::iota(int_lhs_values.begin(), int_lhs_values.end(), -16);
  std::vector<int32_t> int_rhs_values(1024);
  std::vector<int32_t> int_expected(1024);
  for (size_t flat = 0; flat < int_rhs_values.size(); ++flat) {
    int_rhs_values[flat] = static_cast<int32_t>(flat % 23) - 11;
    int_expected[flat] =
        int_lhs_values[broadcast_sources[flat]] + int_rhs_values[flat];
  }
  check_int32_values(
      add(
          array(int_lhs_values.begin(), base_shape, int32),
          array(int_rhs_values.begin(), broadcast_shape, int32),
          stream),
      int_expected,
      stream);

  // Leading-axis reduction now computes through the general kernel.
  array matrix({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, float32);
  check_values(sum(matrix, 0, false, stream), {4.0f, 6.0f}, stream);

  auto construction_error = [](auto&& build) -> std::string {
    try {
      build();
    } catch (const std::exception& error) {
      return error.what();
    }
    return {};
  };

  std::string float64_error = construction_error([&] {
    add(array({1.0, 2.0}, float64), array({3.0, 4.0}, float64), stream);
  });
  CHECK(
      float64_error.find("float64 is not supported on the GPU") !=
      std::string::npos);

  // complex64 Add computed as of 2026-09-02, when complex64 transport
  // landed, so this no longer pins a refusal. It stays as a value check
  // because an unsupported-dtype case is exactly where a regression to
  // "refuses again" would otherwise pass unnoticed.
  array complex_sum =
      add(array(complex64_t{1.0f, 2.0f}), array(complex64_t{3.0f, 4.0f}),
          stream);
  complex_sum.eval();
  CHECK(complex_sum.dtype() == complex64);
  complex64_t complex_value = complex_sum.data<complex64_t>()[0];
  CHECK(complex_value.real() == doctest::Approx(4.0f));
  CHECK(complex_value.imag() == doctest::Approx(6.0f));

  // The linalg family now dispatches to the Omarchy GPU backend. These
  // pins ride float64, which the API layer admits and this backend will
  // never carry, so the named dtype refusal holds even as the linalg
  // implementations land.
  array spd64({4.0, 0.0, 0.0, 9.0}, {2, 2}, float64);
  std::string cholesky_error = construction_error([&] {
    linalg::cholesky(spd64, false, stream).eval();
  });
  CHECK(cholesky_error.find("float64") != std::string::npos);

  std::string svd_error = construction_error([&] {
    linalg::svd(spd64, true, stream).at(0).eval();
  });
  CHECK(svd_error.find("float64") != std::string::npos);

  std::string inv_error = construction_error([&] {
    linalg::inv(spd64, stream).eval();
  });
  CHECK(inv_error.find("float64") != std::string::npos);
}

TEST_CASE(
    "integer Add Multiply Square cover scalar broadcast, transposed"
    " views, and empty arrays") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Scalar broadcast rides the normalizer's size-1 fast path.
  std::vector<int32_t> rv = {1, -2, 3};
  array r(rv.begin(), Shape{3}, int32);
  check_int32_values(add(r, array(1, int32), stream), {2, -1, 4}, stream);
  check_int32_values(
      multiply(r, array(-2, int32), stream), {-2, 4, -6}, stream);

  // A transposed view keeps its strides and feeds the general
  // broadcast transport, so the add must read it strided, not dense.
  std::vector<int32_t> mv = {1, 2, 3, 4, 5, 6};
  array m(mv.begin(), Shape{2, 3}, int32);
  array mt = transpose(m, stream);
  check_int32_values(add(mt, mt, stream), {2, 8, 4, 10, 6, 12}, stream);

  // Empty operands return empty results and touch no kernel.
  std::vector<int32_t> empty_values;
  array empty(empty_values.begin(), Shape{0}, int32);
  auto empty_sum = add(empty, empty, stream);
  empty_sum.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(empty_sum.shape(), Shape{0});
  CHECK_EQ(empty_sum.dtype(), int32);

  // uint32 shares the route; one wrapped vector add pins the dtype.
  std::vector<uint32_t> uv = {4294967290u, 7u};
  array u(uv.begin(), Shape{2}, uint32);
  check_uint32_values(add(u, u, stream), {4294967284u, 14u}, stream);
}

// Host reference: numerically stable softmax over the last axis.
std::vector<float> host_softmax(
    const std::vector<float>& values, int rows, int row_length) {
  std::vector<float> expected(values.size());
  for (int row = 0; row < rows; ++row) {
    const float* input = values.data() + row * row_length;
    float* output = expected.data() + row * row_length;
    float maximum = input[0];
    for (int index = 1; index < row_length; ++index) {
      maximum = std::max(maximum, input[index]);
    }
    float normalizer = 0.0f;
    for (int index = 0; index < row_length; ++index) {
      output[index] = std::exp(input[index] - maximum);
      normalizer += output[index];
    }
    for (int index = 0; index < row_length; ++index) {
      output[index] /= normalizer;
    }
  }
  return expected;
}

// Host reference: numerically stable log-sum-exp over the last axis, one
// value per row. An infinite row max is already the answer.
std::vector<float> host_logsumexp(
    const std::vector<float>& values, int rows, int row_length) {
  std::vector<float> expected(rows);
  for (int row = 0; row < rows; ++row) {
    const float* input = values.data() + row * row_length;
    float maximum = input[0];
    for (int index = 1; index < row_length; ++index) {
      maximum = std::max(maximum, input[index]);
    }
    if (std::isinf(maximum)) {
      expected[row] = maximum;
      continue;
    }
    float sum = 0.0f;
    for (int index = 0; index < row_length; ++index) {
      sum += std::exp(input[index] - maximum);
    }
    expected[row] = maximum + std::log(sum);
  }
  return expected;
}

TEST_CASE("softmax normalizes rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  array x(xv.begin(), Shape{2, 3}, float32);
  check_values(
      softmax(x, std::vector<int>{-1}, false, stream),
      host_softmax(xv, 2, 3),
      stream);

  // Rows sum to 1.
  check_values(
      sum(softmax(x, std::vector<int>{-1}, false, stream),
          std::vector<int>{-1},
          false,
          stream),
      {1.0f, 1.0f},
      stream);

  // Large logits stay finite because the kernel subtracts the row max.
  std::vector<float> bigv = {80.0f, 81.0f, 80.0f, 79.0f};
  check_values(
      softmax(array(bigv.begin(), Shape{2, 2}, float32),
              std::vector<int>{-1},
              false,
              stream),
      host_softmax(bigv, 2, 2),
      stream);

  // One flat row longer than one workgroup.
  std::vector<float> wide_values(300);
  for (size_t index = 0; index < wide_values.size(); ++index) {
    wide_values[index] =
        static_cast<float>(static_cast<int>(index % 7) - 3);
  }
  check_values(
      softmax(array(wide_values.begin(), Shape{1, 300}, float32),
              std::vector<int>{-1},
              false,
              stream),
      host_softmax(wide_values, 1, 300),
      stream,
      2e-5);

  // More rows than one dispatch can name, so workgroups grid-stride.
  std::vector<float> tall_values(70000, 3.0f);
  check_values(
      softmax(array(tall_values.begin(), Shape{70000, 1}, float32),
              std::vector<int>{-1},
              false,
              stream),
      std::vector<float>(70000, 1.0f),
      stream);

  // Slice views with gaps materialize at eval, so softmax over them runs
  // and normalizes the sliced rows. A transpose view keeps its strides
  // and pins the named layout error.
  array base(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
      {3, 3},
      float32);
  check_values(
      softmax(slice(base, {0, 1}, {3, 3}, {1, 1}, stream),
              std::vector<int>{-1},
              false,
              stream),
      {0.26894142f,
       0.73105858f,
       0.26894142f,
       0.73105858f,
       0.26894142f,
       0.73105858f},
      stream,
      1e-5);
  // Non-suffix axes decompose softmax into general reductions, which now
  // compute; the result matches a host softmax over those axes.
  array grid(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      {2, 2, 2},
      float32);
  array wide_softmax = softmax(grid, std::vector<int>{0, 1}, false, stream);
  std::vector<float> softmax_expected;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        float value = static_cast<float>(i * 4 + j * 2 + k + 1);
        float peak = k == 0 ? 7.0f : 8.0f;
        float total = 0.0f;
        for (int a = 0; a < 2; ++a) {
          for (int b = 0; b < 2; ++b) {
            total +=
                std::exp(static_cast<float>(a * 4 + b * 2 + k + 1) - peak);
          }
        }
        softmax_expected.push_back(std::exp(value - peak) / total);
      }
    }
  }
  check_values(wide_softmax, softmax_expected, stream, 1e-5);
}

TEST_CASE("FP16 and BF16 softmax match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  std::vector<float> xv = {0.0f, 1.0f, 2.0f, 1.0f, 3.0f, -1.0f};
  auto expected = host_softmax(xv, 2, 3);
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    array half = astype(
        array(xv.begin(), Shape{2, 3}, float32), float16, stream);
    check_values(
        astype(
            softmax(half, std::vector<int>{-1}, false, stream),
            float32,
            stream),
        expected,
        stream,
        1e-3);
  } else {
    skip("Vulkan device lacks required FP16 shader and storage features.");
  }
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array brain = astype(
        array(xv.begin(), Shape{2, 3}, float32), bfloat16, stream);
    check_values(
        astype(
            softmax(brain, std::vector<int>{-1}, false, stream),
            float32,
            stream),
        expected,
        stream,
        8e-3);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }
}

TEST_CASE("logsumexp reduces last-axis rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Multi-row case at the default float32 tolerance.
  std::vector<float> xv = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f,
                           -1.0f, -2.0f, -3.0f, -4.0f, -5.0f,
                           10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
  array x(xv.begin(), Shape{3, 5}, float32);
  check_values(
      logsumexp(x, -1, false, stream),
      host_logsumexp(xv, 3, 5),
      stream);

  // The mlx-lm logprobs epilogue: [1, V] logits reduce keepdims to
  // [1, 1] so a broadcast subtract normalizes every row.
  std::vector<float> logits_values(16);
  for (size_t index = 0; index < logits_values.size(); ++index) {
    logits_values[index] = 0.5f * static_cast<float>(static_cast<int>(index) - 8);
  }
  array logits(logits_values.begin(), Shape{1, 16}, float32);
  array lse = logsumexp(logits, -1, true, stream);
  REQUIRE_EQ(lse.shape(), Shape{1, 1});
  check_values(
      lse,
      host_logsumexp(logits_values, 1, 16),
      stream);

  // Large logits stay finite because the kernel subtracts the row max.
  std::vector<float> bigv = {100.0f, 101.0f, 100.0f, 99.0f,
                             -100.0f, -101.0f, -100.0f, -99.0f};
  check_values(
      logsumexp(array(bigv.begin(), Shape{2, 4}, float32), -1, false, stream),
      host_logsumexp(bigv, 2, 4),
      stream);

  // One flat row longer than one workgroup.
  std::vector<float> wide_values(300);
  for (size_t index = 0; index < wide_values.size(); ++index) {
    wide_values[index] =
        static_cast<float>(static_cast<int>(index % 7) - 3);
  }
  check_values(
      logsumexp(array(wide_values.begin(), Shape{1, 300}, float32),
                -1,
                false,
                stream),
      host_logsumexp(wide_values, 1, 300),
      stream,
      2e-5);

  // An all -inf row keeps -inf instead of a NaN sum; a row that holds
  // one finite value next to -infs reduces to that value.
  float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> infs = {neg_inf, neg_inf, neg_inf,
                             neg_inf, 3.0f, neg_inf};
  array inf_result = logsumexp(
      array(infs.begin(), Shape{2, 3}, float32), -1, false, stream);
  inf_result.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(inf_result.size(), 2u);
  const float* inf_values = inf_result.data<float>();
  CHECK(std::isinf(inf_values[0]));
  CHECK(inf_values[0] < 0.0f);
  CHECK_EQ(inf_values[1], 3.0f);

  // FP16 and BF16 match the float32 host reference at their usual
  // tolerances.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    array half = astype(x, float16, stream);
    check_values(
        astype(logsumexp(half, -1, false, stream), float32, stream),
        host_logsumexp(xv, 3, 5),
        stream,
        1e-3);
  } else {
    skip("Vulkan device lacks required FP16 shader and storage features.");
  }
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array brain = astype(logits, bfloat16, stream);
    check_values(
        astype(logsumexp(brain, -1, true, stream), float32, stream),
        host_logsumexp(logits_values, 1, 16),
        stream,
        8e-3);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }
}

TEST_CASE("Log matches host references through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv;
  for (int index = 1; index <= 12; ++index) {
    xv.push_back(0.25f * static_cast<float>(index));
  }
  std::vector<float> expected;
  for (float value : xv) {
    expected.push_back(std::log(value));
  }
  array x(xv.begin(), Shape{3, 4}, float32);
  check_values(log(x, stream), expected, stream, 1e-6);

  // Sampling code negates log probabilities before the cumulative sum.
  check_values(
      negative(log(x, stream), stream),
      [&]() {
        std::vector<float> values;
        for (float value : expected) {
          values.push_back(-value);
        }
        return values;
      }(),
      stream,
      1e-6);

  // BF16 log feeds the mlx-lm sampling path for brain-float models.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    check_values(
        astype(log(astype(x, bfloat16, stream), stream), float32, stream),
        expected,
        stream,
        8e-3);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }
}



TEST_CASE("take gathers table rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> tv = {
      10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f,
      30.0f, 31.0f, 32.0f, 33.0f};
  array table(tv.begin(), Shape{3, 4}, float32);

  // Exact lookups with a repeated index.
  array indices({2, 0, 2}, int32);
  check_values(
      take(table, indices, 0, stream),
      {30.0f,
       31.0f,
       32.0f,
       33.0f,
       10.0f,
       11.0f,
       12.0f,
       13.0f,
       30.0f,
       31.0f,
       32.0f,
       33.0f},
      stream);

  // Negative indices wrap like upstream offset_neg_idx (-1 reads the
  // last row); a genuinely out-of-range index such as 7 still writes
  // the documented zero row because upstream leaves such reads
  // undefined.
  array bounds({1, 7, -1, 0}, int32);
  check_values(
      take(table, bounds, 0, stream),
      {20.0f,
       21.0f,
       22.0f,
       23.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       30.0f,
       31.0f,
       32.0f,
       33.0f,
       10.0f,
       11.0f,
       12.0f,
       13.0f},
      stream);

  // The narrow index dtypes decode through the word transport: the
  // same {2, 0, 2} row lookup computes for every one of them.
  for (Dtype dtype : {int8, uint8, int16, uint16}) {
    check_values(
        take(table, array({2, 0, 2}, dtype), 0, stream),
        {30.0f,
         31.0f,
         32.0f,
         33.0f,
         10.0f,
         11.0f,
         12.0f,
         13.0f,
         30.0f,
         31.0f,
         32.0f,
         33.0f},
        stream);
  }
  // Signed narrow indices wrap like the 32-bit mode: -1 reads the last
  // row.
  check_values(
      take(table, array({-1, 0}, int16), 0, stream),
      {30.0f,
       31.0f,
       32.0f,
       33.0f,
       10.0f,
       11.0f,
       12.0f,
       13.0f},
      stream);
  // Boolean and float indices are rejected one layer up by the shared
  // gather op itself, at graph build time.
  std::string float_error;
  try {
    take(table, array({0}, float32), 0, stream);
  } catch (const std::exception& error) {
    float_error = error.what();
  }
  CHECK(float_error.find("Indices must be integral") != std::string::npos);
  std::string bool_error;
  try {
    take(table, array({0}, bool_), 0, stream);
  } catch (const std::exception& error) {
    bool_error = error.what();
  }
  CHECK(bool_error.find("Boolean indices") != std::string::npos);
  // Non-zero gather axes gather along that axis through the general
  // take kernel: column 1 of each row, in row order.
  check_values(
      take(table, array({1}, int32), 1, stream),
      {11.0f, 21.0f, 31.0f},
      stream);

  // Higher-rank tables take rows through the same kernel: rows {0} of
  // the {1, 3, 4} table keep the full trailing shape.
  array cube(tv.begin(), Shape{1, 3, 4}, float32);
  check_values(
      take(cube, array({0}, int32), 0, stream),
      {10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f,
       30.0f, 31.0f, 32.0f, 33.0f},
      stream);

  const auto& capabilities = omarchy::device(0).capabilities();
  std::vector<float> hv = {0.5f, 1.0f, 2.0f, -1.5f, 0.25f, 4.0f};
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    array half_table = astype(
        array(hv.begin(), Shape{2, 3}, float32), float16, stream);
    check_values(
        astype(
            take(half_table, array({1, 0, 1}, int32), 0, stream),
            float32,
            stream),
        {-1.5f, 0.25f, 4.0f, 0.5f, 1.0f, 2.0f, -1.5f, 0.25f, 4.0f},
        stream,
        1e-3);
  }
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array brain_table = astype(
        array(hv.begin(), Shape{2, 3}, float32), bfloat16, stream);
    check_values(
        astype(
            take(brain_table, array({0, 1}, int32), 0, stream),
            float32,
            stream),
        {0.5f, 1.0f, 2.0f, -1.5f, 0.25f, 4.0f},
        stream,
        8e-3);
  }
}

TEST_CASE("take gathers N-D index arrays as flat row sequences") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> tv = {
      10.0f, 11.0f, 12.0f, 13.0f, 20.0f, 21.0f, 22.0f, 23.0f,
      30.0f, 31.0f, 32.0f, 33.0f, 40.0f, 41.0f, 42.0f, 43.0f,
      50.0f, 51.0f, 52.0f, 53.0f};
  array table(tv.begin(), Shape{5, 4}, float32);

  // A 2-D index array gathers rows in its flat row-major order, and the
  // output shape is indices.shape + [cols].
  std::vector<int> iv = {4, 0, 3, 1, 2, 0};
  array indices(iv.begin(), Shape{2, 3}, int32);
  array gathered = take(table, indices, 0, stream);
  CHECK_EQ(gathered.shape().size(), 3);
  CHECK_EQ(gathered.shape(0), 2);
  CHECK_EQ(gathered.shape(1), 3);
  CHECK_EQ(gathered.shape(2), 4);
  check_values(
      gathered,
      {50.0f, 51.0f, 52.0f, 53.0f,
       10.0f, 11.0f, 12.0f, 13.0f,
       40.0f, 41.0f, 42.0f, 43.0f,
       20.0f, 21.0f, 22.0f, 23.0f,
       30.0f, 31.0f, 32.0f, 33.0f,
       10.0f, 11.0f, 12.0f, 13.0f},
      stream);

  // The decode-time index shape [1, 1] keeps one row.
  std::vector<int> dv = {2};
  array decode(dv.begin(), Shape{1, 1}, int32);
  array decoded = take(table, decode, 0, stream);
  CHECK_EQ(decoded.shape().size(), 3);
  CHECK_EQ(decoded.shape(0), 1);
  CHECK_EQ(decoded.shape(1), 1);
  CHECK_EQ(decoded.shape(2), 4);
  check_values(decoded, {30.0f, 31.0f, 32.0f, 33.0f}, stream);

  // A bf16 table through the mlx-lm embedding shape family with 2-D
  // indices.
  const auto& capabilities = omarchy::device(0).capabilities();
  std::vector<float> hv = {0.5f, 1.0f, 2.0f, -1.5f, 0.25f, 4.0f};
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array brain_table = astype(
        array(hv.begin(), Shape{2, 3}, float32), bfloat16, stream);
    std::vector<int> biv = {1, 0, 0, 1};
    array brain_indices(biv.begin(), Shape{2, 2}, int32);
    check_values(
        astype(
            take(brain_table, brain_indices, 0, stream),
            float32,
            stream),
        {-1.5f,
         0.25f,
         4.0f,
         0.5f,
         1.0f,
         2.0f,
         0.5f,
         1.0f,
         2.0f,
         -1.5f,
         0.25f,
         4.0f},
        stream,
        8e-3);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }

  // A transposed index view is gapless but not row-contiguous; the
  // consumer-boundary normalization (3b30130) gathers it correctly, so
  // pin the flat row sequence instead of a named error. The transposed
  // indices {{1, 4}, {2, 0}, {3, 2}} read rows 1, 4, 2, 0, 3, 2.
  std::vector<int> wv = {1, 2, 3, 4, 0, 2};
  array wide(wv.begin(), Shape{2, 3}, int32);
  check_values(
      take(table, transpose(wide, {1, 0}, stream), 0, stream),
      {20.0f, 21.0f, 22.0f, 23.0f,
       50.0f, 51.0f, 52.0f, 53.0f,
       30.0f, 31.0f, 32.0f, 33.0f,
       10.0f, 11.0f, 12.0f, 13.0f,
       40.0f, 41.0f, 42.0f, 43.0f,
       30.0f, 31.0f, 32.0f, 33.0f},
      stream);
}


TEST_CASE("take gathers uint32 argmax indices through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> tv = {
      10.0f, 11.0f, 12.0f,
      20.0f, 21.0f, 22.0f,
      30.0f, 31.0f, 32.0f,
      40.0f, 41.0f, 42.0f};
  array table(tv.begin(), Shape{4, 3}, float32);

  // The mlx-lm greedy decode shape: argmax over [1, V] logits feeds a
  // [1, 1] index array into the embedding take.
  std::vector<float> dv = {0.1f, 3.0f, 0.2f, 0.3f};
  array decode_logits(dv.begin(), Shape{1, 4}, float32);
  array decode_ids = expand_dims(
      argmax(decode_logits, -1, false, stream), 1, stream);
  CHECK_EQ(decode_ids.dtype(), uint32);
  check_values(
      take(table, decode_ids, 0, stream),
      {20.0f, 21.0f, 22.0f},
      stream);

  // A [2, 3, V] batch reduces to [2, 3] uint32 indices, one gather per
  // batch element.
  std::vector<float> lv = {
      0.1f, 0.2f, 0.3f, 2.0f,
      1.5f, 0.2f, 0.1f, 0.4f,
      0.3f, 0.1f, 1.7f, 0.2f,
      0.1f, 2.2f, 0.3f, 0.1f,
      0.0f, 1.9f, 0.5f, 0.2f,
      0.4f, 0.2f, 0.1f, 1.3f};
  array logits(lv.begin(), Shape{2, 3, 4}, float32);
  array ids = argmax(logits, -1, false, stream);
  CHECK_EQ(ids.dtype(), uint32);
  check_uint32_values(ids, {3, 0, 2, 1, 1, 3}, stream);
  check_values(
      take(table, ids, 0, stream),
      {40.0f, 41.0f, 42.0f,
       10.0f, 11.0f, 12.0f,
       30.0f, 31.0f, 32.0f,
       20.0f, 21.0f, 22.0f,
       20.0f, 21.0f, 22.0f,
       40.0f, 41.0f, 42.0f},
      stream);

  // Plain uint32 indices gather directly, and one above the row count
  // writes the zero row.
  std::vector<uint32_t> uv = {0, 2, 2, 9};
  array raw(uv.begin(), Shape{4}, uint32);
  check_values(
      take(table, raw, 0, stream),
      {10.0f, 11.0f, 12.0f,
       30.0f, 31.0f, 32.0f,
       30.0f, 31.0f, 32.0f,
       0.0f, 0.0f, 0.0f},
      stream);
}

TEST_CASE("take gathers int64 indices and zeroes wide values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> tv = {10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
  array table(tv.begin(), Shape{3, 2}, float32);

  // Two little-endian words per index. A negative value has high word
  // 0xFFFFFFFF and wraps like upstream offset_neg_idx (-1 reads the
  // last row). A value above 2^32 has a high word that is neither 0
  // nor 0xFFFFFFFF, stays out of range, and writes the zero row.
  std::vector<int64_t> iv = {2, 0, 5000000000LL, -1};
  array indices(iv.begin(), Shape{4}, int64);
  check_values(
      take(table, indices, 0, stream),
      {30.0f, 31.0f,
       10.0f, 11.0f,
       0.0f, 0.0f,
       30.0f, 31.0f},
      stream);

  // A [2, 2] batch of int64 indices keeps its flat row-major order.
  std::vector<int64_t> bv = {1, 2, 0, 1};
  array batch(bv.begin(), Shape{2, 2}, int64);
  array gathered = take(table, batch, 0, stream);
  CHECK_EQ(gathered.shape().size(), 3);
  CHECK_EQ(gathered.shape(0), 2);
  CHECK_EQ(gathered.shape(1), 2);
  check_values(
      gathered,
      {20.0f, 21.0f,
       30.0f, 31.0f,
       10.0f, 11.0f,
       20.0f, 21.0f},
      stream);
}

TEST_CASE("take gathers uint32 and int32 tables as raw words") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // QuantizedEmbedding gathers rows of the packed uint32 weight matrix.
  // Packed words carry values above 2^31 that a float detour would
  // corrupt, so the copy must be bitwise.
  std::vector<uint32_t> tv = {
      0x00000001u, 0x00000002u, 0x00000003u, 0x00000004u,
      0x7FFFFFFFu, 0x80000000u, 0x80000001u, 0xDEADBEEFu,
      0xFFFFFFFFu, 0xFFFFFFFEu, 0x12345678u, 0x9ABCDEF0u,
      0x00000000u, 0x00000001u, 0x80000000u, 0x7FFFFFFFu,
      0xCAFEBABEu, 0xFEEDFACEu, 0x0000FFFFu, 0xFFFF0000u};
  array table(tv.begin(), Shape{5, 4}, uint32);
  std::vector<uint32_t> uv = {2, 0, 4, 2};
  array indices(uv.begin(), Shape{4}, uint32);
  array gathered = take(table, indices, 0, stream);
  CHECK_EQ(gathered.dtype(), uint32);
  CHECK_EQ(gathered.shape(0), 4);
  CHECK_EQ(gathered.shape(1), 4);
  std::vector<uint32_t> expected;
  for (uint32_t row : uv) {
    expected.insert(
        expected.end(), tv.begin() + row * 4, tv.begin() + row * 4 + 4);
  }
  check_uint32_values(gathered, expected, stream);

  // Out-of-range indices still write the zero row on raw-word tables.
  std::vector<uint32_t> bv = {1, 5, 0};
  array bounds(bv.begin(), Shape{3}, uint32);
  check_uint32_values(
      take(table, bounds, 0, stream),
      {0x7FFFFFFFu,
       0x80000000u,
       0x80000001u,
       0xDEADBEEFu,
       0u,
       0u,
       0u,
       0u,
       0x00000001u,
       0x00000002u,
       0x00000003u,
       0x00000004u},
      stream);

  // An int32 table shares the raw-word kernel via reinterpret: the copy
  // is bitwise, so negative words round-trip unchanged.
  std::vector<int32_t> sv = {
      -1, -2147483647 - 1, 2147483647, 0, 123456789, -987654321};
  array signed_table(sv.begin(), Shape{3, 2}, int32);
  std::vector<int32_t> siv = {2, 0, 1};
  array signed_indices(siv.begin(), Shape{3}, int32);
  array signed_gathered = take(signed_table, signed_indices, 0, stream);
  CHECK_EQ(signed_gathered.dtype(), int32);
  check_int32_values(
      signed_gathered,
      {123456789,
       -987654321,
       -1,
       -2147483647 - 1,
       2147483647,
       0},
      stream);

  // The QuantizedEmbedding shape family: [1, 40, 1] indices over a
  // [rows, 112] packed table produce [1, 40, 1, 112].
  const int rows = 8;
  const int cols = 112;
  std::vector<uint32_t> wv(rows * cols);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      wv[r * cols + c] =
          static_cast<uint32_t>(r * cols + c) * 2654435761u + 0x80000000u;
    }
  }
  array wide_table(wv.begin(), Shape{rows, cols}, uint32);
  std::vector<int32_t> qv(40);
  for (int i = 0; i < 40; ++i) {
    qv[i] = (i * 3 + 1) % rows;
  }
  array q_indices(qv.begin(), Shape{1, 40, 1}, int32);
  array q_gathered = take(wide_table, q_indices, 0, stream);
  CHECK_EQ(q_gathered.shape().size(), 4);
  CHECK_EQ(q_gathered.shape(0), 1);
  CHECK_EQ(q_gathered.shape(1), 40);
  CHECK_EQ(q_gathered.shape(2), 1);
  CHECK_EQ(q_gathered.shape(3), 112);
  std::vector<uint32_t> wide_expected;
  wide_expected.reserve(40 * cols);
  for (int row : qv) {
    wide_expected.insert(
        wide_expected.end(),
        wv.begin() + row * cols,
        wv.begin() + row * cols + cols);
  }
  check_uint32_values(q_gathered, wide_expected, stream);

  // The uint32 and int64 index modes stay available over a raw-word
  // table.
  std::vector<uint32_t> uiv = {3, 0};
  array u_indices(uiv.begin(), Shape{2}, uint32);
  std::vector<uint32_t> mode_expected;
  for (uint32_t row : uiv) {
    mode_expected.insert(
        mode_expected.end(),
        wv.begin() + row * cols,
        wv.begin() + row * cols + cols);
  }
  check_uint32_values(
      take(wide_table, u_indices, 0, stream), mode_expected, stream);
  std::vector<int64_t> liv = {1, 0, 0x100000000LL};
  array l_indices(liv.begin(), Shape{3}, int64);
  std::vector<uint32_t> l_expected(
      wv.begin() + cols, wv.begin() + 2 * cols);
  l_expected.insert(l_expected.end(), wv.begin(), wv.begin() + cols);
  l_expected.insert(l_expected.end(), cols, 0u);
  check_uint32_values(
      take(wide_table, l_indices, 0, stream), l_expected, stream);

  // A uint16 table rides the raw halfword copy: rows land bit-exact.
  std::vector<uint16_t> hv = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                              11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  array u16_table(hv.begin(), Shape{5, 4}, uint16);
  std::vector<uint16_t> u16_expected;
  for (uint32_t row : uv) {
    u16_expected.insert(
        u16_expected.end(),
        hv.begin() + row * 4,
        hv.begin() + row * 4 + 4);
  }
  check_uint16_values(
      take(u16_table, indices, 0, stream), u16_expected, stream);

  array f64_table(tv.begin(), Shape{5, 4}, float64);
  std::string f64_error;
  try {
    take(f64_table, indices, 0, stream);
  } catch (const std::exception& error) {
    f64_error = error.what();
  }
  CHECK(
      f64_error.find("float64 is not supported on the GPU") !=
      std::string::npos);
}

TEST_CASE("general broadcast elementwise matches host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // (rows,1) * (rows,cols): the broadcast axis is the last lhs axis, so
  // this pins the shape-aware path.
  std::vector<float> rv = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> mv = {2.0f, 1.0f, 0.5f, 3.0f, -1.0f, 2.0f,
                           0.5f, 4.0f, -2.0f, 1.5f, 0.25f, -0.5f};
  array rows(rv.begin(), Shape{4, 1}, float32);
  array matrix(mv.begin(), Shape{4, 3}, float32);
  std::vector<float> expected(12);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      expected[r * 3 + c] = rv[r] * mv[r * 3 + c];
    }
  }
  check_values(multiply(rows, matrix, stream), expected, stream);

  // (1,cols) * (rows,cols): leading-axis broadcast keeps the trailing
  // modulo fast path.
  std::vector<float> cv = {0.5f, -1.0f, 2.0f};
  array cols(cv.begin(), Shape{1, 3}, float32);
  std::vector<float> expected_leading(12);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      expected_leading[r * 3 + c] = cv[c] * mv[r * 3 + c];
    }
  }
  check_values(multiply(cols, matrix, stream), expected_leading, stream);

  // Scalar broadcast view.
  check_values(
      multiply(matrix, array(2.0f), stream),
      {4.0f, 2.0f, 1.0f, 6.0f, -2.0f, 4.0f,
       1.0f, 8.0f, -4.0f, 3.0f, 0.5f, -1.0f},
      stream);

  // 3D middle-axis broadcast: (2,1,3) * (2,2,3).
  std::vector<float> tv = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  array middle(tv.begin(), Shape{2, 1, 3}, float32);
  std::vector<float> bv = {1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                           3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f};
  array cube(bv.begin(), Shape{2, 2, 3}, float32);
  std::vector<float> expected_middle(12);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 3; ++k) {
        expected_middle[(i * 2 + j) * 3 + k] =
            tv[i * 3 + k] * bv[(i * 2 + j) * 3 + k];
      }
    }
  }
  check_values(multiply(middle, cube, stream), expected_middle, stream);

  // One bf16 broadcast case through the same shape-aware path.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array brain_rows = astype(rows, bfloat16, stream);
    array brain_matrix = astype(matrix, bfloat16, stream);
    check_values(
        astype(
            multiply(brain_rows, brain_matrix, stream), float32, stream),
        expected,
        stream,
        8e-3);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }
}

TEST_CASE("four-wide 16-bit binary path matches the general kernel bit for bit") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.storage_buffer_16bit_access ||
      !capabilities.shader_float16 || !capabilities.shader_int16) {
    skip("Vulkan device lacks required 16-bit storage and shader features.");
    return;
  }
  // Values chosen so the f32 results land between storage
  // representables: the host reference rounds once (float op, then one
  // storage round), the contract both kernels carry, and the check is
  // on the stored bits. count 288 with period 96 hits the four-wide
  // path; count 231 and the odd-offset slice keep the general kernel.
  auto fill = [](size_t n, float seed) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
      v[i] = std::sin(0.37f * static_cast<float>(i) + seed) * 3.0f +
          static_cast<float>(i % 7) * 0.001f + 0.5f;
    }
    return v;
  };
  auto stored_bits = [&](const std::vector<float>& values, Dtype dtype) {
    std::vector<uint16_t> bits(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      if (dtype == float16) {
        float16_t h(values[i]);
        std::memcpy(&bits[i], &h, 2);
      } else {
        bfloat16_t h(values[i]);
        std::memcpy(&bits[i], &h, 2);
      }
    }
    return bits;
  };
  auto gpu_bits = [&](array got) {
    eval(got);
    return std::vector<uint16_t>(
        got.data<uint16_t>(), got.data<uint16_t>() + got.size());
  };
  auto mismatches = [](const std::vector<uint16_t>& e,
                       const std::vector<uint16_t>& g) {
    size_t n = 0;
    for (size_t i = 0; i < e.size(); ++i) {
      n += e[i] != g[i];
    }
    return n;
  };
  for (Dtype dtype : {float16, bfloat16}) {
    for (const Shape& shape : {Shape{3, 8, 12}, Shape{3, 7, 11}}) {
      size_t count = shape[0] * shape[1] * shape[2];
      size_t period = shape[1] * shape[2];
      array a = astype(
          array(fill(count, 0.1f).begin(), shape, float32), dtype, stream);
      array b = astype(
          array(fill(period, 1.9f).begin(), Shape{shape[1], shape[2]},
                float32),
          dtype,
          stream);
      array a32 = astype(a, float32, stream);
      array b32 = astype(b, float32, stream);
      eval(a32, b32);
      const float* ap = a32.data<float>();
      const float* bp = b32.data<float>();
      for (int op = 0; op < 4; ++op) {
        std::vector<float> ref(count);
        for (size_t i = 0; i < count; ++i) {
          float x = ap[i];
          float y = bp[i % period];
          ref[i] = op == 0 ? x + y : op == 1 ? x * y : op == 2 ? x / y : x - y;
        }
        array got = op == 0 ? add(a, b, stream)
            : op == 1     ? multiply(a, b, stream)
            : op == 2     ? divide(a, b, stream)
                          : subtract(a, b, stream);
        INFO("dtype " << dtype << " op " << op << " count " << count);
        CHECK_EQ(mismatches(stored_bits(ref, dtype), gpu_bits(got)), 0u);
      }
      array flat = reshape(a, Shape{static_cast<int>(count)}, stream);
      array bflat = reshape(b, Shape{static_cast<int>(period)}, stream);
      for (int start : {12, 13}) {
        array sl = slice(flat, {start}, {start + 64}, stream);
        array sb = slice(bflat, {0}, {64}, stream);
        std::vector<float> ref(64);
        for (size_t i = 0; i < 64; ++i) {
          ref[i] = ap[start + i] * bp[i];
        }
        INFO("dtype " << dtype << " slice start " << start);
        CHECK_EQ(
            mismatches(stored_bits(ref, dtype), gpu_bits(multiply(sl, sb, stream))),
            0u);
      }
    }
  }
}

TEST_CASE("broadcast divide carries collapsed ranks 5 through 8") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Rank-6 broadcast divide: the stride-0 axes break every mergeable
  // run, so the transport collapses to rank 6 and needs the axis-
  // metadata storage buffer (four inline push-constant slots hold at
  // most rank 4). Host reference walks the same stride pattern.
  std::vector<float> six_base_values(8);
  std::iota(six_base_values.begin(), six_base_values.end(), 1.0f);
  array six_base(six_base_values.begin(), Shape{2, 1, 2, 1, 2, 1}, float32);
  array six_view = broadcast_to(six_base, Shape{2, 2, 2, 2, 2, 2}, stream);
  std::vector<float> six_rhs_values(64);
  for (int i = 0; i < 64; ++i) {
    six_rhs_values[i] = 1.0f + static_cast<float>(i % 4);
  }
  array six_rhs(six_rhs_values.begin(), Shape{2, 2, 2, 2, 2, 2}, float32);
  std::vector<float> six_expected;
  six_expected.reserve(64);
  for (int i0 = 0; i0 < 2; ++i0) {
    for (int i1 = 0; i1 < 2; ++i1) {
      for (int i2 = 0; i2 < 2; ++i2) {
        for (int i3 = 0; i3 < 2; ++i3) {
          for (int i4 = 0; i4 < 2; ++i4) {
            for (int i5 = 0; i5 < 2; ++i5) {
              int flat =
                  ((((i0 * 2 + i1) * 2 + i2) * 2 + i3) * 2 + i4) * 2 + i5;
              six_expected.push_back(
                  six_base_values[i0 * 4 + i2 * 2 + i4] / six_rhs_values[flat]);
            }
          }
        }
      }
    }
  }
  check_values(divide(six_view, six_rhs, stream), six_expected, stream, 1e-6);
  // The reversed operand order routes the broadcast view through the
  // rhs slot instead; decode the flat index back to the view's source
  // element (even axes carry the base strides 4, 2, 1).
  std::vector<float> six_reversed_expected(64);
  for (int flat = 0; flat < 64; ++flat) {
    int source = 0;
    int rem = flat;
    int stride = 32;
    for (int axis = 0; axis < 6; ++axis) {
      int coord = rem / stride;
      rem %= stride;
      if (axis % 2 == 0) {
        source += coord * (4 >> (axis / 2));
      }
      stride /= 2;
    }
    six_reversed_expected[flat] =
        six_rhs_values[flat] / six_base_values[source];
  }
  check_values(
      divide(six_rhs, six_view, stream),
      six_reversed_expected,
      stream,
      1e-6);
  // Rank-8: one more stride-0 break per axis pair, the transport's top
  // supported collapsed rank.
  std::vector<float> eight_base_values(16);
  std::iota(eight_base_values.begin(), eight_base_values.end(), 1.0f);
  array eight_base(
      eight_base_values.begin(), Shape{2, 1, 2, 1, 2, 1, 2, 1}, float32);
  array eight_view = broadcast_to(
      eight_base, Shape{2, 2, 2, 2, 2, 2, 2, 2}, stream);
  std::vector<float> eight_rhs_values(256);
  for (int i = 0; i < 256; ++i) {
    eight_rhs_values[i] = 1.0f + static_cast<float>(i % 4);
  }
  array eight_rhs(
      eight_rhs_values.begin(), Shape{2, 2, 2, 2, 2, 2, 2, 2}, float32);
  std::vector<float> eight_expected;
  eight_expected.reserve(256);
  for (int flat = 0; flat < 256; ++flat) {
    int source = 0;
    int rem = flat;
    int stride = 128;
    for (int axis = 0; axis < 8; ++axis) {
      int coord = rem / stride;
      rem %= stride;
      if (axis % 2 == 0) {
        source += coord * (8 >> (axis / 2));
      }
      stride /= 2;
    }
    eight_expected.push_back(
        eight_base_values[source] / eight_rhs_values[flat]);
  }
  check_values(
      divide(eight_view, eight_rhs, stream), eight_expected, stream, 1e-6);
}

TEST_CASE("value_and_grad runs softmax times input through broadcast views") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {
      0.25f, -1.0f, 2.0f, 0.5f, 0.125f, -0.75f, 1.5f, -2.0f};
  array x(xv.begin(), Shape{2, 4}, float32);

  auto fun = [&](const std::vector<array>& inputs) {
    array weights = softmax(inputs[0], std::vector<int>{-1}, false, stream);
    return sum(multiply(weights, inputs[0], stream), stream);
  };
  auto [value, grads] = value_and_grad(fun, std::vector<int>{0})({x});

  // d/dx sum_j softmax(x)_j * x_j = s_j * (1 + x_j - dot(s_row, x_row)).
  float expected_value = 0.0f;
  std::vector<float> expected_dx(8);
  for (int row = 0; row < 2; ++row) {
    const float* xr = xv.data() + row * 4;
    float maximum = xr[0];
    for (int j = 1; j < 4; ++j) {
      maximum = std::max(maximum, xr[j]);
    }
    float weights[4];
    float normalizer = 0.0f;
    for (int j = 0; j < 4; ++j) {
      weights[j] = std::exp(xr[j] - maximum);
      normalizer += weights[j];
    }
    float dot = 0.0f;
    for (int j = 0; j < 4; ++j) {
      weights[j] /= normalizer;
      dot += weights[j] * xr[j];
    }
    expected_value += dot;
    for (int j = 0; j < 4; ++j) {
      expected_dx[row * 4 + j] = weights[j] * (1.0f + xr[j] - dot);
    }
  }
  check_values(value, {expected_value}, stream, 1e-4);
  check_values(grads.at(0), expected_dx, stream, 1e-4);
}

TEST_CASE("batched Matmul matches host references across layouts") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  auto fill = [](std::vector<float>& values, int modulo, int bias) {
    for (size_t index = 0; index < values.size(); ++index) {
      values[index] =
          static_cast<float>(static_cast<int>(index % modulo) - bias);
    }
  };

  // Rank 3: (2, 3, 4) @ (2, 4, 5).
  constexpr size_t batch3 = 2;
  constexpr size_t m3 = 3;
  constexpr size_t k3 = 4;
  constexpr size_t n3 = 5;
  std::vector<float> a3_values(batch3 * m3 * k3);
  std::vector<float> b3_values(batch3 * k3 * n3);
  fill(a3_values, 7, 3);
  fill(b3_values, 5, 2);
  std::vector<float> expected3(batch3 * m3 * n3, 0.0f);
  for (size_t batch = 0; batch < batch3; ++batch) {
    for (size_t row = 0; row < m3; ++row) {
      for (size_t column = 0; column < n3; ++column) {
        for (size_t inner = 0; inner < k3; ++inner) {
          expected3[batch * m3 * n3 + row * n3 + column] +=
              a3_values[batch * m3 * k3 + row * k3 + inner] *
              b3_values[batch * k3 * n3 + inner * n3 + column];
        }
      }
    }
  }
  array a3(a3_values.begin(), Shape{2, 3, 4}, float32);
  array b3(b3_values.begin(), Shape{2, 4, 5}, float32);
  check_values(matmul(a3, b3, stream), expected3, stream);
  // Per-matrix transposed views of both operands. The stores hold the
  // operands in transposed layout and the transpose views restore the
  // matmul operand shapes.
  std::vector<float> b3_store_values(batch3 * k3 * n3);
  for (size_t batch = 0; batch < batch3; ++batch) {
    for (size_t row = 0; row < k3; ++row) {
      for (size_t column = 0; column < n3; ++column) {
        b3_store_values[batch * n3 * k3 + column * k3 + row] =
            b3_values[batch * k3 * n3 + row * n3 + column];
      }
    }
  }
  array b3_store(b3_store_values.begin(), Shape{2, 5, 4}, float32);
  check_values(
      matmul(a3, transpose(b3_store, {0, 2, 1}, stream), stream),
      expected3,
      stream);
  std::vector<float> a3_store_values(batch3 * k3 * m3);
  for (size_t batch = 0; batch < batch3; ++batch) {
    for (size_t row = 0; row < m3; ++row) {
      for (size_t inner = 0; inner < k3; ++inner) {
        a3_store_values[batch * k3 * m3 + inner * m3 + row] =
            a3_values[batch * m3 * k3 + row * k3 + inner];
      }
    }
  }
  array a3_store(a3_store_values.begin(), Shape{2, 4, 3}, float32);
  check_values(
      matmul(transpose(a3_store, {0, 2, 1}, stream), b3, stream),
      expected3,
      stream);

  // Rank 4: (2, 2, 3, 4) @ (2, 2, 4, 5), dense and transposed-view B.
  constexpr size_t batch4 = 4;
  constexpr size_t m4 = 3;
  constexpr size_t k4 = 4;
  constexpr size_t n4 = 5;
  std::vector<float> a4_values(batch4 * m4 * k4);
  std::vector<float> b4_values(batch4 * k4 * n4);
  fill(a4_values, 6, 3);
  fill(b4_values, 4, 2);
  std::vector<float> expected4(batch4 * m4 * n4, 0.0f);
  for (size_t batch = 0; batch < batch4; ++batch) {
    for (size_t row = 0; row < m4; ++row) {
      for (size_t column = 0; column < n4; ++column) {
        for (size_t inner = 0; inner < k4; ++inner) {
          expected4[batch * m4 * n4 + row * n4 + column] +=
              a4_values[batch * m4 * k4 + row * k4 + inner] *
              b4_values[batch * k4 * n4 + inner * n4 + column];
        }
      }
    }
  }
  array a4(a4_values.begin(), Shape{2, 2, 3, 4}, float32);
  array b4(b4_values.begin(), Shape{2, 2, 4, 5}, float32);
  check_values(matmul(a4, b4, stream), expected4, stream);
  std::vector<float> b4_store_values(batch4 * k4 * n4);
  for (size_t batch = 0; batch < batch4; ++batch) {
    for (size_t row = 0; row < k4; ++row) {
      for (size_t column = 0; column < n4; ++column) {
        b4_store_values[batch * n4 * k4 + column * k4 + row] =
            b4_values[batch * k4 * n4 + row * n4 + column];
      }
    }
  }
  array b4_store(b4_store_values.begin(), Shape{2, 2, 5, 4}, float32);
  check_values(
      matmul(a4, transpose(b4_store, {0, 1, 3, 2}, stream), stream),
      expected4,
      stream);

  // Batched AddMM: scalar, per-row, and full per-batch bias.
  check_values(
      addmm(array(1.0f, float32), a3, b3, 2.0f, 0.5f, stream),
      [&]() {
        std::vector<float> values;
        for (float value : expected3) {
          values.push_back(0.5f + 2.0f * value);
        }
        return values;
      }(),
      stream);
  std::vector<float> bias(n3);
  fill(bias, 3, 1);
  std::vector<float> row_bias_expected;
  for (size_t batch = 0; batch < batch3; ++batch) {
    for (size_t row = 0; row < m3; ++row) {
      for (size_t column = 0; column < n3; ++column) {
        row_bias_expected.push_back(
            2.0f * expected3[batch * m3 * n3 + row * n3 + column] +
            0.5f * bias[column]);
      }
    }
  }
  check_values(
      addmm(array(bias.begin(), Shape{5}, float32), a3, b3, 2.0f, 0.5f, stream),
      row_bias_expected,
      stream);
  std::vector<float> full_bias(batch3 * m3 * n3);
  fill(full_bias, 8, 4);
  std::vector<float> full_bias_expected;
  for (size_t index = 0; index < expected3.size(); ++index) {
    full_bias_expected.push_back(
        2.0f * expected3[index] + 0.5f * full_bias[index]);
  }
  check_values(
      addmm(
          array(full_bias.begin(), Shape{2, 3, 5}, float32),
          a3,
          b3,
          2.0f,
          0.5f,
          stream),
      full_bias_expected,
      stream);

  // Broadcast batch axes run through the stride-0 views: every batch
  // step multiplies the same stored matrix, verified against the host
  // loop. This is the shape the composed GQA attention emits.
  array a3_single(a3_values.begin(), Shape{1, 3, 4}, float32);
  array broadcast_a = broadcast_to(a3_single, {3, 3, 4}, stream);
  std::vector<float> b3_wide_values(3 * 4 * 5);
  fill(b3_wide_values, 5, 2);
  array b3_wide(b3_wide_values.begin(), Shape{3, 4, 5}, float32);
  std::vector<float> broadcast_expected;
  for (size_t batch = 0; batch < 3; ++batch) {
    for (size_t row = 0; row < m3; ++row) {
      for (size_t column = 0; column < n3; ++column) {
        float dot = 0.0f;
        for (size_t inner = 0; inner < k3; ++inner) {
          dot += a3_values[row * k3 + inner] *
              b3_wide_values[inner * n3 + column];
        }
        broadcast_expected.push_back(dot);
      }
    }
  }
  check_values(matmul(broadcast_a, b3_wide, stream), broadcast_expected, stream);

  // Rank 5 passes with dense, stride-0 broadcast, and per-matrix
  // transposed batch operands. Every host reference is computed from
  // the same vectors that back the arrays.
  std::vector<float> a5_values(2 * 2 * 2 * 3 * 4);
  std::vector<float> b5_values(2 * 2 * 2 * 4 * 5);
  fill(a5_values, 6, 3);
  fill(b5_values, 4, 2);
  auto host5 = [&](size_t b1, size_t b2, size_t b2_source) {
    std::vector<float> out(3 * 5);
    for (size_t row = 0; row < 3; ++row) {
      for (size_t column = 0; column < 5; ++column) {
        float dot = 0.0f;
        for (size_t inner = 0; inner < 4; ++inner) {
          dot += a5_values[(b1 * 2 + b2) * 12 + row * 4 + inner] *
              b5_values[(b1 * 2 + b2_source) * 20 + inner * 5 + column];
        }
        out[row * 5 + column] = dot;
      }
    }
    return out;
  };
  array a5(a5_values.begin(), Shape{2, 2, 2, 3, 4}, float32);
  array b5(b5_values.begin(), Shape{2, 2, 2, 4, 5}, float32);
  for (size_t b1 = 0; b1 < 2; ++b1) {
    for (size_t b2 = 0; b2 < 2; ++b2) {
      check_values(matmul(a5, b5, stream), b1, b2, host5(b1, b2, b2), stream);
    }
  }
  // A stride-0 broadcast batch axis repeats one stored matrix per
  // batch step: batches (b1, 0) and (b1, 1) share the (b1, 0) source.
  array b5_single(b5_values.begin(), Shape{2, 2, 1, 4, 5}, float32);
  array b5_broadcast =
      broadcast_to(b5_single, {2, 2, 2, 4, 5}, stream);
  for (size_t b1 = 0; b1 < 2; ++b1) {
    for (size_t b2 = 0; b2 < 2; ++b2) {
      check_values(
          matmul(a5, b5_broadcast, stream), b1, b2, host5(b1, b2, 0), stream);
    }
  }
  // A per-matrix transposed stack works at rank 5 as well.
  std::vector<float> b5_store_values(2 * 2 * 2 * 5 * 4);
  for (size_t b1 = 0; b1 < 2; ++b1) {
    for (size_t b2 = 0; b2 < 2; ++b2) {
      for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 5; ++column) {
          b5_store_values[(b1 * 2 + b2) * 20 + column * 4 + row] =
              b5_values[(b1 * 2 + b2) * 20 + row * 5 + column];
        }
      }
    }
  }
  array b5_store(b5_store_values.begin(), Shape{2, 2, 2, 5, 4}, float32);
  for (size_t b1 = 0; b1 < 2; ++b1) {
    for (size_t b2 = 0; b2 < 2; ++b2) {
      check_values(
          matmul(a5, transpose(b5_store, {0, 1, 2, 4, 3}, stream), stream),
          b1,
          b2,
          host5(b1, b2, b2),
          stream);
    }
  }
}

TEST_CASE("scaled_dot_product_attention matches a batched matmul reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr size_t heads = 2;
  constexpr size_t queries_length = 4;
  constexpr size_t keys_length = 8;
  constexpr size_t head_dim = 8;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  std::vector<float> q_values(heads * queries_length * head_dim);
  std::vector<float> k_values(heads * keys_length * head_dim);
  std::vector<float> v_values(heads * keys_length * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index);
  }
  for (size_t index = 0; index < k_values.size(); ++index) {
    k_values[index] = pattern(index + 3);
  }
  for (size_t index = 0; index < v_values.size(); ++index) {
    v_values[index] = pattern(index + 7);
  }
  array q(q_values.begin(), Shape{1, 2, 4, 8}, float32);
  array k(k_values.begin(), Shape{1, 2, 8, 8}, float32);
  array v(v_values.begin(), Shape{1, 2, 8, 8}, float32);
  constexpr float scale = 0.25f;

  array attention = fast::scaled_dot_product_attention(
      q, k, v, scale, "", std::nullopt, std::nullopt, false, stream);
  std::string blocked = evaluation_error(attention);
  REQUIRE(blocked.empty());

  std::vector<float> expected(heads * queries_length * head_dim, 0.0f);
  std::vector<float> scores(keys_length);
  for (size_t head = 0; head < heads; ++head) {
    for (size_t row = 0; row < queries_length; ++row) {
      float max_score = -std::numeric_limits<float>::infinity();
      for (size_t column = 0; column < keys_length; ++column) {
        float dot = 0.0f;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          dot += q_values[head * queries_length * head_dim + row * head_dim +
                          inner] *
              k_values[head * keys_length * head_dim + column * head_dim +
                       inner];
        }
        scores[column] = scale * dot;
        max_score = std::max(max_score, scores[column]);
      }
      float normalizer = 0.0f;
      for (size_t column = 0; column < keys_length; ++column) {
        scores[column] = std::exp(scores[column] - max_score);
        normalizer += scores[column];
      }
      for (size_t inner = 0; inner < head_dim; ++inner) {
        float sum = 0.0f;
        for (size_t column = 0; column < keys_length; ++column) {
          sum += scores[column] / normalizer *
              v_values[head * keys_length * head_dim + column * head_dim +
                       inner];
        }
        expected[head * queries_length * head_dim + row * head_dim + inner] =
            sum;
      }
    }
  }
  check_values(attention, expected, stream, 1e-3);
}

TEST_CASE(
    "scaled_dot_product_attention expands kv heads through a rank-5 score matmul") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr size_t q_heads = 4;
  constexpr size_t kv_heads = 2;
  constexpr size_t seq_length = 8;
  constexpr size_t head_dim = 8;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  std::vector<float> q_values(q_heads * seq_length * head_dim);
  std::vector<float> kv_values(kv_heads * seq_length * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index);
  }
  for (size_t index = 0; index < kv_values.size(); ++index) {
    kv_values[index] = pattern(index + 5);
  }
  array q(q_values.begin(), Shape{1, 4, 8, 8}, float32);
  array k(kv_values.begin(), Shape{1, 2, 8, 8}, float32);
  array v(kv_values.begin(), Shape{1, 2, 8, 8}, float32);
  constexpr float scale = 0.25f;

  // The primitive unflattens q to [B, kv_heads, n_rep, L, D] and
  // broadcasts the kv view, so the scores matmul runs at rank 5 with a
  // stride-0 batch axis.
  array attention = fast::scaled_dot_product_attention(
      q, k, v, scale, "", std::nullopt, std::nullopt, false, stream);
  std::string blocked = evaluation_error(attention);
  REQUIRE(blocked.empty());

  auto host_reference = [&](const std::vector<float>& qv) {
    std::vector<float> out(q_heads * seq_length * head_dim, 0.0f);
    for (size_t head = 0; head < q_heads; ++head) {
      size_t kv_head = head / (q_heads / kv_heads);
      for (size_t row = 0; row < seq_length; ++row) {
        float max_score = -std::numeric_limits<float>::infinity();
        std::vector<float> scores(seq_length);
        for (size_t column = 0; column < seq_length; ++column) {
          float dot_qk = 0.0f;
          for (size_t inner = 0; inner < head_dim; ++inner) {
            dot_qk += qv[head * seq_length * head_dim + row * head_dim +
                         inner] *
                kv_values[kv_head * seq_length * head_dim + column *
                              head_dim + inner];
          }
          scores[column] = scale * dot_qk;
          max_score = std::max(max_score, scores[column]);
        }
        float normalizer = 0.0f;
        for (size_t column = 0; column < seq_length; ++column) {
          scores[column] = std::exp(scores[column] - max_score);
          normalizer += scores[column];
        }
        for (size_t inner = 0; inner < head_dim; ++inner) {
          float sum = 0.0f;
          for (size_t column = 0; column < seq_length; ++column) {
            sum += scores[column] / normalizer *
                kv_values[kv_head * seq_length * head_dim + column *
                              head_dim + inner];
          }
          out[head * seq_length * head_dim + row * head_dim + inner] = sum;
        }
      }
    }
    return out;
  };
  check_values(attention, host_reference(q_values), stream, 1e-3);

  // The bf16 variant runs the same composition through the bf16
  // matmul, softmax, and select kernels.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 || !capabilities.storage_buffer_16bit_access ||
      !capabilities.shader_int16) {
    skip("Vulkan device lacks required BF16 shader and storage features.");
    return;
  }
  array q_bf16 = astype(q, bfloat16, stream);
  array k_bf16 = astype(k, bfloat16, stream);
  array v_bf16 = astype(v, bfloat16, stream);
  array attention_bf16 = fast::scaled_dot_product_attention(
      q_bf16, k_bf16, v_bf16, scale, "", std::nullopt, std::nullopt, false,
      stream);
  std::string blocked_bf16 = evaluation_error(attention_bf16);
  REQUIRE(blocked_bf16.empty());
  check_values(
      astype(attention_bf16, float32, stream),
      host_reference(q_values),
      stream,
      1e-2);
}


TEST_CASE("cold-cache GQA decode matmul runs over cache slice views") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Qwen2.5-0.5B geometry: 14 query heads = 2 kv heads x 7 repeats. A
  // fresh cache preallocates, slice_updates one decode step, and reads
  // the state prefix back as a strided view, so the scores matmul sees
  // batch strides that are uniform but not contiguous.
  constexpr size_t kv_heads = 2;
  constexpr size_t n_rep = 7;
  constexpr size_t head_dim = 8;
  constexpr float scale = 0.25f;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  auto new_token = [&](size_t token) {
    std::vector<float> values(kv_heads * head_dim);
    for (size_t index = 0; index < values.size(); ++index) {
      values[index] = pattern(index + token * 17 + 3);
    }
    array update(
        values.begin(),
        Shape{static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)},
        float32);
    return std::pair<array, std::vector<float>>{update, values};
  };

  array cache = zeros(
      {1,
       static_cast<int>(kv_heads),
       static_cast<int>(16),
       static_cast<int>(head_dim)},
      float32,
      stream);

  // First decode step: state [1, 2, 1, 8] with strides {128, 64, 8, 1}.
  // The scores matmul output is exactly the [1, 2, 7, 1, 1] shape from
  // the M1 smoke blocker.
  auto [update1, k1_values] = new_token(0);
  cache = slice_update(
      cache,
      update1,
      Shape{0, 0, 0, 0},
      Shape{1, static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)},
      stream);
  array state1 = slice(
      cache,
      {0, 0, 0, 0},
      {1, static_cast<int>(kv_heads), 1, static_cast<int>(head_dim)},
      stream);

  std::vector<float> q_values(kv_heads * n_rep * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index);
  }
  array q(q_values.begin(), Shape{1, 14, 1, 8}, float32);
  array q5 = unflatten(
      multiply(array(scale, float32), q, stream),
      1,
      Shape{static_cast<int>(kv_heads), static_cast<int>(n_rep)},
      stream);
  array scores1 = matmul(
      q5, swapaxes(expand_dims(state1, 2, stream), -1, -2, stream), stream);
  const Shape scores1_shape{1, 2, 7, 1, 1};
  REQUIRE_EQ(scores1.shape(), scores1_shape);
  std::vector<float> expected_scores1(kv_heads * n_rep, 0.0f);
  for (size_t kv = 0; kv < kv_heads; ++kv) {
    for (size_t rep = 0; rep < n_rep; ++rep) {
      float dot = 0.0f;
      for (size_t inner = 0; inner < head_dim; ++inner) {
        dot += scale * q_values[(kv * n_rep + rep) * head_dim + inner] *
            k1_values[kv * head_dim + inner];
      }
      expected_scores1[kv * n_rep + rep] = dot;
    }
  }
  check_values(scores1, expected_scores1, stream, 1e-5);

  // End to end: the composed causal attention over the same cache view
  // matches the key row (one key makes the softmax trivial). The second
  // step below checks a real two-key softmax.
  array attention1 = fast::scaled_dot_product_attention(
      q,
      state1,
      state1,
      scale,
      "causal",
      std::nullopt,
      std::nullopt,
      false,
      stream);
  std::string blocked1 = evaluation_error(attention1);
  REQUIRE(blocked1.empty());
  std::vector<float> expected_attention1(kv_heads * n_rep * head_dim, 0.0f);
  for (size_t head = 0; head < kv_heads * n_rep; ++head) {
    size_t kv_head = head / n_rep;
    for (size_t inner = 0; inner < head_dim; ++inner) {
      expected_attention1[head * head_dim + inner] =
          k1_values[kv_head * head_dim + inner];
    }
  }
  check_values(attention1, expected_attention1, stream, 1e-5);

  // Second decode step: two keys, still a strided state view, and a
  // non-trivial softmax over the two scores.
  auto [update2, k2_values] = new_token(1);
  cache = slice_update(
      cache,
      update2,
      Shape{0, 0, 1, 0},
      Shape{1, static_cast<int>(kv_heads), 2, static_cast<int>(head_dim)},
      stream);
  array state2 = slice(
      cache,
      {0, 0, 0, 0},
      {1, static_cast<int>(kv_heads), 2, static_cast<int>(head_dim)},
      stream);
  array scores2 = matmul(
      q5, swapaxes(expand_dims(state2, 2, stream), -1, -2, stream), stream);
  const Shape scores2_shape{1, 2, 7, 1, 2};
  REQUIRE_EQ(scores2.shape(), scores2_shape);

  auto host_attention = [&](const std::vector<float>& kv1,
                            const std::vector<float>& kv2) {
    std::vector<float> out(kv_heads * n_rep * head_dim, 0.0f);
    for (size_t kv = 0; kv < kv_heads; ++kv) {
      for (size_t rep = 0; rep < n_rep; ++rep) {
        const float* q_row =
            q_values.data() + (kv * n_rep + rep) * head_dim;
        float s1 = 0.0f;
        float s2 = 0.0f;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          s1 += scale * q_row[inner] * kv1[kv * head_dim + inner];
          s2 += scale * q_row[inner] * kv2[kv * head_dim + inner];
        }
        float maximum = std::max(s1, s2);
        float p1 = std::exp(s1 - maximum);
        float p2 = std::exp(s2 - maximum);
        float normalizer = p1 + p2;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          out[(kv * n_rep + rep) * head_dim + inner] =
              (p1 * kv1[kv * head_dim + inner] +
               p2 * kv2[kv * head_dim + inner]) /
              normalizer;
        }
      }
    }
    return out;
  };
  check_values(
      fast::scaled_dot_product_attention(
          q,
          state2,
          state2,
          scale,
          "causal",
          std::nullopt,
          std::nullopt,
          false,
          stream),
      host_attention(k1_values, k2_values),
      stream,
      1e-5);

  // The bf16 model path runs the same materialized scores matmul.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.storage_buffer_16bit_access &&
      capabilities.shader_int16) {
    array scores_bf16 = matmul(
        astype(q5, bfloat16, stream),
        swapaxes(
            expand_dims(astype(state1, bfloat16, stream), 2, stream),
            -1,
            -2,
            stream),
        stream);
    check_values(
        astype(scores_bf16, float32, stream),
        expected_scores1,
        stream,
        1e-2);
  } else {
    skip("Vulkan device lacks required BF16 storage and shader features.");
  }
}

TEST_CASE("causal scaled_dot_product_attention masks future keys") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr size_t q_heads = 4;
  constexpr size_t kv_heads = 2;
  constexpr size_t seq_length = 32;
  constexpr size_t head_dim = 8;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  std::vector<float> q_values(q_heads * seq_length * head_dim);
  std::vector<float> kv_values(kv_heads * seq_length * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index);
  }
  for (size_t index = 0; index < kv_values.size(); ++index) {
    kv_values[index] = pattern(index + 5);
  }
  array q(q_values.begin(), Shape{1, 4, 32, 8}, float32);
  array k(kv_values.begin(), Shape{1, 2, 32, 8}, float32);
  array v(kv_values.begin(), Shape{1, 2, 32, 8}, float32);
  constexpr float scale = 0.25f;

  // The causal mask enters the primitive as an additive float32 term:
  // 0 for attended positions and -1e30 for future keys, with no Select
  // against the dtype minimum.
  array attention = fast::scaled_dot_product_attention(
      q, k, v, scale, "causal", std::nullopt, std::nullopt, false, stream);
  std::string blocked = evaluation_error(attention);
  REQUIRE(blocked.empty());

  std::vector<float> expected(q_heads * seq_length * head_dim, 0.0f);
  for (size_t head = 0; head < q_heads; ++head) {
    size_t kv_head = head / (q_heads / kv_heads);
    for (size_t row = 0; row < seq_length; ++row) {
      float max_score = -std::numeric_limits<float>::infinity();
      std::vector<float> scores(seq_length);
      for (size_t column = 0; column < seq_length; ++column) {
        if (column > row) {
          scores[column] = -std::numeric_limits<float>::infinity();
          continue;
        }
        float dot_qk = 0.0f;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          dot_qk += q_values[head * seq_length * head_dim + row * head_dim +
                             inner] *
              kv_values[kv_head * seq_length * head_dim + column * head_dim +
                        inner];
        }
        scores[column] = scale * dot_qk;
        max_score = std::max(max_score, scores[column]);
      }
      float normalizer = 0.0f;
      for (size_t column = 0; column <= row; ++column) {
        scores[column] = std::exp(scores[column] - max_score);
        normalizer += scores[column];
      }
      for (size_t inner = 0; inner < head_dim; ++inner) {
        float sum = 0.0f;
        for (size_t column = 0; column <= row; ++column) {
          sum += scores[column] / normalizer *
              kv_values[kv_head * seq_length * head_dim + column * head_dim +
                        inner];
        }
        expected[head * seq_length * head_dim + row * head_dim + inner] = sum;
      }
    }
  }
  check_values(attention, expected, stream, 1e-3);
}

TEST_CASE("f16 causal attention equals the additive storage-floor mask bit for bit") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }
  // The causal f16 path runs the softmax in causal mode over unmasked
  // scores (and lets the register-blocked matmuls skip masked tiles);
  // the array-mask path still adds the mask to the scores. With the
  // 0 / -65504 mask the causal composition used to build, both must
  // store identical bits. q_len 200 exercises the 64-wide column-tile
  // and k-tile skips (row tile 0 needs keys 0..63 only); q_len 70 with
  // a 30-key prefix the k_len > q_len offset.
  auto pattern = [](size_t index, float seed) {
    return std::sin(0.21f * static_cast<float>(index) + seed) * 2.0f;
  };
  for (auto [q_len, k_len] : {std::pair<int, int>{200, 200},
                              std::pair<int, int>{70, 100}}) {
    constexpr int q_heads = 4;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 64;
    std::vector<float> qv(q_heads * q_len * head_dim);
    std::vector<float> kv(kv_heads * k_len * head_dim);
    std::vector<float> vv(kv_heads * k_len * head_dim);
    for (size_t i = 0; i < qv.size(); ++i) {
      qv[i] = pattern(i, 0.3f);
    }
    for (size_t i = 0; i < kv.size(); ++i) {
      kv[i] = pattern(i, 1.1f);
      vv[i] = pattern(i, 2.7f);
    }
    array q = astype(
        array(qv.begin(), Shape{1, q_heads, q_len, head_dim}, float32),
        float16,
        stream);
    array k = astype(
        array(kv.begin(), Shape{1, kv_heads, k_len, head_dim}, float32),
        float16,
        stream);
    array v = astype(
        array(vv.begin(), Shape{1, kv_heads, k_len, head_dim}, float32),
        float16,
        stream);
    std::vector<float16_t> mask_values(q_len * k_len);
    int offset = k_len - q_len;
    for (int row = 0; row < q_len; ++row) {
      for (int col = 0; col < k_len; ++col) {
        mask_values[row * k_len + col] =
            float16_t(offset + row >= col ? 0.0f : -65504.0f);
      }
    }
    array mask(mask_values.begin(), Shape{q_len, k_len}, float16);
    const float scale = 0.125f;
    array causal = fast::scaled_dot_product_attention(
        q, k, v, scale, "causal", std::nullopt, std::nullopt, false, stream);
    array masked = fast::scaled_dot_product_attention(
        q, k, v, scale, "", mask, std::nullopt, false, stream);
    eval(causal, masked);
    const uint16_t* a = causal.data<uint16_t>();
    const uint16_t* b = masked.data<uint16_t>();
    size_t mismatches = 0;
    size_t nonzero = 0;
    for (size_t i = 0; i < causal.size(); ++i) {
      mismatches += a[i] != b[i];
      nonzero += a[i] != 0;
    }
    INFO("q_len " << q_len << " k_len " << k_len);
    CHECK_EQ(mismatches, 0u);
    CHECK_GT(nonzero, causal.size() / 2);
  }
}

// Host float64 attention over f16-representable inputs. The inputs use
// 0.25-step values, so the float16/bfloat16 device tensors are exact
// and the reference sees the same numbers.
std::vector<float> host_attention_f64(
    const std::vector<float>& q_values,
    const std::vector<float>& kv_values,
    const std::vector<float>& v_values,
    size_t q_heads,
    size_t kv_heads,
    size_t q_len,
    size_t k_len,
    size_t head_dim,
    double scale,
    bool causal) {
  size_t v_dim = v_values.size() /
      (kv_heads * k_len * head_dim) * head_dim;
  std::vector<float> out(q_heads * q_len * v_dim, 0.0f);
  for (size_t head = 0; head < q_heads; ++head) {
    size_t kv_head = head / (q_heads / kv_heads);
    for (size_t row = 0; row < q_len; ++row) {
      double max_score = -std::numeric_limits<double>::infinity();
      std::vector<double> scores(k_len);
      for (size_t column = 0; column < k_len; ++column) {
        if (causal && column > k_len - q_len + row) {
          scores[column] = 0.0;
          continue;
        }
        double dot = 0.0;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          dot += (double)q_values[head * q_len * head_dim + row * head_dim +
                                  inner] *
              (double)kv_values[kv_head * k_len * head_dim + column *
                                head_dim + inner];
        }
        scores[column] = scale * dot;
        max_score = std::max(max_score, scores[column]);
      }
      double normalizer = 0.0;
      for (size_t column = 0; column < k_len; ++column) {
        if (causal && column > k_len - q_len + row) {
          continue;
        }
        scores[column] = std::exp(scores[column] - max_score);
        normalizer += scores[column];
      }
      for (size_t dim = 0; dim < v_dim; ++dim) {
        double sum = 0.0;
        for (size_t column = 0; column < k_len; ++column) {
          if (causal && column > k_len - q_len + row) {
            continue;
          }
          sum += scores[column] / normalizer *
              (double)v_values[kv_head * k_len * head_dim + column *
                               head_dim + dim];
        }
        out[head * q_len * v_dim + row * v_dim + dim] = (float)sum;
      }
    }
  }
  return out;
}
void check_attention_deviation(
    array attention,
    const std::vector<float>& expected,
    const Stream& stream,
    double relative_tolerance) {
  attention.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(attention.size(), expected.size());
  const float* values = attention.data<float>();
  double max_deviation = 0.0;
  double reference_magnitude = 0.0;
  for (size_t index = 0; index < expected.size(); ++index) {
    max_deviation =
        std::max(max_deviation, std::abs((double)values[index] - expected[index]));
    reference_magnitude =
        std::max(reference_magnitude, std::abs((double)expected[index]));
  }
  CHECK(max_deviation <= relative_tolerance * reference_magnitude);
}

TEST_CASE(
    "scaled_dot_product_attention keeps ~600-magnitude float16 scores exact") {
  if (!compute_available()) {
    return;
  }
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 || !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required Float16 shader and storage features.");
    return;
  }
  Stream stream = gpu_stream();
  // Qwen2.5-0.5B geometry and the qdiag18/qdiag19b prefill length: the
  // f16 composed fallback materialized scores of absmax ~647 in f16
  // (ulp 0.5) and flipped softmax winners; 59.8% of causal rows had a
  // top1-top2 gap below 1.0.
  constexpr size_t q_heads = 14;
  constexpr size_t kv_heads = 2;
  constexpr size_t q_len = 41;
  constexpr size_t k_len = 41;
  constexpr size_t head_dim = 64;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  std::vector<float> q_values(q_heads * q_len * head_dim);
  std::vector<float> kv_values(kv_heads * k_len * head_dim);
  std::vector<float> v_values(kv_heads * k_len * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index);
  }
  for (size_t index = 0; index < kv_values.size(); ++index) {
    kv_values[index] = pattern(index + 5);
    v_values[index] = 0.25f * static_cast<float>(static_cast<int>(index % 9) - 4);
  }
  // Pin the failure mode: the scale is tuned so the largest score
  // reaches ~600, far beyond exact float16 addition at that magnitude.
  double dot_absmax = 0.0;
  for (size_t head = 0; head < q_heads; ++head) {
    size_t kv_head = head / (q_heads / kv_heads);
    for (size_t row = 0; row < q_len; ++row) {
      for (size_t column = 0; column < k_len; ++column) {
        double dot = 0.0;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          dot += (double)q_values[head * q_len * head_dim + row * head_dim +
                                  inner] *
              (double)kv_values[kv_head * k_len * head_dim + column *
                                head_dim + inner];
        }
        dot_absmax = std::max(dot_absmax, std::abs(dot));
      }
    }
  }
  float scale = static_cast<float>(600.0 / dot_absmax);
  REQUIRE(std::abs((double)scale * dot_absmax - 600.0) < 1.0);

  array q = astype(
      array(q_values.begin(), Shape{1, 14, 41, 64}, float32), float16, stream);
  array k = astype(
      array(kv_values.begin(), Shape{1, 2, 41, 64}, float32), float16, stream);
  array v = astype(
      array(v_values.begin(), Shape{1, 2, 41, 64}, float32), float16, stream);

  array attention = fast::scaled_dot_product_attention(
      q, k, v, scale, "", std::nullopt, std::nullopt, false, stream);
  std::string blocked = evaluation_error(attention);
  REQUIRE(blocked.empty());
  auto expected = host_attention_f64(
      q_values,
      kv_values,
      v_values,
      q_heads,
      kv_heads,
      q_len,
      k_len,
      head_dim,
      (double)scale,
      false);
  // The reference mixes v rows of magnitude ~1, so outputs carry
  // magnitude ~0.3; the old f16-score path deviated by ~0.44 here
  // (the doc measured 0.4429 against 0.17-magnitude outputs).
  check_attention_deviation(
      astype(attention, float32, stream), expected, stream, 1e-2);
}

TEST_CASE(
    "causal scaled_dot_product_attention keeps large float16 scores exact across a cache offset") {
  if (!compute_available()) {
    return;
  }
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 || !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required Float16 shader and storage features.");
    return;
  }
  Stream stream = gpu_stream();
  constexpr size_t q_heads = 14;
  constexpr size_t kv_heads = 2;
  constexpr size_t q_len = 8;
  constexpr size_t k_len = 41;
  constexpr size_t head_dim = 64;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  std::vector<float> q_values(q_heads * q_len * head_dim);
  std::vector<float> kv_values(kv_heads * k_len * head_dim);
  std::vector<float> v_values(kv_heads * k_len * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index + 1);
  }
  for (size_t index = 0; index < kv_values.size(); ++index) {
    kv_values[index] = pattern(index + 7);
    v_values[index] = 0.25f * static_cast<float>(static_cast<int>(index % 9) - 4);
  }
  double dot_absmax = 0.0;
  for (size_t head = 0; head < q_heads; ++head) {
    size_t kv_head = head / (q_heads / kv_heads);
    for (size_t row = 0; row < q_len; ++row) {
      for (size_t column = 0; column <= k_len - q_len + row; ++column) {
        double dot = 0.0;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          dot += (double)q_values[head * q_len * head_dim + row * head_dim +
                                  inner] *
              (double)kv_values[kv_head * k_len * head_dim + column *
                                head_dim + inner];
        }
        dot_absmax = std::max(dot_absmax, std::abs(dot));
      }
    }
  }
  float scale = static_cast<float>(600.0 / dot_absmax);

  array q = astype(
      array(q_values.begin(), Shape{1, 14, 8, 64}, float32), float16, stream);
  array k = astype(
      array(kv_values.begin(), Shape{1, 2, 41, 64}, float32), float16, stream);
  array v = astype(
      array(v_values.begin(), Shape{1, 2, 41, 64}, float32), float16, stream);
  array attention = fast::scaled_dot_product_attention(
      q, k, v, scale, "causal", std::nullopt, std::nullopt, false, stream);
  std::string blocked = evaluation_error(attention);
  REQUIRE(blocked.empty());
  auto expected = host_attention_f64(
      q_values,
      kv_values,
      v_values,
      q_heads,
      kv_heads,
      q_len,
      k_len,
      head_dim,
      (double)scale,
      true);
  check_attention_deviation(
      astype(attention, float32, stream), expected, stream, 1e-2);
}

TEST_CASE(
    "scaled_dot_product_attention keeps ~600-magnitude bfloat16 scores exact") {
  if (!compute_available()) {
    return;
  }
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.storage_buffer_16bit_access ||
      !capabilities.shader_int16) {
    skip("Vulkan device lacks required BF16 storage and shader features.");
    return;
  }
  Stream stream = gpu_stream();
  constexpr size_t q_heads = 4;
  constexpr size_t kv_heads = 2;
  constexpr size_t q_len = 41;
  constexpr size_t k_len = 41;
  constexpr size_t head_dim = 64;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  std::vector<float> q_values(q_heads * q_len * head_dim);
  std::vector<float> kv_values(kv_heads * k_len * head_dim);
  std::vector<float> v_values(kv_heads * k_len * head_dim);
  for (size_t index = 0; index < q_values.size(); ++index) {
    q_values[index] = pattern(index + 2);
  }
  for (size_t index = 0; index < kv_values.size(); ++index) {
    kv_values[index] = pattern(index + 6);
    v_values[index] = 0.25f * static_cast<float>(static_cast<int>(index % 9) - 4);
  }
  double dot_absmax = 0.0;
  for (size_t head = 0; head < q_heads; ++head) {
    size_t kv_head = head / (q_heads / kv_heads);
    for (size_t row = 0; row < q_len; ++row) {
      for (size_t column = 0; column < k_len; ++column) {
        double dot = 0.0;
        for (size_t inner = 0; inner < head_dim; ++inner) {
          dot += (double)q_values[head * q_len * head_dim + row * head_dim +
                                  inner] *
              (double)kv_values[kv_head * k_len * head_dim + column *
                                head_dim + inner];
        }
        dot_absmax = std::max(dot_absmax, std::abs(dot));
      }
    }
  }
  float scale = static_cast<float>(600.0 / dot_absmax);

  array q = astype(
      array(q_values.begin(), Shape{1, 4, 41, 64}, float32), bfloat16, stream);
  array k = astype(
      array(kv_values.begin(), Shape{1, 2, 41, 64}, float32), bfloat16, stream);
  array v = astype(
      array(v_values.begin(), Shape{1, 2, 41, 64}, float32), bfloat16, stream);
  array attention = fast::scaled_dot_product_attention(
      q, k, v, scale, "", std::nullopt, std::nullopt, false, stream);
  std::string blocked = evaluation_error(attention);
  REQUIRE(blocked.empty());
  auto expected = host_attention_f64(
      q_values,
      kv_values,
      v_values,
      q_heads,
      kv_heads,
      q_len,
      k_len,
      head_dim,
      (double)scale,
      false);
  check_attention_deviation(
      astype(attention, float32, stream), expected, stream, 1e-2);
}

TEST_CASE("repeat materializes broadcast reshapes through the strided copy engine") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The GQA repeat shape from qdiag19: the reshape of the broadcast
  // view flat-copied past the source allocation and produced values
  // near 2e27 and NaN on the tail rows.
  constexpr size_t kv_heads = 2;
  constexpr size_t seq_length = 41;
  constexpr size_t head_dim = 64;
  constexpr size_t repeats = 7;
  auto pattern = [](size_t index) {
    return 0.25f * static_cast<float>(static_cast<int>(index % 11) - 5);
  };
  std::vector<float> kv_values(kv_heads * seq_length * head_dim);
  for (size_t index = 0; index < kv_values.size(); ++index) {
    kv_values[index] = pattern(index + 3);
  }
  array tiled = tile(array({1, 2, 3}, {3}, int32), {2, 2, 2}, stream);
  check_int32_values(
      tiled,
      {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
       1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3},
      stream);

  array source(
      kv_values.begin(),
      Shape{1, 2, 41, 64},
      float32);
  array expanded = repeat(source, repeats, 1, stream);
  REQUIRE_EQ(expanded.shape(), Shape{1, 14, 41, 64});

  std::vector<float> expected(14 * seq_length * head_dim, 0.0f);
  for (size_t head = 0; head < 14; ++head) {
    size_t kv_head = head / repeats;
    for (size_t index = 0; index < seq_length * head_dim; ++index) {
      expected[head * seq_length * head_dim + index] =
          kv_values[kv_head * seq_length * head_dim + index];
    }
  }
  check_values(expanded, expected, stream, 1e-5);

  // Integer broadcast reshapes ride the same engine as raw words: the
  // int32 copy is bitwise, so repeated rows must match the source
  // exactly, negative values included (their words sit above 2^31).
  std::vector<int32_t> int_values(kv_heads * seq_length * head_dim);
  for (size_t index = 0; index < int_values.size(); ++index) {
    int_values[index] = static_cast<int32_t>(index % 17) - 8;
  }
  array int_source(int_values.begin(), Shape{1, 2, 41, 64}, int32);
  array int_expanded = repeat(int_source, repeats, 1, stream);
  REQUIRE_EQ(int_expanded.shape(), Shape{1, 14, 41, 64});
  std::vector<int32_t> int_expected;
  int_expected.reserve(14 * seq_length * head_dim);
  for (size_t head = 0; head < 14; ++head) {
    size_t kv_head = head / repeats;
    int_expected.insert(
        int_expected.end(),
        int_values.begin() + kv_head * seq_length * head_dim,
        int_values.begin() + (kv_head + 1) * seq_length * head_dim);
  }
  check_int32_values(int_expanded, int_expected, stream);

  // The same strided reshape handles wide, narrow, and packed-bool storage.
  std::vector<int64_t> wide_values(kv_heads * seq_length * head_dim);
  for (size_t index = 0; index < wide_values.size(); ++index) {
    wide_values[index] = (int64_t{1} << 40) + static_cast<int64_t>(index);
  }
  array wide_source(wide_values.begin(), Shape{1, 2, 41, 64}, int64);
  array wide_expanded = repeat(wide_source, repeats, 1, stream);
  std::vector<int64_t> wide_expected;
  wide_expected.reserve(14 * seq_length * head_dim);
  for (size_t head = 0; head < 14; ++head) {
    size_t kv_head = head / repeats;
    wide_expected.insert(
        wide_expected.end(),
        wide_values.begin() + kv_head * seq_length * head_dim,
        wide_values.begin() + (kv_head + 1) * seq_length * head_dim);
  }
  check_int64_values(wide_expanded, wide_expected, stream);

  std::vector<uint8_t> byte_values(kv_heads * seq_length * head_dim);
  for (size_t index = 0; index < byte_values.size(); ++index) {
    byte_values[index] = static_cast<uint8_t>(index % 251);
  }
  array byte_source(byte_values.begin(), Shape{1, 2, 41, 64}, uint8);
  array byte_expanded = repeat(byte_source, repeats, 1, stream);
  byte_expanded.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const auto* byte_result = byte_expanded.data<uint8_t>();
  for (size_t head = 0; head < 14; ++head) {
    size_t kv_head = head / repeats;
    for (size_t index = 0; index < seq_length * head_dim; ++index) {
      CHECK_EQ(
          byte_result[head * seq_length * head_dim + index],
          byte_values[kv_head * seq_length * head_dim + index]);
    }
  }

  std::vector<uint8_t> bool_values(kv_heads * seq_length * head_dim);
  for (size_t index = 0; index < bool_values.size(); ++index) {
    bool_values[index] = static_cast<uint8_t>((index % 5) == 0);
  }
  array bool_source(bool_values.begin(), Shape{1, 2, 41, 64}, bool_);
  array bool_expanded = repeat(bool_source, repeats, 1, stream);
  bool_expanded.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const auto* bool_result = bool_expanded.data<bool>();
  for (size_t head = 0; head < 14; ++head) {
    size_t kv_head = head / repeats;
    for (size_t index = 0; index < seq_length * head_dim; ++index) {
      CHECK_EQ(
          bool_result[head * seq_length * head_dim + index],
          bool_values[kv_head * seq_length * head_dim + index] != 0);
    }
  }
}

namespace {

void check_indices(
    array value,
    const std::vector<uint32_t>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.dtype(), uint32);
  REQUIRE_EQ(value.size(), expected.size());
  const uint32_t* values = value.data<uint32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

} // namespace

TEST_CASE("argmax reduces last-axis rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Exact indices, a tie row, negative values, and NaN rows: mlx
  // reduction semantics propagate NaN, so the first NaN index wins for
  // argmin and argmax alike (the all-NaN row returns index 0).
  std::vector<float> xv = {
      1.0f, 3.0f, 2.0f,
      3.0f, 3.0f, 1.0f,
      -5.0f, -1.0f, -3.0f,
      1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f,
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN()};
  array x(xv.begin(), Shape{5, 3}, float32);
  check_indices(argmax(x, -1, false, stream), {1, 0, 1, 1, 0}, stream);

  // keepdims keeps the reduced axis with size 1.
  array kept = argmax(x, -1, true, stream);
  kept.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(kept.shape().size(), 2);
  CHECK_EQ(kept.shape(0), 5);
  CHECK_EQ(kept.shape(1), 1);
  check_indices(std::move(kept), {1, 0, 1, 1, 0}, stream);

  // A 1000-wide row crosses several 256-thread blocks. The spike at 512
  // ties the one at 768, so the first occurrence wins; the low spike at
  // 257 is the unique row minimum.
  std::vector<float> wide(1000);
  for (size_t index = 0; index < wide.size(); ++index) {
    wide[index] = static_cast<float>(static_cast<int>(index % 7) - 3);
  }
  wide[512] = 3.5f;
  wide[768] = 3.5f;
  wide[257] = -4.0f;
  array w(wide.begin(), Shape{1, 1000}, float32);
  check_indices(argmax(w, -1, false, stream), {512}, stream);
  check_indices(argmin(w, -1, false, stream), {257}, stream);

  // Non-suffix axes reduce through the moved-axis path: argmax over
  // axis 0 of {2, 3} picks the second row everywhere.
  array matrix({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  check_indices(argmax(matrix, 0, false, stream), {1, 1, 1}, stream);
}

TEST_CASE("argmin matches first-occurrence ties through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {
      1.0f, 3.0f, 2.0f,
      3.0f, 3.0f, 1.0f,
      -5.0f, -1.0f, -3.0f,
      1.0f, std::numeric_limits<float>::quiet_NaN(), 2.0f,
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::quiet_NaN()};
  array x(xv.begin(), Shape{5, 3}, float32);
  check_indices(argmin(x, -1, false, stream), {0, 2, 0, 1, 0}, stream);

  // A tie on the minimum keeps the first occurrence.
  array ties({2.0f, -1.0f, -1.0f, 4.0f}, {4}, float32);
  check_indices(argmin(ties, -1, false, stream), {1}, stream);
}

TEST_CASE("RandomBits matches the host threefry reference bit for bit") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<std::pair<uint32_t, uint32_t>> key_vectors = {
      {0x01234567u, 0x89abcdefu},
      {0x00000000u, 0x00000000u},
      {0xffffffffu, 0xffffffffu},
      {0xdeadbeefu, 0x12345678u}};
  // Word counts cover the even layout, the odd layout with its middle
  // word, and the single-word case.
  for (const auto& shape : {Shape{5}, Shape{4}, Shape{1}, Shape{3}}) {
    for (const auto& key_pair : key_vectors) {
      array key({key_pair.first, key_pair.second}, uint32);
      array bits = random::bits(shape, 4, key, stream);
      bits.eval();
      omarchy::get_command_encoder(stream).synchronize();
      REQUIRE_EQ(bits.size(), shape[0]);
      const uint32_t* words = bits.data<uint32_t>();
      for (uint32_t word = 0; word < bits.size(); ++word) {
        CHECK_EQ(
            words[word],
            host_random_word(key_pair, bits.size(), word));
      }
    }
  }

  // The mx.random.split key shape: one {2} key filling a {2, 2} output,
  // four words through the even layout.
  std::vector<uint32_t> kv = {0x01234567u, 0x89abcdefu};
  array split_key(kv.begin(), Shape{2}, uint32);
  array split_bits = random::bits(Shape{2, 2}, 4, split_key, stream);
  split_bits.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(split_bits.size(), 4u);
  const uint32_t* words = split_bits.data<uint32_t>();
  for (uint32_t word = 0; word < 4; ++word) {
    CHECK_EQ(
        words[word],
        host_random_word({kv[0], kv[1]}, 4, word));
  }
}

TEST_CASE("RandomBits widths match the CPU stream bit for bit") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  Stream cpu_stream = new_stream(Device::cpu);
  // Odd element counts land a partial word in every width: 1 and 2
  // byte elements make bytes_per_key miss a word boundary, and the
  // trailing bytes must match the CPU's copy_remaining byte for byte.
  for (int width : {1, 2, 4}) {
    for (auto& shape : {Shape{1}, Shape{3}, Shape{5}, Shape{7}, Shape{9}}) {
      // The same raw key words feed both streams; each bits call
      // schedules its own copy onto its stream.
      array key({0x01234567u, 0x89abcdefu}, uint32);
      array from_gpu = random::bits(shape, width, key, stream);
      array from_cpu = random::bits(shape, width, key, cpu_stream);
      from_gpu.eval();
      from_cpu.eval();
      omarchy::get_command_encoder(stream).synchronize();
      mlx::core::synchronize(cpu_stream);
      REQUIRE_EQ(from_gpu.nbytes(), from_cpu.nbytes());
      REQUIRE_EQ(
          std::vector<uint8_t>(
              from_gpu.data<uint8_t>(),
              from_gpu.data<uint8_t>() + from_gpu.nbytes()),
          std::vector<uint8_t>(
              from_cpu.data<uint8_t>(),
              from_cpu.data<uint8_t>() + from_cpu.nbytes()));
    }
  }
}

TEST_CASE("uniform with a pinned key is deterministic through Vulkan") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array key = random::key(0x5eed1234u);
  auto first = random::uniform(Shape{257}, float32, key, stream);
  auto second = random::uniform(Shape{257}, float32, key, stream);
  first.eval();
  second.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(first.size(), 257u);
  const float* a = first.data<float>();
  const float* b = second.data<float>();
  for (size_t index = 0; index < first.size(); ++index) {
    CHECK_EQ(a[index], b[index]);
    CHECK(a[index] >= 0.0f);
    CHECK(a[index] < 1.0f);
  }
}

TEST_CASE("categorical samples every class through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Uniform logits over four classes: 1000 draws must hit every class.
  array logits = zeros({1000, 4}, float32, stream);
  array samples = random::categorical(logits, -1, std::nullopt, stream);
  samples.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(samples.size(), 1000u);
  CHECK_EQ(samples.dtype(), uint32);
  std::vector<size_t> counts(4, 0);
  const uint32_t* drawn = samples.data<uint32_t>();
  for (size_t index = 0; index < samples.size(); ++index) {
    REQUIRE(drawn[index] < 4u);
    counts[drawn[index]]++;
  }
  for (size_t class_index = 0; class_index < 4; ++class_index) {
    CHECK(counts[class_index] >= 10);
    CHECK(counts[class_index] <= 500);
  }
}

TEST_CASE("RandomBits maps every width to its upstream dtype") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array key({1u, 2u}, uint32);
  // Widths 1, 2, and 4 all run the threefry kernel now; each keeps
  // its upstream dtype mapping (mlx/random.cpp). Invalid widths are
  // rejected upstream in mlx/random.cpp before a primitive exists.
  CHECK_EQ(random::bits(Shape{4}, 4, key, stream).dtype(), uint32);
  CHECK_EQ(random::bits(Shape{4}, 2, key, stream).dtype(), uint16);
  CHECK_EQ(random::bits(Shape{4}, 1, key, stream).dtype(), uint8);
  // The width check throws at construction, so wrap the build itself.
  std::string width_error = [&] {
    try {
      auto value = random::bits(Shape{4}, 3, key, stream);
      value.eval();
      return std::string{};
    } catch (const std::exception& error) {
      return std::string(error.what());
    }
  }();
  CHECK(width_error.find("Bit width must be in {1, 2, 4}") !=
        std::string::npos);
}

TEST_CASE("FP16 argmax matches host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {1.0f, 3.0f, 2.0f, -2.0f, -2.0f, -1.0f};
  array x(xv.begin(), Shape{2, 3}, float16);
  // Row 1 ties on -2, so argmin keeps the first occurrence.
  check_indices(argmin(x, -1, false, stream), {0, 0}, stream);
}

TEST_CASE("Cos and Sin match host references through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.5f, 1.0f, -0.75f, -2.0f, 3.5f};
  array x(xv.begin(), Shape{static_cast<int>(xv.size())}, float32);
  std::vector<float> cos_expected;
  std::vector<float> sin_expected;
  for (float value : xv) {
    cos_expected.push_back(std::cos(value));
    sin_expected.push_back(std::sin(value));
  }
  check_values(cos(x, stream), cos_expected, stream);
  check_values(sin(x, stream), sin_expected, stream);
}

TEST_CASE("Arange fills start plus step times index through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  check_values(
      arange(0, 10, 1, float32, stream),
      {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
      stream);

  // Upstream derives the length from ceil((stop - start) / step), so a
  // negative step over a descending range is a valid request.
  check_values(
      arange(2.5, 0.5, -0.25, float32, stream),
      {2.5f, 2.25f, 2.0f, 1.75f, 1.5f, 1.25f, 1.0f, 0.75f},
      stream);

  std::vector<int32_t> int_expected;
  for (int32_t value = 0; value < 10; ++value) {
    int_expected.push_back(value);
  }
  check_int32_values(arange(0, 10, 1, int32, stream), int_expected, stream);
  check_int32_values(
      arange(10, 0, -2, int32, stream), {10, 8, 6, 4, 2}, stream);

  check_int32_values(
      arange(0, 3, 0.2, int32, stream), std::vector<int32_t>(15, 0), stream);
  check_int32_values(
      arange(-1, -4, -0.9, int32, stream), {-1, -1, -1, -1}, stream);
  check_int32_values(
      arange(-1, -20, -1.2, int32, stream),
      {-1, -2, -3, -4, -5, -6, -7, -8,
       -9, -10, -11, -12, -13, -14, -15, -16},
      stream);
  check_int32_values(
      arange(0.9, 2.1, 0.2, int32, stream), {0, 1, 2, 3, 4, 5, 6}, stream);

  constexpr int64_t kWide = int64_t{1} << 40;
  check_int64_values(
      arange(kWide, kWide + 3, 1, int64, stream),
      {kWide, kWide + 1, kWide + 2},
      stream);
  check_int64_values(
      arange(-kWide, -kWide + 3, 1, int64, stream),
      {-kWide, -kWide + 1, -kWide + 2},
      stream);
  check_int64_values(
      arange(kWide + 3, kWide, -1, int64, stream),
      {kWide + 3, kWide + 2, kWide + 1},
      stream);
  check_uint64_values(
      arange(kWide, kWide + 3, 1, uint64, stream),
      {uint64_t(kWide), uint64_t(kWide + 1), uint64_t(kWide + 2)},
      stream);

  constexpr int32_t kMaxI32 = std::numeric_limits<int32_t>::max();
  check_int32_values(
      arange(
          static_cast<double>(kMaxI32) - 1.0,
          static_cast<double>(kMaxI32) + 3.0,
          1.0,
          int32,
          stream),
      {kMaxI32 - 1, kMaxI32, std::numeric_limits<int32_t>::min(),
       std::numeric_limits<int32_t>::min() + 1},
      stream);
  check_int32_values(
      arange(kWide, kWide + 3, 1, int32, stream), {0, 1, 2}, stream);

  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }
  array half = arange(0, 2, 0.5, float16, stream);
  check_values(
      astype(half, float32, stream), {0.0f, 0.5f, 1.0f, 1.5f}, stream, 1e-2);
}

TEST_CASE("grad of sum sin matches cos through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.1f, 0.4f, -0.9f, 1.3f};
  array x(xv.begin(), Shape{static_cast<int>(xv.size())}, float32);

  // Sin::vjp lowers to Cos and Multiply only, both backend-supported.
  auto fun = [&](const std::vector<array>& inputs) {
    return sum(sin(inputs[0], stream), stream);
  };
  auto [value, grads] = value_and_grad(fun, std::vector<int>{0})({x});

  std::vector<float> expected(xv.size());
  float expected_value = 0.0f;
  for (size_t index = 0; index < xv.size(); ++index) {
    expected[index] = std::cos(xv[index]);
    expected_value += std::sin(xv[index]);
  }
  check_values(value, {expected_value}, stream, 1e-5);
  check_values(grads.at(0), expected, stream, 1e-5);
}

TEST_CASE("sort and argsort order last-axis rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Shuffled rows with duplicates and negatives. Equal keys keep source
  // order, which the argsort indices pin exactly: row 1 ties on zero at
  // indices 0, 1, and 4, and row 2 ties on both -3.5 and 2.5.
  std::vector<float> xv = {
      3.0f, -1.0f, 2.0f, -1.0f, 0.5f,
      0.0f, -0.0f, 5.0f, -7.5f, 0.0f,
      2.5f, -3.5f, 2.5f, -3.5f, 1.0f};
  array x(xv.begin(), Shape{3, 5}, float32);
  check_values(
      sort(x, -1, stream),
      {-1.0f, -1.0f, 0.5f, 2.0f, 3.0f, -7.5f, 0.0f, -0.0f, 0.0f, 5.0f,
       -3.5f, -3.5f, 1.0f, 2.5f, 2.5f},
      stream);
  check_indices(
      argsort(x, -1, stream),
      {1, 3, 4, 2, 0, 3, 0, 1, 4, 2, 1, 3, 4, 0, 2},
      stream);

  // A 1-D row is its own last axis.
  array flat({4.0f, -2.0f, 4.0f, 0.0f}, float32);
  check_values(sort(flat, 0, stream), {-2.0f, 0.0f, 4.0f, 4.0f}, stream);
  check_indices(argsort(flat, 0, stream), {1, 3, 0, 2}, stream);
}

TEST_CASE("sort places NaN after every number through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // The kernel mirrors the upstream CPU comparator: NaN sorts after every
  // number, and two NaN keys order by source index, so the NaN pair at
  // indices 1 and 3 yields 1 then 3.
  float nan = std::numeric_limits<float>::quiet_NaN();
  std::vector<float> xv = {1.0f, nan, -2.0f, nan, 0.0f};
  array x(xv.begin(), Shape{5}, float32);
  array order = argsort(x, -1, stream);
  check_indices(std::move(order), {2, 4, 0, 1, 3}, stream);

  array sorted = sort(x, -1, stream);
  sorted.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(sorted.size(), 5);
  const float* values = sorted.data<float>();
  CHECK_EQ(values[0], -2.0f);
  CHECK_EQ(values[1], 0.0f);
  CHECK_EQ(values[2], 1.0f);
  CHECK(std::isnan(values[3]));
  CHECK(std::isnan(values[4]));
}

TEST_CASE("complex sort and argsort keep the stable MLX order through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  float nan = std::numeric_limits<float>::quiet_NaN();
  auto same_component = [](float lhs, float rhs) {
    return lhs == rhs || (std::isnan(lhs) && std::isnan(rhs));
  };
  auto check_result = [&](const array& input,
                          int axis,
                          const std::vector<complex64_t>& expected_values,
                          const std::vector<uint32_t>& expected_order) {
    array sorted = sort(input, axis, stream);
    array order = argsort(input, axis, stream);
    sorted.eval();
    order.eval();
    omarchy::get_command_encoder(stream).synchronize();
    REQUIRE_EQ(sorted.size(), expected_values.size());
    REQUIRE_EQ(order.size(), expected_order.size());
    const complex64_t* actual_values = sorted.data<complex64_t>();
    const uint32_t* actual_order = order.data<uint32_t>();
    for (size_t i = 0; i < expected_values.size(); ++i) {
      CHECK(same_component(actual_values[i].real(), expected_values[i].real()));
      CHECK(same_component(actual_values[i].imag(), expected_values[i].imag()));
      CHECK_EQ(actual_order[i], expected_order[i]);
    }
  };
  auto check_row = [&](const std::vector<complex64_t>& values) {
    std::vector<uint32_t> expected_order(values.size());
    std::iota(expected_order.begin(), expected_order.end(), 0u);
    std::stable_sort(
        expected_order.begin(),
        expected_order.end(),
        [&](uint32_t lhs, uint32_t rhs) {
          const auto& a = values[lhs];
          const auto& b = values[rhs];
          bool nan_a = std::isnan(a.real()) || std::isnan(a.imag());
          bool nan_b = std::isnan(b.real()) || std::isnan(b.imag());
          if (nan_a != nan_b) {
            return nan_b;
          }
          if (nan_a) {
            return lhs < rhs;
          }
          if (a.real() != b.real()) {
            return a.real() < b.real();
          }
          return a.imag() < b.imag() ||
              (a.imag() == b.imag() && lhs < rhs);
        });
    std::vector<complex64_t> expected_values(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
      expected_values[i] = values[expected_order[i]];
    }
    array input(values.begin(), Shape{static_cast<int>(values.size())}, complex64);
    check_result(input, -1, expected_values, expected_order);
  };

  check_row({
      complex64_t{2.0f, 1.0f},
      complex64_t{1.0f, 3.0f},
      complex64_t{1.0f, -2.0f},
      complex64_t{nan, 4.0f},
      complex64_t{0.0f, nan},
      complex64_t{1.0f, -2.0f},
      complex64_t{nan, -1.0f}});
  check_row({
      complex64_t{3.0f, 1.0f},
      complex64_t{nan, 2.0f},
      complex64_t{2.0f, 1.0f},
      complex64_t{0.0f, 1.0f}});
  check_row({
      complex64_t{1.0f, 0.0f},
      complex64_t{nan, 0.0f},
      complex64_t{0.0f, 0.0f}});

  std::vector<complex64_t> matrix_values = {
      {2.0f, 1.0f}, {1.0f, 3.0f}, {1.0f, -2.0f},
      {9.0f, 9.0f}, {9.0f, 9.0f}, {9.0f, 9.0f},
      {0.0f, 4.0f}, {2.0f, -1.0f}, {0.0f, -2.0f},
      {8.0f, 8.0f}, {8.0f, 8.0f}, {8.0f, 8.0f},
      {1.0f, 0.0f}, {1.0f, -4.0f}, {0.0f, -2.0f}};
  array matrix(matrix_values.begin(), Shape{5, 3}, complex64);
  array strided = slice(matrix, {0, 0}, {5, 3}, {2, 1}, stream);
  check_result(
      strided,
      0,
      {{0.0f, 4.0f}, {1.0f, -4.0f}, {0.0f, -2.0f},
       {1.0f, 0.0f}, {1.0f, 3.0f}, {0.0f, -2.0f},
       {2.0f, 1.0f}, {2.0f, -1.0f}, {1.0f, -2.0f}},
      {1, 2, 1, 2, 0, 2, 0, 1, 0});

  constexpr size_t length = 2051;
  std::vector<complex64_t> wide(length);
  for (size_t i = 0; i < length; ++i) {
    wide[i] = complex64_t{
        static_cast<float>(static_cast<int>((i * 37) % 29) - 14),
        static_cast<float>(static_cast<int>((i * 17) % 11) - 5)};
  }
  wide[5] = complex64_t{nan, 4.0f};
  wide[1024] = complex64_t{0.0f, nan};
  wide[2050] = complex64_t{nan, -1.0f};
  check_row(wide);
}

TEST_CASE("sort and argsort handle wide rows through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // A 1000-wide row pads to 1024 inside the kernel. Each row holds a
  // different shuffle of the generator, so the host reference runs
  // stable_sort under the same (value, index) rule once per row.
  std::vector<float> wide(2000);
  for (size_t index = 0; index < wide.size(); ++index) {
    wide[index] =
        static_cast<float>((static_cast<int>(index * 37) % 101) - 50);
  }
  array x(wide.begin(), Shape{2, 1000}, float32);
  std::vector<float> sorted_expected;
  std::vector<uint32_t> order_expected;
  for (int repeat = 0; repeat < 2; ++repeat) {
    std::vector<float> row(
        wide.begin() + repeat * 1000, wide.begin() + (repeat + 1) * 1000);
    std::vector<uint32_t> order(1000);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return row[a] < row[b] || (row[a] == row[b] && a < b);
    });
    std::stable_sort(row.begin(), row.end());
    sorted_expected.insert(sorted_expected.end(), row.begin(), row.end());
    order_expected.insert(order_expected.end(), order.begin(), order.end());
  }
  check_values(sort(x, -1, stream), sorted_expected, stream);
  check_indices(argsort(x, -1, stream), order_expected, stream);

  // An exact 1024-wide descending row skips the padding path.
  std::vector<float> exact(1024);
  for (size_t index = 0; index < exact.size(); ++index) {
    exact[index] = static_cast<float>(1023 - static_cast<int>(index));
  }
  array y(exact.begin(), Shape{1, 1024}, float32);
  std::vector<uint32_t> exact_order(1024);
  std::iota(exact_order.begin(), exact_order.end(), 0u);
  std::reverse(exact_order.begin(), exact_order.end());
  check_indices(argsort(y, -1, stream), exact_order, stream);
}

TEST_CASE("sort rejects non-float dtypes with named errors") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Long rows sort now (the merge cases below); int64 still refuses by
  // name.
  array ints({3, 1, 2}, int64);
  std::string dtype_error = evaluation_error(sort(ints, -1, stream));
  CHECK(dtype_error.find("[omarchy] Sort dtype") != std::string::npos);
  std::string index_dtype_error =
      evaluation_error(argsort(ints, -1, stream));
  CHECK(
      index_dtype_error.find("[omarchy] ArgSort dtype") != std::string::npos);

  array matrix({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, float32);
  check_values(sort(matrix, 0, stream), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, stream);
  check_indices(argsort(matrix, 0, stream), {0, 0, 0, 1, 1, 1}, stream);
}

TEST_CASE("sort and argsort merge long rows exactly through global memory") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Every length crosses the 1024-element chunk stage and one or more
  // global merge stages; 65537 pads to 131072. Two rows per length keep
  // the padded rows from contaminating each other, and the generator
  // repeats values so stability is observable through the argsort.
  auto make_row = [](size_t length, int seed) {
    std::vector<float> row(length);
    for (size_t i = 0; i < length; ++i) {
      row[i] = static_cast<float>((i * i + 3 * i + seed * 571) % 997) *
              0.25f -
          100.0f;
    }
    return row;
  };
  auto reference = [](const std::vector<float>& row) {
    std::vector<uint32_t> order(row.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      float va = row[a];
      float vb = row[b];
      bool nan_a = std::isnan(va);
      bool nan_b = std::isnan(vb);
      if (nan_a != nan_b) {
        return nan_b;
      }
      if (nan_a) {
        return a < b;
      }
      return va < vb || (va == vb && a < b);
    });
    std::vector<float> values(row.size());
    for (size_t i = 0; i < row.size(); ++i) {
      values[i] = row[order[i]];
    }
    return std::make_pair(values, order);
  };

  for (size_t length : {size_t{1025}, size_t{4097}, size_t{32769},
                        size_t{65537}}) {
    CAPTURE(length);
    std::vector<float> rows = make_row(length, 1);
    std::vector<float> row_two = make_row(length, 2);
    rows.insert(rows.end(), row_two.begin(), row_two.end());
    array x(rows.begin(), Shape{2, static_cast<int>(length)}, float32);

    array sorted = sort(x, -1, stream);
    array order = argsort(x, -1, stream);
    sorted.eval();
    order.eval();
    omarchy::get_command_encoder(stream).synchronize();
    for (int repeat = 0; repeat < 2; ++repeat) {
      auto [expected_values, expected_order] =
          reference(make_row(length, repeat == 0 ? 1 : 2));
      const float* values =
          sorted.data<float>() + repeat * length;
      const uint32_t* indices =
          order.data<uint32_t>() + repeat * length;
      for (size_t i = 0; i < length; ++i) {
        CHECK_EQ(values[i], expected_values[i]);
        CHECK_EQ(indices[i], expected_order[i]);
      }
    }
  }

  // NaNs and a signed-zero pair placed across chunk boundaries: the
  // NaNs must land after every number in source order and the pads
  // (which are NaN words) must stay above every real element.
  {
    constexpr size_t length = 4097;
    std::vector<float> row = make_row(length, 3);
    row[5] = std::numeric_limits<float>::quiet_NaN();
    row[1024] = std::numeric_limits<float>::quiet_NaN();
    row[1025] = -0.0f;
    row[2048] = 0.0f;
    row[4096] = std::numeric_limits<float>::quiet_NaN();
    array x(row.begin(), Shape{1, static_cast<int>(length)}, float32);
    array sorted = sort(x, -1, stream);
    array order = argsort(x, -1, stream);
    sorted.eval();
    order.eval();
    omarchy::get_command_encoder(stream).synchronize();
    auto [expected_values, expected_order] = reference(row);
    const float* values = sorted.data<float>();
    const uint32_t* indices = order.data<uint32_t>();
    for (size_t i = 0; i < length; ++i) {
      bool value_ok =
          values[i] == expected_values[i] ||
          (std::isnan(values[i]) && std::isnan(expected_values[i]));
      CHECK(value_ok);
      CHECK_EQ(indices[i], expected_order[i]);
    }
    CHECK(std::isnan(values[length - 1]));
    CHECK(std::isnan(values[length - 2]));
    CHECK(std::isnan(values[length - 3]));
  }

  // int32 through the same merge path: negatives flip to order-preserving
  // unsigned keys, duplicates tie on source index, and a real INT32_MAX
  // element shares the pad key but must stay inside the row.
  {
    constexpr size_t length = 4097;
    std::vector<int32_t> row(length);
    for (size_t i = 0; i < length; ++i) {
      row[i] = static_cast<int32_t>((i * 48271 + 13) % 2009) - 1000;
    }
    row[17] = std::numeric_limits<int32_t>::max();
    row[2048] = -1;
    row[4096] = std::numeric_limits<int32_t>::max();
    array x(row.begin(), Shape{1, static_cast<int>(length)}, int32);
    std::vector<uint32_t> order(length);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      uint32_t ka = static_cast<uint32_t>(row[a]) ^ 0x80000000u;
      uint32_t kb = static_cast<uint32_t>(row[b]) ^ 0x80000000u;
      return ka < kb || (ka == kb && a < b);
    });
    array sorted_order = argsort(x, -1, stream);
    sorted_order.eval();
    omarchy::get_command_encoder(stream).synchronize();
    const uint32_t* indices = sorted_order.data<uint32_t>();
    for (size_t i = 0; i < length; ++i) {
      CHECK_EQ(indices[i], order[i]);
    }
    array sorted_values = sort(x, -1, stream);
    sorted_values.eval();
    omarchy::get_command_encoder(stream).synchronize();
    const int32_t* values = sorted_values.data<int32_t>();
    for (size_t i = 0; i < length; ++i) {
      CHECK_EQ(values[i], row[order[i]]);
    }
  }

  // FP16 rides the same merge kernel family with 16-bit storage; gated
  // like the narrow FP16 case above.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    constexpr size_t length = 1031;
    std::vector<float> row = make_row(length, 4);
    array x(row.begin(), Shape{1, static_cast<int>(length)}, float16);
    array order = argsort(x, -1, stream);
    order.eval();
    omarchy::get_command_encoder(stream).synchronize();
    std::vector<uint32_t> expected(length);
    std::iota(expected.begin(), expected.end(), 0u);
    std::stable_sort(expected.begin(), expected.end(), [&](uint32_t a, uint32_t b) {
      return row[a] < row[b] || (row[a] == row[b] && a < b);
    });
    const uint32_t* indices = order.data<uint32_t>();
    for (size_t i = 0; i < length; ++i) {
      CHECK_EQ(indices[i], expected[i]);
    }
  }
}

TEST_CASE(
    "integer sort and argsort match the host stable order through Vulkan "
    "compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // int32: negatives, mixed signs, duplicates, zero, INT32_MIN and
  // INT32_MAX. Equal keys order by source index, so the argsort pins
  // the unique stable order the float path gives.
  std::vector<int32_t> iv = {
      7, -7, 0, 7, -7,
      0, std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::max(), 0, -1};
  array xi(iv.begin(), Shape{2, 5}, int32);
  std::vector<int32_t> sorted_iv;
  std::vector<uint32_t> order_iv;
  for (int repeat = 0; repeat < 2; ++repeat) {
    std::vector<int32_t> row(
        iv.begin() + repeat * 5, iv.begin() + (repeat + 1) * 5);
    std::vector<uint32_t> order(5);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return row[a] < row[b] || (row[a] == row[b] && a < b);
    });
    std::stable_sort(row.begin(), row.end());
    sorted_iv.insert(sorted_iv.end(), row.begin(), row.end());
    order_iv.insert(order_iv.end(), order.begin(), order.end());
  }
  array sorted_xi = sort(xi, -1, stream);
  sorted_xi.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(sorted_xi.size(), sorted_iv.size());
  const int32_t* got_ints = sorted_xi.data<int32_t>();
  for (size_t index = 0; index < sorted_iv.size(); ++index) {
    CHECK_EQ(got_ints[index], sorted_iv[index]);
  }
  check_indices(argsort(xi, -1, stream), order_iv, stream);

  // uint32: straight magnitude order, duplicates, zero, UINT32_MAX,
  // across two batched rows.
  std::vector<uint32_t> uv = {
      5u, 0u, std::numeric_limits<uint32_t>::max(), 5u,
      0u, 1u, std::numeric_limits<uint32_t>::max(), 3u};
  array xu(uv.begin(), Shape{2, 4}, uint32);
  std::vector<uint32_t> sorted_uv;
  std::vector<uint32_t> order_uv;
  for (int repeat = 0; repeat < 2; ++repeat) {
    std::vector<uint32_t> row(
        uv.begin() + repeat * 4, uv.begin() + (repeat + 1) * 4);
    std::vector<uint32_t> order(4);
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return row[a] < row[b] || (row[a] == row[b] && a < b);
    });
    std::stable_sort(row.begin(), row.end());
    sorted_uv.insert(sorted_uv.end(), row.begin(), row.end());
    order_uv.insert(order_uv.end(), order.begin(), order.end());
  }
  check_indices(argsort(xu, -1, stream), order_uv, stream);
  array sorted_xu = sort(xu, -1, stream);
  sorted_xu.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const uint32_t* got_uints = sorted_xu.data<uint32_t>();
  for (size_t index = 0; index < sorted_uv.size(); ++index) {
    CHECK_EQ(got_uints[index], sorted_uv[index]);
  }

  // The 8/16-bit families ride the widened key maps: sign-bit flips
  // for the signed modes, magnitude order for the unsigned ones.
  auto narrow_sort_case = [&](Dtype dtype, auto values, Shape shape) {
    using T = typename decltype(values)::value_type;
    array input(values.begin(), shape, dtype);
    size_t row_length = shape.back();
    size_t rows = values.size() / row_length;
    std::vector<T> sorted;
    for (size_t r = 0; r < rows; ++r) {
      std::vector<T> row(
          values.begin() + r * row_length,
          values.begin() + (r + 1) * row_length);
      std::stable_sort(row.begin(), row.end());
      sorted.insert(sorted.end(), row.begin(), row.end());
    }
    array out = sort(input, -1, stream);
    out.eval();
    omarchy::get_command_encoder(stream).synchronize();
    REQUIRE_EQ(out.size(), sorted.size());
    const T* got = out.data<T>();
    for (size_t index = 0; index < sorted.size(); ++index) {
      CHECK_EQ(got[index], sorted[index]);
    }
  };
  narrow_sort_case(
      int8,
      std::vector<int8_t>{7, -7, 0, 7,
                          std::numeric_limits<int8_t>::min(), 5,
                          std::numeric_limits<int8_t>::max(), -1},
      Shape{2, 4});
  narrow_sort_case(
      uint8,
      std::vector<uint8_t>{5, 0, std::numeric_limits<uint8_t>::max(), 5, 0,
                           1, 3, std::numeric_limits<uint8_t>::max()},
      Shape{2, 4});
  narrow_sort_case(
      int16,
      std::vector<int16_t>{-300, 300, 0, -300, 7,
                           std::numeric_limits<int16_t>::min(),
                           std::numeric_limits<int16_t>::max(), 7},
      Shape{2, 4});
  narrow_sort_case(
      uint16,
      std::vector<uint16_t>{5, 0, std::numeric_limits<uint16_t>::max(), 5, 0,
                            1, 3, std::numeric_limits<uint16_t>::max()},
      Shape{2, 4});

  // A 1000-wide int32 row pads to 1024 inside the kernel; the signed
  // transform must stay monotone across the whole padded width.
  std::vector<int32_t> wide(1000);
  for (size_t index = 0; index < wide.size(); ++index) {
    wide[index] = static_cast<int32_t>((static_cast<int>(index) * 37) % 201) -
        100;
  }
  array xw(wide.begin(), Shape{1, 1000}, int32);
  std::vector<uint32_t> wide_order(1000);
  std::iota(wide_order.begin(), wide_order.end(), 0u);
  std::stable_sort(
      wide_order.begin(), wide_order.end(), [&](uint32_t a, uint32_t b) {
        return wide[a] < wide[b] || (wide[a] == wide[b] && a < b);
      });
  check_indices(argsort(xw, -1, stream), wide_order, stream);
  std::vector<int32_t> wide_sorted(wide);
  std::stable_sort(wide_sorted.begin(), wide_sorted.end());
  array sorted_xw = sort(xw, -1, stream);
  sorted_xw.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const int32_t* got_wide = sorted_xw.data<int32_t>();
  for (size_t index = 0; index < wide_sorted.size(); ++index) {
    CHECK_EQ(got_wide[index], wide_sorted[index]);
  }
}

TEST_CASE("partition redirects to sort and topk returns the right set") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {3.0f, -1.0f, 2.0f, -1.0f, 0.5f};
  array x(xv.begin(), Shape{5}, float32);

  // A full sort satisfies the partition contract, so every position holds
  // the sorted value and argpartition equals argsort exactly.
  check_values(
      partition(x, 2, -1, stream), {-1.0f, -1.0f, 0.5f, 2.0f, 3.0f}, stream);
  check_indices(argpartition(x, 2, -1, stream), {1, 3, 4, 2, 0}, stream);

  // mx.topk lowers to partition plus a tail slice, so a 1-D topk returns
  // the k largest in ascending order through the same path.
  check_values(topk(x, 2, -1, stream), {2.0f, 3.0f}, stream);

  // The same full-sort redirect satisfies the partition contract for
  // several rows at once.
  array m({5.0f, 1.0f, 4.0f, 2.0f, 3.0f, 0.0f}, {2, 3}, float32);
  check_values(
      partition(m, 0, -1, stream),
      {1.0f, 4.0f, 5.0f, 0.0f, 2.0f, 3.0f},
      stream);

  // Partition over axis 0 routes through the moved-axis path: each
  // column sorts independently, so the result is [[2, 1, 0], [5, 3, 4]]
  // in row-major order. A column-major temp once produced
  // {1, 5, 2, 4, 0, 3} here.
  check_values(partition(m, 0, 0, stream), {2.0f, 1.0f, 0.0f, 5.0f, 3.0f, 4.0f}, stream);
  check_indices(argsort(m, 0, stream), {1, 0, 1, 0, 1, 0}, stream);
  // A 3-D case where the moved shape is not a transpose of the input.
  array t(
      {9.0f, 2.0f, 7.0f, 4.0f, 1.0f, 8.0f, 3.0f, 6.0f, 5.0f, 0.0f, 11.0f, 10.0f},
      {2, 2, 3},
      float32);
  check_values(
      sort(t, 1, stream),
      {4.0f, 1.0f, 7.0f, 9.0f, 2.0f, 8.0f, 0.0f, 6.0f, 5.0f, 3.0f, 11.0f, 10.0f},
      stream);
}

TEST_CASE("strided slice views materialize exact values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // db10f53 keeps evaluated slices as retained views; a host read of a
  // retained view goes through contiguous, which materializes via the
  // strided-copy engine (the corrected contract from the
  // kv-state-views receipt). Values below are unchanged.
  // Column 2 of each row: the tail-slice shape that mx.topk lowers to.
  array m({5.0f, 1.0f, 4.0f, 2.0f, 3.0f, 0.0f}, {2, 3}, float32);
  array m_tail = slice(m, {0, 2}, {2, 3}, stream);
  m_tail.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_FALSE(m_tail.flags().contiguous);
  check_values(contiguous(m_tail, false, stream), {4.0f, 0.0f}, stream);

  // mx.topk(m, 2, -1) is partition plus a strided tail slice; the
  // partition output is exact and contiguous flattens the tail view.
  check_values(
      contiguous(topk(m, 2, -1, stream), false, stream),
      {4.0f, 5.0f, 2.0f, 3.0f},
      stream);

  // Inner-axis slices leave gaps between rows.
  array p({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
          {2, 4},
          float32);
  check_values(
      contiguous(slice(p, {0, 1}, {2, 3}, {1, 1}, stream), false, stream),
      {1.0f, 2.0f, 5.0f, 6.0f},
      stream);

  // A 1-D stride-2 slice gathers every other element.
  array base({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
             {8},
             float32);
  check_values(
      contiguous(slice(base, {0}, {7}, {2}, stream), false, stream),
      {0.0f, 2.0f, 4.0f, 6.0f},
      stream);
}

TEST_CASE("elementwise ops over strided slice views match host values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // The mx.fast.rope half-split pattern: x[..., 0:2] and x[..., 2:4] are
  // strided views over the same parent rows.
  array x({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f},
          {2, 4},
          float32);
  array x1 = slice(x, {0, 0}, {2, 2}, {1, 1}, stream);
  array x2 = slice(x, {0, 2}, {2, 4}, {1, 1}, stream);

  check_values(multiply(x1, x2, stream), {0.0f, 3.0f, 24.0f, 35.0f}, stream);
  check_values(subtract(x2, x1, stream), {2.0f, 2.0f, 2.0f, 2.0f}, stream);
  check_values(add(x1, x2, stream), {2.0f, 4.0f, 10.0f, 12.0f}, stream);
}

TEST_CASE("FP16 sort and argsort match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }

  // Row ties on -2 at indices 1 and 3, so argsort keeps 1 then 3.
  std::vector<float> xv = {1.0f, -2.0f, 0.5f, -2.0f};
  array x(xv.begin(), Shape{4}, float16);
  check_values(
      astype(sort(x, -1, stream), float32, stream),
      {-2.0f, -2.0f, 0.5f, 1.0f},
      stream,
      1e-2);
  check_indices(argsort(x, -1, stream), {1, 3, 2, 0}, stream);
}

TEST_CASE("Equal matches host references across dtypes and broadcast shapes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // The scalar-only shape=[] case that the sampler chain hits.
  array scalar_equal = equal(array(2.0f), array(3.0f), stream);
  scalar_equal.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(scalar_equal.shape().size(), 0u);
  CHECK_EQ(scalar_equal.data<bool>()[0], false);
  scalar_equal = equal(array(4.0f), array(4.0f), stream);
  scalar_equal.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(scalar_equal.data<bool>()[0], true);

  // Row against a scalar and row against a row, float32 and int32.
  std::vector<float> xv = {1.0f, 2.0f, 3.0f, 4.0f};
  array x(xv.begin(), Shape{4}, float32);
  std::vector<bool> expected = {false, true, false, false};
  array row_equal = equal(x, array(2.0f), stream);
  row_equal.eval();
  omarchy::get_command_encoder(stream).synchronize();
  for (size_t index = 0; index < 4; ++index) {
    CHECK_EQ(row_equal.data<bool>()[index], expected[index]);
  }
  std::vector<int32_t> iv = {7, 0, -3, 7};
  array i(iv.begin(), Shape{4}, int32);
  std::vector<bool> iexpected = {true, false, false, true};
  array i_equal = equal(i, array(7, int32), stream);
  i_equal.eval();
  omarchy::get_command_encoder(stream).synchronize();
  for (size_t index = 0; index < 4; ++index) {
    CHECK_EQ(i_equal.data<bool>()[index], iexpected[index]);
  }

  // A general broadcast over the leading axis uses the stride transport.
  array wide(xv.begin(), Shape{1, 4}, float32);
  std::vector<float> yv = {1.0f, 9.0f, 9.0f, 9.0f, 9.0f, 2.0f, 9.0f, 9.0f};
  array y(yv.begin(), Shape{2, 4}, float32);
  array broadcast_equal = equal(y, wide, stream);
  broadcast_equal.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const bool* broadcast_bits = broadcast_equal.data<bool>();
  CHECK_EQ(broadcast_bits[0], true);
  CHECK_EQ(broadcast_bits[1], false);
  CHECK_EQ(broadcast_bits[4], false);
  CHECK_EQ(broadcast_bits[5], true);

  // Float16 and bfloat16 compare through the 16-bit storage variants.
  std::vector<float> hv = {0.5f, 1.5f, -2.0f, 0.25f};
  array h(hv.begin(), Shape{4}, float16);
  array h_equal = equal(h, array(0.5f, float16), stream);
  h_equal.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(h_equal.data<bool>()[0], true);
  CHECK_EQ(h_equal.data<bool>()[1], false);
  array b(hv.begin(), Shape{4}, bfloat16);
  array b_equal = equal(b, array(-2.0f, bfloat16), stream);
  b_equal.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(b_equal.data<bool>()[2], true);
  CHECK_EQ(b_equal.data<bool>()[3], false);

  // Equality with NaN stays false, matching the upstream comparator.
  std::vector<float> nv = {std::nanf("")};
  array n(nv.begin(), Shape{1}, float32);
  array nan_equal = equal(n, n, stream);
  nan_equal.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(nan_equal.data<bool>()[0], false);

  // uint64 compares through the CompareU64 blob: one 64-bit word per
  // element. The third pair differs only in the high word, the read a
  // 32-bit kernel cannot see.
  std::vector<uint64_t> uv = {
      0xa11cc311cb6acd70ull,
      0x7a375ac3ebb533f3ull,
      0x0000000100000000ull,
      0xffffffffffffffffull};
  std::vector<uint64_t> uv2 = {
      0xa11cc311cb6acd70ull,
      0x7a375ac3ebb533f3ull,
      0x0000000100000001ull,
      0xffffffffffffffffull};
  array u(uv.begin(), Shape{4}, uint64);
  array u2(uv2.begin(), Shape{4}, uint64);
  array u_equal = equal(u, u2, stream);
  u_equal.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(u_equal.data<bool>()[0], true);
  CHECK_EQ(u_equal.data<bool>()[1], true);
  CHECK_EQ(u_equal.data<bool>()[2], false);
  CHECK_EQ(u_equal.data<bool>()[3], true);

  // sign on uint64 keeps the upstream unsigned rule (0 -> 0, else 1).
  std::vector<uint64_t> sv = {
      0xb400515a4f673424ull, 0x0000000000000000ull, 0x0000000000000001ull};
  array s_in(sv.begin(), Shape{3}, uint64);
  array s_out = sign(s_in, stream);
  s_out.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(s_out.data<uint64_t>()[0], 1u);
  CHECK_EQ(s_out.data<uint64_t>()[1], 0u);
  CHECK_EQ(s_out.data<uint64_t>()[2], 1u);
  // all(equal(sign(x), expected)) exercises Equal on the bool output
  // of the sign comparison chain, the shape array_equal builds.
  CHECK(all(equal(sign(s_in, stream), s_out), false, stream).item<bool>());
}

TEST_CASE("bool Abs and Sign are the identity through the byte lanes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Upstream abs on bool is the identity and sign on bool is
  // x != 0, which is the value itself for canonical 0/1 bytes.
  std::vector<bool> values = {false, true, false, true, true, false, true};
  array x(values.begin(), Shape{7}, bool_);
  array a = abs(x, stream);
  a.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(a.dtype(), bool_);
  const bool* a_bytes = a.data<bool>();
  for (size_t index = 0; index < 7; ++index) {
    CHECK_EQ(a_bytes[index], values[index]);
  }
  array s = sign(x, stream);
  s.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(s.dtype(), bool_);
  const bool* s_bytes = s.data<bool>();
  for (size_t index = 0; index < 7; ++index) {
    CHECK_EQ(s_bytes[index], values[index]);
  }
}

TEST_CASE("sin and cos on an empty argument return empty") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // The trig magnitude gate must not run its max() host check over a
  // size-0 axis; upstream just returns the empty array.
  array empty = array({});
  array s = sin(empty, stream);
  s.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(s.size(), 0u);
  CHECK_EQ(s.dtype(), float32);
  array c = cos(empty, stream);
  c.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(c.size(), 0u);
  CHECK_EQ(c.dtype(), float32);
}

TEST_CASE("RandomBits batched keys keep every key's elements in place") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  Stream cpu_stream = new_stream(Device::cpu);

  // Widths 1 and 2 with per-key element counts that end mid-word
  // under multiple keys: the regression behind the vmap
  // take(out, array(1), 0) mismatch, where a word-granular layout let
  // key i's masked tail word zero key i+1's leading bytes and shift
  // every later element by one slot. Host reference: the same raw key
  // words through the CPU stream, row by row.
  std::vector<uint32_t> kv = {
      0x01234567u, 0x89abcdefu,
      0xdeadbeefu, 0x12345678u,
      0x00000000u, 0xffffffffu};
  array keys(kv.begin(), Shape{3, 2}, uint32);
  for (int width : {1, 2, 4}) {
    for (auto& row_length : {size_t(5), size_t(3), size_t(9)}) {
      auto fn = [&](array k) {
        return random::bits(
            Shape{static_cast<int>(row_length)}, width, k, stream);
      };
      array out = vmap(fn, 0)(keys);
      out.eval();
      omarchy::get_command_encoder(stream).synchronize();
      REQUIRE_EQ(out.size(), keys.shape(0) * row_length);
      for (int row = 0; row < keys.shape(0); ++row) {
        std::vector<uint32_t> row_kv{kv[row * 2], kv[row * 2 + 1]};
        array row_key(row_kv.begin(), Shape{2}, uint32);
        array ref = random::bits(
            Shape{static_cast<int>(row_length)}, width, row_key, cpu_stream);
        ref.eval();
        mlx::core::synchronize(cpu_stream);
        const uint8_t* got = out.data<uint8_t>();
        const uint8_t* want = ref.data<uint8_t>();
        size_t base = row * row_length * width;
        for (size_t byte = 0; byte < row_length * width; ++byte) {
          CHECK_EQ(got[base + byte], want[byte]);
        }
      }
    }
  }
}

TEST_CASE("isinf composes Equal and LogicalOr through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {
      0.0f,
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity(),
      1.5f,
      std::nanf("")};
  array x(xv.begin(), Shape{5}, float32);
  array flags = isinf(x, stream);
  flags.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(flags.data<bool>()[0], false);
  CHECK_EQ(flags.data<bool>()[1], true);
  CHECK_EQ(flags.data<bool>()[2], true);
  CHECK_EQ(flags.data<bool>()[3], false);
  CHECK_EQ(flags.data<bool>()[4], false);

  // The sampler calls isinf on the scalar row max.
  array scalar_flags = isinf(
      array(std::numeric_limits<float>::infinity()), stream);
  scalar_flags.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(scalar_flags.data<bool>()[0], true);
}

TEST_CASE("Select picks between two row-contiguous values under a scalar condition") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> tv = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> fv = {10.0f, 20.0f, 30.0f, 40.0f};
  array t(tv.begin(), Shape{4}, float32);
  array f(fv.begin(), Shape{4}, float32);
  // False everywhere, so the result is the whole false operand.
  check_values(where(array(false), t, f, stream), fv, stream);
  // True everywhere, so the result is the whole true operand.
  check_values(where(array(true), t, f, stream), tv, stream);
  // A row condition mixes both operands through the broadcast views.
  std::vector<float> cv = {1.0f, 0.0f, 1.0f, 0.0f};
  array row_condition(cv.begin(), Shape{4}, bool_);
  check_values(
      where(row_condition, t, f, stream), {1.0f, 20.0f, 3.0f, 40.0f}, stream);
}

TEST_CASE("bool to float32 casts exact zero and one through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {1.0f, 2.0f, 2.0f, 0.0f};
  array x(xv.begin(), Shape{4}, float32);
  array mask = astype(equal(x, array(2.0f), stream), float32, stream);
  check_values(mask, {0.0f, 1.0f, 1.0f, 0.0f}, stream);
}

TEST_CASE("bool to float16 and bfloat16 casts exact zero and one") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  std::vector<float> xv = {1.0f, 2.0f, 2.0f, 0.0f};
  array x(xv.begin(), Shape{4}, float32);
  if (capabilities.shader_float16 && capabilities.storage_buffer_16bit_access) {
    check_values(
        astype(
            astype(equal(x, array(2.0f), stream), float16, stream),
            float32,
            stream),
        {0.0f, 1.0f, 1.0f, 0.0f},
        stream,
        1e-3);
  }
  if (capabilities.storage_buffer_16bit_access && capabilities.shader_int16) {
    check_values(
        astype(
            astype(equal(x, array(2.0f), stream), bfloat16, stream),
            float32,
            stream),
        {0.0f, 1.0f, 1.0f, 0.0f},
        stream,
        8e-3);
  }
}

TEST_CASE("CumSum scans suffix rows against host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  array x(xv.begin(), Shape{5}, float32);
  // Exclusive scan, the form categorical_inverse_cdf uses.
  check_values(cumsum(x, 0, false, false, stream), {0, 1, 3, 6, 10}, stream);
  // Inclusive scan.
  check_values(cumsum(x, 0, false, true, stream), {1, 3, 6, 10, 15}, stream);

  // A caller-composed output may arrive with existing storage that shares a
  // parent buffer (the composed-SDPA pattern). Scan::eval_gpu must keep that
  // storage: reallocating detaches the view and the scan lands in scratch
  // while the parent stays zero-filled.
  array parent = full({5}, 0.0f, stream);
  eval(parent);
  omarchy::get_command_encoder(stream).synchronize();
  x.eval();
  omarchy::get_command_encoder(stream).synchronize();
  array pre_out = array::unsafe_weak_copy(parent);
  pre_out.primitive_ptr() =
      std::make_shared<Scan>(stream, Scan::Sum, 0, false, true);
  std::vector<array> scan_inputs{x};
  std::vector<array> scan_outputs{pre_out};
  pre_out.primitive().eval_gpu(scan_inputs, scan_outputs);
  omarchy::get_command_encoder(stream).synchronize();
  const float* parent_data = parent.data<float>();
  const float expected_inclusive[5] = {1.0f, 3.0f, 6.0f, 10.0f, 15.0f};
  for (int index = 0; index < 5; ++index) {
    CHECK(parent_data[index] ==
          doctest::Approx(expected_inclusive[index]).epsilon(1e-5));
  }

  // A row longer than one workgroup: one invocation owns the whole row.
  std::vector<float> lv(5000);
  float running = 0.0f;
  for (size_t index = 0; index < lv.size(); ++index) {
    lv[index] = static_cast<float>(index % 7);
    running += lv[index];
  }
  std::vector<float> lexclusive(lv.size());
  running = 0.0f;
  for (size_t index = 0; index < lv.size(); ++index) {
    lexclusive[index] = running;
    running += lv[index];
  }
  array long_row(lv.begin(), Shape{5000}, float32);
  check_values(cumsum(long_row, 0, false, false, stream), lexclusive, stream);

  // Multiple rows scan independently.
  std::vector<float> mv = {1.0f, 10.0f, 100.0f, 2.0f, 20.0f, 200.0f};
  array rows(mv.begin(), Shape{2, 3}, float32);
  check_values(
      cumsum(rows, -1, false, false, stream),
      {0.0f, 1.0f, 11.0f, 0.0f, 2.0f, 22.0f},
      stream);

  // Reverse scans and leading-axis scans now compute through the general
  // kernel.
  check_values(
      cumsum(x, 0, true, false, stream),
      {14.0f, 12.0f, 9.0f, 5.0f, 0.0f},
      stream);
  check_values(
      cumsum(rows, 0, false, true, stream),
      {1.0f, 10.0f, 100.0f, 3.0f, 30.0f, 300.0f},
      stream);
}

TEST_CASE("searchsorted matches the upstream binary search on both sides") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> sv = {1.0f, 3.0f, 3.0f, 5.0f, 7.0f, 9.0f};
  array sorted(sv.begin(), Shape{6}, float32);

  // Duplicates make left and right differ, so both sides are pinned.
  std::vector<float> vv = {1.5f, 1.0f, 3.0f, 4.0f, 9.0f, 10.0f};
  array values(vv.begin(), Shape{6}, float32);
  array left = searchsorted(sorted, values, "left", stream);
  check_indices(left, {1, 0, 1, 3, 5, 6}, stream);
  array right = searchsorted(sorted, values, "right", stream);
  check_indices(right, {1, 1, 3, 3, 6, 6}, stream);

  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (Dtype dtype : {float32, float16, bfloat16}) {
    auto ordered = astype(
        array({-inf, -0.0f, 0.0f, 1.0f, inf, nan, nan}), dtype, stream);
    auto queries = astype(array({-inf, 0.0f, inf, nan}), dtype, stream);
    check_indices(
        searchsorted(ordered, queries, "left", stream), {0, 1, 4, 5}, stream);
    check_indices(
        searchsorted(ordered, queries, "right", stream), {1, 3, 5, 7}, stream);
  }

  // uint32 indices minus one, the exact epilogue random.cpp composes.
  check_uint32_values(
      subtract(right, array(1u, uint32), stream),
      {0, 0, 2, 2, 5, 5},
      stream);
  check_int32_values(
      subtract(array(5, int32), array(7, int32), stream),
      {-2},
      stream);
}

TEST_CASE("categorical samples in range with a pinned key over 32 classes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Near-flat logits in a permuted order: every class holds a few
  // percent of the mass, so 200 draws land in every class with slack.
  std::vector<float> lv(32);
  for (size_t index = 0; index < lv.size(); ++index) {
    lv[index] = (static_cast<float>((index * 7) % 32) - 15.5f) * 0.02f;
  }
  array logits(lv.begin(), Shape{1, 32}, float32);
  array key = random::key(0xc0ffeeu);
  constexpr int draws = 200;
  array samples =
      random::categorical(logits, -1, Shape{draws}, key, stream);
  samples.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(samples.size(), static_cast<size_t>(draws));
  CHECK_EQ(samples.dtype(), uint32);
  std::vector<size_t> counts(32, 0);
  for (size_t index = 0; index < samples.size(); ++index) {
    uint32_t drawn = samples.data<uint32_t>()[index];
    REQUIRE(drawn < 32u);
    counts[drawn]++;
  }
  // The strongest logit must win some draws and weak classes some too,
  // so the samples are varied, not a constant.
  CHECK(counts[0] > 0);
  CHECK(counts[31] > 0);
  CHECK(std::any_of(counts.begin(), counts.end(), [](size_t c) {
    return c > 1;
  }));

  // Same key, same draws: deterministic.
  array repeat = random::categorical(logits, -1, Shape{draws}, key, stream);
  repeat.eval();
  omarchy::get_command_encoder(stream).synchronize();
  for (size_t index = 0; index < samples.size(); ++index) {
    CHECK_EQ(repeat.data<uint32_t>()[index], samples.data<uint32_t>()[index]);
  }

  // Mirror the inverse-CDF graph with public ops, pull the intermediates,
  // and finish on the host: every draw must match bit for bit.
  array w = where(
      isinf(max(logits, stream), stream),
      astype(equal(logits, max(logits, stream), stream), float32, stream),
      exp(subtract(logits, max(logits, stream), stream), stream),
      stream);
  array cdf = cumsum(w, -1, false, false, stream);
  array u = multiply(
      random::uniform(Shape{draws}, float32, key, stream),
      sum(w, stream),
      stream);
  cdf.eval();
  u.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* cdf_host = cdf.data<float>();
  const float* u_host = u.data<float>();
  for (int draw = 0; draw < draws; ++draw) {
    uint32_t low = 0;
    uint32_t high = 32;
    while (low < high) {
      uint32_t mid = low + (high - low) / 2;
      if (!(u_host[draw] < cdf_host[mid])) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    uint32_t expected = low == 0 ? 0 : low - 1;
    CHECK_EQ(samples.data<uint32_t>()[draw], expected);
  }
}

TEST_CASE("temp sampling chain runs the vocab-wide categorical on device") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The mlx-lm decode shape: one vocab-wide logprob row in bf16, scaled
  // by 1/temp exactly as categorical_sampling does, then sampled.
  constexpr int vocab = 151936;
  std::vector<float> hv(vocab);
  std::mt19937 host_rng(1234);
  std::uniform_real_distribution<float> host_uniform(-20.0f, -0.1f);
  for (size_t index = 0; index < hv.size(); ++index) {
    hv[index] = host_uniform(host_rng);
  }
  array logprobs(hv.begin(), Shape{1, vocab}, bfloat16);
  array key = random::key(0x5eed1234u);
  array token = random::categorical(
      multiply(logprobs, array(1.0f / 0.9f), stream), -1, key, stream);
  token.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK(token.data<uint32_t>()[0] < static_cast<uint32_t>(vocab));

  // Deterministic per key and still in range for other keys.
  array repeat = random::categorical(
      multiply(logprobs, array(1.0f / 0.9f), stream), -1, key, stream);
  repeat.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(repeat.data<uint32_t>()[0], token.data<uint32_t>()[0]);
  array other = random::categorical(
      multiply(logprobs, array(1.0f / 0.9f), stream),
      -1,
      random::key(0xabcdefu),
      stream);
  other.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK(other.data<uint32_t>()[0] < static_cast<uint32_t>(vocab));
}


// Reference transcribed from pinned mlx/backend/metal/kernels/quantized.h.
// This is not an independent M1 execution oracle or the CPU quantizer contract.
struct HostQuantizedWeights {
  std::vector<uint32_t> words;
  std::vector<float> scales;
  std::vector<float> biases;
};

HostQuantizedWeights host_affine_quantize(
    const std::vector<float>& matrix,
    int rows,
    int cols,
    int group_size,
    int bits) {
  HostQuantizedWeights result;
  int groups = cols / group_size;
  int words_per_row = cols * bits / 32;
  float n_bins = static_cast<float>((1 << bits) - 1);
  result.words.assign(static_cast<size_t>(rows) * words_per_row, 0);
  result.scales.resize(static_cast<size_t>(rows) * groups);
  result.biases.resize(static_cast<size_t>(rows) * groups);
  for (int row = 0; row < rows; ++row) {
    // Every affine packing upstream ships - uint32 words LSB-first for
    // power-of-two bits, and the 8/3, 8/5, 4/3 byte packs for bits
    // 3/5/6 - is one little-endian bitstream: element e occupies bits
    // [e*bits, (e+1)*bits). The uint32 output words hold that byte
    // stream verbatim.
    std::vector<uint8_t> stream(static_cast<size_t>(cols) * bits / 8, 0);
    auto pack_bits = [&](size_t element, uint32_t value) {
      size_t bit_offset = element * static_cast<size_t>(bits);
      for (int b = 0; b < bits; ++b) {
        if (((value >> b) & 1u) != 0u) {
          stream[(bit_offset + b) / 8] |=
              static_cast<uint8_t>(1u << ((bit_offset + b) % 8));
        }
      }
    };
    for (int group = 0; group < groups; ++group) {
      float w_max = 0.0f;
      float w_min = std::numeric_limits<float>::infinity();
      for (int i = 0; i < group_size; ++i) {
        float value = matrix[row * cols + group * group_size + i];
        w_max = std::max(w_max, value);
        w_min = std::min(w_min, value);
      }
      bool min_dominant = std::abs(w_min) > std::abs(w_max);
      float scale = std::max((w_max - w_min) / n_bins, 1e-7f);
      if (!min_dominant) {
        scale = -scale;
      }
      float edge = min_dominant ? w_min : w_max;
      float q0 = std::round(edge / scale);
      if (q0 != 0.0f) {
        scale = edge / q0;
      }
      float bias = (q0 == 0.0f) ? 0.0f : edge;
      result.scales[row * groups + group] = scale;
      result.biases[row * groups + group] = bias;
      for (int i = 0; i < group_size; ++i) {
        float value = matrix[row * cols + group * group_size + i];
        float q =
            std::clamp(std::round((value - bias) / scale), 0.0f, n_bins);
        pack_bits(group * group_size + i, static_cast<uint32_t>(q));
      }
    }
    for (int word = 0; word < words_per_row; ++word) {
      const uint8_t* bytes = stream.data() + word * 4;
      result.words[row * words_per_row + word] =
          static_cast<uint32_t>(bytes[0]) |
          (static_cast<uint32_t>(bytes[1]) << 8) |
          (static_cast<uint32_t>(bytes[2]) << 16) |
          (static_cast<uint32_t>(bytes[3]) << 24);
    }
  }
  return result;
}

// Host dequant dot in double precision: the truth the device dot must
// reproduce from the same packed words, scales, and biases. The packed
// words are one little-endian bitstream: element e of a row sits at bit
// e*bits, whatever the packing granularity upstream chose.
std::vector<float> host_quantized_matmul(
    const HostQuantizedWeights& w,
    const std::vector<float>& x,
    int m,
    int n,
    int k,
    int group_size,
    int bits) {
  int words_per_row = k * bits / 32;
  int groups = k / group_size;
  auto unpack = [&](int column, int inner) {
    size_t bit = static_cast<size_t>(column) * words_per_row * 32 + inner * bits;
    uint32_t code = 0;
    for (int b = 0; b < bits; ++b) {
      if (((w.words[bit / 32] >> (bit % 32)) & 1u) != 0u) {
        code |= 1u << b;
      }
      ++bit;
    }
    return code;
  };
  std::vector<float> out(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      double acc = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        uint32_t code = unpack(column, inner);
        double dequant = static_cast<double>(code) *
                w.scales[column * groups + inner / group_size] +
            w.biases[column * groups + inner / group_size];
        acc += static_cast<double>(x[row * k + inner]) * dequant;
      }
      out[row * n + column] = static_cast<float>(acc);
    }
  }
  return out;
}

std::vector<float> readback_f32(const Stream& stream, array value) {
  value = astype(value, float32, stream);
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = value.data<float>();
  return std::vector<float>(data, data + value.size());
}

TEST_CASE("quantized matmul matches dequant and host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 gen(7);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  auto run_case = [&](int m, int n, int k, int group_size, int bits) {
    CAPTURE(m);
    CAPTURE(n);
    CAPTURE(k);
    CAPTURE(group_size);
    CAPTURE(bits);
    int groups = k / group_size;
    int words_per_row = k * bits / 32;
    std::vector<float> w_values(static_cast<size_t>(n) * k);
    std::vector<float> x_values(static_cast<size_t>(m) * k);
    for (auto& value : w_values) {
      value = dist(gen);
    }
    for (auto& value : x_values) {
      value = dist(gen);
    }
    HostQuantizedWeights host =
        host_affine_quantize(w_values, n, k, group_size, bits);

    // Reference (a): the dequantized dense matmul on the same device.
    // Independent kernel, identical dequant values.
    std::vector<float> dense_values;
    dense_values.reserve(w_values.size());
    auto unpack = [&](int column, int inner) {
      size_t bit =
          static_cast<size_t>(column) * words_per_row * 32 + inner * bits;
      uint32_t code = 0;
      for (int b = 0; b < bits; ++b) {
        if (((host.words[bit / 32] >> (bit % 32)) & 1u) != 0u) {
          code |= 1u << b;
        }
        ++bit;
      }
      return code;
    };
    for (int column = 0; column < n; ++column) {
      for (int inner = 0; inner < k; ++inner) {
        uint32_t code = unpack(column, inner);
        dense_values.push_back(
            static_cast<float>(code) *
                host.scales[column * groups + inner / group_size] +
            host.biases[column * groups + inner / group_size]);
      }
    }
    array w_words(
        host.words.begin(), Shape{n, words_per_row}, uint32);
    array w_scales(
        host.scales.begin(), Shape{n, groups}, float32);
    array w_biases(
        host.biases.begin(), Shape{n, groups}, float32);
    array x(x_values.begin(), Shape{m, k}, float32);
    array w_dense(dense_values.begin(), Shape{n, k}, float32);
    array dense_out = matmul(x, transpose(w_dense), stream);

    // Reference (b): the double-precision host dot over the same words.
    std::vector<float> expected =
        host_quantized_matmul(host, x_values, m, n, k, group_size, bits);

    array out = quantized_matmul(
        x,
        w_words,
        w_scales,
        w_biases,
        /*transpose=*/true,
        group_size,
        bits,
        "affine",
        stream);
    std::string blocked = evaluation_error(out);
    REQUIRE(blocked.empty());
    std::vector<float> device_values = readback_f32(stream, out);
    REQUIRE_EQ(device_values.size(), expected.size());
    std::vector<float> dense_out_values = readback_f32(stream, dense_out);
    REQUIRE_EQ(dense_out_values.size(), expected.size());
    for (size_t index = 0; index < expected.size(); ++index) {
      CHECK(
          device_values[index] ==
          doctest::Approx(expected[index]).epsilon(2e-4));
      CHECK(
          device_values[index] ==
          doctest::Approx(dense_out_values[index]).epsilon(2e-4));
    }
  };

  // Multiple groups per row, N off the 16-wide tile edge, and the two
  // decode shapes M=1 (one row) and M=7 (a short prefill).
  for (int bits : {2, 3, 4, 5, 6, 8}) {
    for (int group_size : {32, 64}) {
      run_case(1, 37, 128, group_size, bits);
      run_case(7, 37, 128, group_size, bits);
    }
  }
  // An exact-tile N and a K whose group count never lands on the
  // smaller group size word boundary.
  run_case(7, 16, 192, 32, 4);
  run_case(1, 16, 192, 64, 8);
}

TEST_CASE("quantized matmul runs f16 and bf16 activations") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }
  constexpr int m = 7;
  constexpr int n = 20;
  constexpr int k = 128;
  constexpr int group_size = 64;
  constexpr int bits = 4;
  constexpr int groups = k / group_size;
  constexpr int words_per_row = k / 8;
  std::mt19937 gen(11);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> w_values(static_cast<size_t>(n) * k);
  std::vector<float> x_values(static_cast<size_t>(m) * k);
  for (auto& value : w_values) {
    value = dist(gen);
  }
  for (auto& value : x_values) {
    value = dist(gen);
  }
  HostQuantizedWeights host =
      host_affine_quantize(w_values, n, k, group_size, bits);

  // Round x and the group parameters through the 16-bit dtype on the
  // device, then read the exact 16-bit values back so the host
  // reference sees what the kernel sees.
  auto round_trip = [&](const std::vector<float>& values, Dtype dtype) {
    array device(
        values.begin(),
        Shape{static_cast<int>(values.size())},
        float32);
    return readback_f32(
        stream, astype(astype(device, dtype, stream), float32, stream));
  };

  for (Dtype dtype : {float16, bfloat16}) {
    if (dtype == bfloat16 && !capabilities.shader_int16) {
      skip("Vulkan device lacks required BF16 storage features.");
      continue;
    }
    CAPTURE(dtype);
    std::vector<float> x_rounded = round_trip(x_values, dtype);
    std::vector<float> scales_rounded = round_trip(host.scales, dtype);
    std::vector<float> biases_rounded = round_trip(host.biases, dtype);
    HostQuantizedWeights rounded_host = host;
    rounded_host.scales = scales_rounded;
    rounded_host.biases = biases_rounded;
    std::vector<float> expected = host_quantized_matmul(
        rounded_host, x_rounded, m, n, k, group_size, bits);

    array w_words(
        host.words.begin(), Shape{n, words_per_row}, uint32);
    array w_scales(
        scales_rounded.begin(), Shape{n, groups}, float32);
    array w_biases(
        biases_rounded.begin(), Shape{n, groups}, float32);
    array x(x_rounded.begin(), Shape{m, k}, float32);
    array out = quantized_matmul(
        astype(x, dtype, stream),
        w_words,
        astype(w_scales, dtype, stream),
        astype(w_biases, dtype, stream),
        /*transpose=*/true,
        group_size,
        bits,
        "affine",
        stream);
    std::string blocked = evaluation_error(out);
    REQUIRE(blocked.empty());
    std::vector<float> device_values = readback_f32(stream, out);
    REQUIRE_EQ(device_values.size(), expected.size());
    double epsilon = (dtype == float16) ? 4e-3 : 2e-2;
    for (size_t index = 0; index < expected.size(); ++index) {
      CHECK(
          device_values[index] ==
          doctest::Approx(expected[index]).epsilon(epsilon));
    }
  }
}

TEST_CASE("quantized matmul binds affine streams at storage offsets") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }
  constexpr int n = 20;
  constexpr int k = 128;
  constexpr int group_size = 64;
  constexpr int bits = 4;
  constexpr int groups = k / group_size;
  constexpr int words_per_row = k / 8;
  for (int m : {1, 7}) {
    CAPTURE(m);
    std::mt19937 gen(13);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> w_values(static_cast<size_t>(n) * k);
    std::vector<float> x_values(static_cast<size_t>(m) * k);
    for (auto& value : w_values) {
      value = dist(gen);
    }
    for (auto& value : x_values) {
      value = dist(gen);
    }
    HostQuantizedWeights host =
        host_affine_quantize(w_values, n, k, group_size, bits);

    auto round_trip = [&](const std::vector<float>& values, Dtype dtype) {
      array device(
          values.begin(),
          Shape{static_cast<int>(values.size())},
          float32);
      return readback_f32(
          stream, astype(astype(device, dtype, stream), float32, stream));
    };
    std::vector<float> x_rounded = round_trip(x_values, float16);
    std::vector<float> scales_rounded = round_trip(host.scales, float16);
    std::vector<float> biases_rounded = round_trip(host.biases, float16);
    // The contract under test is storage offset invariance: the same
    // logical scale and bias streams, bound once as slices of padded
    // f16 storages at item bases 1 and 3 and once as fresh zero-offset
    // arrays, must drive the identical dispatch to bit-identical
    // output. Absolute values stay pinned by the host-oracle cases
    // above; this case only compares the two bindings, bitwise. A
    // mis-composed scale or bias base moves every output column by
    // O(1), which bitwise equality refuses. History: this case used to
    // compare the decode leg against a double-precision oracle at a
    // 4e-3 epsilon and failed as 0.416748 vs 0.409468 on the M1 - the
    // f16 quad-sum arithmetic of the native qmv route, not an offset
    // defect (receipts/2026-09-12-q4-gemv-offset-oracle).

    std::vector<float> scales_pad(1 + n * groups, 0.0f);
    std::vector<float> biases_pad(3 + n * groups, 0.0f);
    std::copy(
        scales_rounded.begin(), scales_rounded.end(), scales_pad.begin() + 1);
    std::copy(
        biases_rounded.begin(), biases_rounded.end(), biases_pad.begin() + 3);
    array w_words(host.words.begin(), Shape{n, words_per_row}, uint32);
    array x(x_rounded.begin(), Shape{m, k}, float32);
    array scales_padded(
        scales_pad.begin(),
        Shape{static_cast<int>(scales_pad.size())},
        float32);
    array biases_padded(
        biases_pad.begin(),
        Shape{static_cast<int>(biases_pad.size())},
        float32);
    array scales_view = reshape(
        slice(
            astype(scales_padded, float16, stream),
            {1},
            {1 + n * groups},
            {1},
            stream),
        {n, groups},
        stream);
    array biases_view = reshape(
        slice(
            astype(biases_padded, float16, stream),
            {3},
            {3 + n * groups},
            {1},
            stream),
        {n, groups},
        stream);
    array out = quantized_matmul(
        astype(x, float16, stream),
        w_words,
        scales_view,
        biases_view,
        /*transpose=*/true,
        group_size,
        bits,
        "affine",
        stream);
    std::string blocked = evaluation_error(out);
    REQUIRE(blocked.empty());
    std::vector<float> device_values = readback_f32(stream, out);
    array scales_zero(scales_rounded.begin(), Shape{n, groups}, float32);
    array biases_zero(biases_rounded.begin(), Shape{n, groups}, float32);
    array out_zero = quantized_matmul(
        astype(x, float16, stream),
        w_words,
        astype(scales_zero, float16, stream),
        astype(biases_zero, float16, stream),
        /*transpose=*/true,
        group_size,
        bits,
        "affine",
        stream);
    std::string blocked_zero = evaluation_error(out_zero);
    REQUIRE(blocked_zero.empty());
    std::vector<float> zero_values = readback_f32(stream, out_zero);
    REQUIRE_EQ(device_values.size(), zero_values.size());
    // The two runs are the identical dispatch on identical logical
    // operands, so the outputs must match bit for bit; any tolerance
    // here would reintroduce the oracle route this case once
    // miscalibrated.
    for (size_t index = 0; index < device_values.size(); ++index) {
      CHECK(device_values[index] == zero_values[index]);
    }
  }
}

TEST_CASE("quantized matmul pins named errors outside the linear shape") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> x_values(2 * 128, 0.5f);
  array x(x_values.begin(), Shape{2, 128}, float32);
  std::vector<uint32_t> words(4 * 16, 0x33221100u);
  std::vector<float> group_params(4 * 4, 0.03125f);
  array w4(words.begin(), Shape{4, 16}, uint32);
  array sb4(group_params.begin(), Shape{4, 4}, float32);
  array w2(words.begin(), Shape{4, 8}, uint32);
  array sb1(group_params.begin(), Shape{4, 1}, float32);
  std::vector<uint8_t> u8_params(4 * 4, 100);
  array sb_u8(u8_params.begin(), Shape{4, 4}, uint8);

  // The op-level shape math rejects fp-mode bits/group mismatches
  // before the backend is reached, so the surviving backend-level bits
  // refusal is the affine one: bits outside {2, 3, 4, 5, 6, 8}.
  std::vector<uint32_t> words7(4 * 28, 0x33221100u);
  array w7(words7.begin(), Shape{4, 28}, uint32);
  std::string mode_error = evaluation_error(
      quantized_matmul(x, w7, sb4, sb4, true, 32, 7, "affine", stream));
  CHECK(mode_error.find("QuantizedMatmul bits") != std::string::npos);

  // Every x element is 0.5 and the pinned word 0x33221100 holds the
  // LSB-first codes 0,0,1,1,2,2,3,3 with scale and bias 0.03125, so a
  // transposed K=128 dot product is 0.5 * (192 + 128) * 0.03125 = 5.
  // Group size 128 computes now (e286c0b).
  check_values(
      quantized_matmul(x, w4, sb1, sb1, true, 128, 4, "affine", stream),
      std::vector<float>(2 * 4, 5.0f),
      stream,
      1e-5);

  // transpose=false quantizes w as [K, N]: words [128, 4], params [128, 1].
  // Output column n reads code n % 8 from every k, so the row is
  // 0.5 * 128 * 0.03125 * (code + 1) = 2 * (code + 1).
  std::vector<uint32_t> nt_words(128 * 4, 0x33221100u);
  std::vector<float> nt_params(128, 0.03125f);
  array w_nt(nt_words.begin(), Shape{128, 4}, uint32);
  array sb_nt(nt_params.begin(), Shape{128, 1}, float32);
  std::vector<float> nt_expected;
  for (int row = 0; row < 2; ++row) {
    for (int n = 0; n < 32; ++n) {
      const int code = (n % 8) / 2;
      nt_expected.push_back(2.0f * static_cast<float>(code + 1));
    }
  }
  check_values(
      quantized_matmul(x, w_nt, sb_nt, sb_nt, false, 32, 4, "affine", stream),
      nt_expected,
      stream,
      1e-5);

  // Batched rank-3 weights pair with batched x.
  std::vector<float> xb_values(2 * 2 * 128, 0.5f);
  array xb(xb_values.begin(), Shape{2, 2, 128}, float32);
  std::vector<uint32_t> batched_words(2 * 4 * 16, 0x33221100u);
  std::vector<float> batched_params(2 * 4 * 4, 0.03125f);
  array wb(batched_words.begin(), Shape{2, 4, 16}, uint32);
  array sbb(batched_params.begin(), Shape{2, 4, 4}, float32);
  check_values(
      quantized_matmul(xb, wb, sbb, sbb, true, 32, 4, "affine", stream),
      std::vector<float>(2 * 2 * 4, 5.0f),
      stream,
      1e-5);

  // A transposed x view is not row-contiguous; the consumer-boundary
  // normalization (3b30130) materializes it, so pin the result against
  // the host reference instead of a named error. Every x_view element
  // is 0.5 and every group reads LSB-first codes 0..3 from the pinned
  // word with scale and bias 0.03125, so each output element is
  // sum_j 0.5 * (code_j * 0.03125 + 0.03125) over K=128.
  std::vector<float> wide_values(128 * 4, 0.5f);
  array wide(wide_values.begin(), Shape{128, 4}, float32);
  array x_view = transpose(wide);
  array qmm_view = quantized_matmul(
      x_view, w4, sb4, sb4, true, 32, 4, "affine", stream);
  const Shape qmm_view_shape({4, 4});
  CHECK_EQ(qmm_view.shape(), qmm_view_shape);
  qmm_view.eval();
  omarchy::get_command_encoder(stream).synchronize();
  double expected_dot = 0.0;
  for (int column = 0; column < 128; ++column) {
    uint32_t code = (0x33221100u >> ((column % 8) * 4)) & 0xFu;
    expected_dot += 0.5 * (static_cast<double>(code) * 0.03125 + 0.03125);
  }
  const float* qmm_values = qmm_view.data<float>();
  for (size_t index = 0; index < qmm_view.size(); ++index) {
    CHECK(qmm_values[index] == doctest::Approx(expected_dot).epsilon(1e-6));
  }
}

// Host unpack of hand-packed affine words: one little-endian bitstream
// (element e at bit e*bits), one scale and one bias per group. The
// mirror of the upstream affine_dequantize fallback and of the
// dequant.comp packing read for every supported bit width.
std::vector<double> host_affine_dequantize(
    const std::vector<uint32_t>& words,
    const std::vector<float>& scales,
    const std::vector<float>& biases,
    int rows,
    int out_columns,
    int group_size,
    int bits) {
  int words_per_row = out_columns * bits / 32;
  int groups = out_columns / group_size;
  std::vector<double> out(static_cast<size_t>(rows) * out_columns);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < out_columns; ++column) {
      size_t bit =
          (static_cast<size_t>(row) * words_per_row * 32) + column * bits;
      uint32_t code = 0;
      for (int b = 0; b < bits; ++b) {
        if (((words[bit / 32] >> (bit % 32)) & 1u) != 0u) {
          code |= 1u << b;
        }
        ++bit;
      }
      out[static_cast<size_t>(row) * out_columns + column] =
          static_cast<double>(code) *
              scales[row * groups + column / group_size] +
          biases[row * groups + column / group_size];
    }
  }
  return out;
}

TEST_CASE("dequantize reproduces hand-packed affine words") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  const bool f16_ok =
      capabilities.shader_float16 && capabilities.storage_buffer_16bit_access;

  // The QuantizedEmbedding shape family: a gathered [1, 40, words]
  // uint32 block dequantizes to a [1, 40, 896] fp16 embedding row.
  // Words come from the same upstream pack reference the quantized
  // matmul test uses; scales and biases stay dyadic so every q * scale
  // + bias step is exactly representable and the device words must
  // equal the host words bit for bit, independent of FMA contraction
  // on either side.
  constexpr int rows = 40;
  constexpr int columns = 896;
  std::mt19937 gen(13);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> matrix(static_cast<size_t>(rows) * columns);
  for (auto& value : matrix) {
    value = dist(gen);
  }
  std::vector<int> groups_list;
  for (int bits : {2, 3, 4, 5, 6, 8}) {
    for (int group_size : {32, 64, 128}) {
      CAPTURE(bits);
      CAPTURE(group_size);
      HostQuantizedWeights host =
          host_affine_quantize(matrix, rows, columns, group_size, bits);
      int words_per_row = columns * bits / 32;
      int groups = columns / group_size;

      // Dyadic group parameters: k * 2^-5 scales and k * 2^-6 biases
      // with tiny integer k keep every product and sum exact in f32
      // and in f16. The largest mantissa, 2 * 255 * 3 + 2 = 1532 for
      // 8-bit codes, stays inside the 11-bit f16 significand, so the
      // exactness claim holds for the f16 output too.
      std::vector<float> scales(static_cast<size_t>(rows) * groups);
      std::vector<float> biases(static_cast<size_t>(rows) * groups);
      for (int row = 0; row < rows; ++row) {
        for (int group = 0; group < groups; ++group) {
          size_t index = static_cast<size_t>(row) * groups + group;
          scales[index] = 0.03125f * (1.0f + static_cast<float>(index % 3));
          biases[index] =
              0.015625f * (static_cast<float>(index % 5) - 2.0f);
        }
      }
      std::vector<double> expected = host_affine_dequantize(
          host.words, scales, biases, rows, columns, group_size, bits);

      for (Dtype dtype : {float32, float16}) {
        if (dtype == float16 && !f16_ok) {
          skip("Vulkan device lacks required FP16 shader and storage "
               "features.");
          continue;
        }
        CAPTURE(dtype);
        array words(
            host.words.begin(),
            Shape{1, rows, words_per_row},
            uint32);
        array scales_f32(scales.begin(), Shape{1, rows, groups}, float32);
        array biases_f32(biases.begin(), Shape{1, rows, groups}, float32);
        array out = dequantize(
            words,
            astype(scales_f32, dtype, stream),
            astype(biases_f32, dtype, stream),
            group_size,
            bits,
            "affine",
            std::nullopt,
            std::nullopt,
            stream);
        std::string blocked = evaluation_error(out);
        REQUIRE(blocked.empty());
        CHECK_EQ(out.dtype(), dtype);
        CHECK_EQ(out.shape(0), 1);
        CHECK_EQ(out.shape(1), rows);
        CHECK_EQ(out.shape(2), columns);
        std::vector<float> device_values = readback_f32(stream, out);
        REQUIRE_EQ(device_values.size(), expected.size());
        for (size_t index = 0; index < expected.size(); ++index) {
          // Exact: the true value is dyadic, so the device value and
          // the correctly rounded host value must be the same float.
          CHECK_EQ(
              device_values[index],
              static_cast<float>(expected[index]));
        }
      }

      // Round trip with the quantizer's own group parameters: the
      // decode of the packed words must land on the host unpack within
      // f32 rounding noise (FMA contraction differs by at most one
      // ulp of the result).
      std::vector<double> round_expected = host_affine_dequantize(
          host.words,
          host.scales,
          host.biases,
          rows,
          columns,
          group_size,
          bits);
      array words(
          host.words.begin(), Shape{1, rows, words_per_row}, uint32);
      array scales_f32(host.scales.begin(), Shape{1, rows, groups}, float32);
      array biases_f32(host.biases.begin(), Shape{1, rows, groups}, float32);
      array round_out =
          dequantize(
              words,
              scales_f32,
              biases_f32,
              group_size,
              bits,
              "affine",
              std::nullopt,
              std::nullopt,
              stream);
      std::string round_blocked = evaluation_error(round_out);
      REQUIRE(round_blocked.empty());
      std::vector<float> round_values = readback_f32(stream, round_out);
      REQUIRE_EQ(round_values.size(), round_expected.size());
      for (size_t index = 0; index < round_expected.size(); ++index) {
        double tolerance =
            1e-6 + 2e-7 * std::abs(round_expected[index]);
        CHECK(std::abs(round_values[index] - round_expected[index]) <=
              tolerance);
      }
    }
  }
}

TEST_CASE("quantize matches the pinned Metal-source affine reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  const bool f16_ok =
      capabilities.shader_float16 && capabilities.storage_buffer_16bit_access;
  constexpr int rows = 6;
  constexpr int columns = 128;
  std::mt19937 gen(29);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

  for (int bits : {2, 3, 4, 5, 6, 8}) {
    for (int group_size : {32, 64, 128}) {
      CAPTURE(bits);
      CAPTURE(group_size);
      std::vector<float> matrix(static_cast<size_t>(rows) * columns);
      for (auto& value : matrix) {
        value = dist(gen);
      }
      float high = static_cast<float>(1 << (bits - 1));
      matrix[0] = high - static_cast<float>((1 << bits) - 1);
      matrix[1] = high;
      matrix[2] = high - 0.5f;
      std::fill(
          matrix.begin() + columns - group_size,
          matrix.begin() + columns,
          0.0f);

      std::fill(matrix.begin() + columns, matrix.begin() + 2 * columns, -1.0f);
      std::fill(matrix.begin() + 2 * columns, matrix.begin() + 3 * columns, -2.0f);
      std::fill(
          matrix.begin() + 2 * columns + group_size / 2,
          matrix.begin() + 2 * columns + group_size,
          -1.0f);
      for (Dtype dtype : {float32, float16}) {
        if (dtype == float16 && !f16_ok) {
          skip("Vulkan device lacks required FP16 shader and storage features.");
          continue;
        }
        CAPTURE(dtype);
        array source(
            matrix.begin(), Shape{2, 3, columns}, float32);
        array input = dtype == float32 ? source : astype(source, dtype, stream);
        std::vector<float> input_values =
            dtype == float32 ? matrix : readback_f32(stream, input);
        HostQuantizedWeights expected = host_affine_quantize(
            input_values, rows, columns, group_size, bits);

        auto outputs = quantize(
            input, group_size, bits, "affine", std::nullopt, stream);
        REQUIRE_EQ(outputs.size(), 3);
        uint64_t compute_before =
            omarchy::trace::counters().vk_compute_dispatches.load();
        outputs[0].eval();
        omarchy::get_command_encoder(stream).synchronize();
        CHECK(
            omarchy::trace::counters().vk_compute_dispatches.load() >
            compute_before);
        REQUIRE_EQ(outputs[0].dtype(), uint32);
        REQUIRE_EQ(outputs[1].dtype(), dtype);
        REQUIRE_EQ(outputs[2].dtype(), dtype);
        CHECK_EQ(outputs[0].shape(), Shape{2, 3, columns * bits / 32});
        CHECK_EQ(outputs[1].shape(), Shape{2, 3, columns / group_size});
        CHECK_EQ(outputs[2].shape(), outputs[1].shape());

        const uint32_t* words = outputs[0].data<uint32_t>();
        for (size_t index = 0; index < expected.words.size(); ++index) {
          CHECK_EQ(words[index], expected.words[index]);
        }
        std::vector<float> scales = readback_f32(stream, outputs[1]);
        std::vector<float> biases = readback_f32(stream, outputs[2]);
        REQUIRE_EQ(scales.size(), expected.scales.size());
        REQUIRE_EQ(biases.size(), expected.biases.size());
        const double parameter_tolerance = dtype == float32 ? 1e-6 : 1e-3;
        for (size_t index = 0; index < scales.size(); ++index) {
          CHECK(std::abs(scales[index] - expected.scales[index]) <=
                parameter_tolerance);
          CHECK(std::abs(biases[index] - expected.biases[index]) <=
                parameter_tolerance);
        }

        array reconstructed = dequantize(
            outputs[0],
            outputs[1],
            outputs[2],
            group_size,
            bits,
            "affine",
            std::nullopt,
            std::nullopt,
            stream);
        std::vector<float> dequantized = readback_f32(stream, reconstructed);
        REQUIRE_EQ(dequantized.size(), input_values.size());
        const int groups = columns / group_size;
        const float reconstruction_tolerance =
            dtype == float32 ? 1e-6f : 1e-3f;
        for (size_t index = 0; index < dequantized.size(); ++index) {
          size_t group = (index / columns) * groups +
              (index % columns) / group_size;
          CHECK(std::abs(dequantized[index] - input_values[index]) <=
                std::abs(scales[group]) + reconstruction_tolerance);
        }
      }
    }
  }
}


TEST_CASE("quantize and dequantize pin named errors outside the affine gate") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  auto construction_error = [](auto&& build) {
    try {
      build();
    } catch (const std::exception& error) {
      return std::string(error.what());
    }
    return std::string{};
  };

  std::vector<float> matrix(4 * 128, 0.5f);
  array x(matrix.begin(), Shape{4, 128}, float32);
  // Upstream's op-level mode/bits validation rejects these before the
  // backend is reached; the backend bits gate only guards direct
  // primitive construction.
  CHECK(construction_error([&] {
          quantize(x, 32, 4, "mxfp8", std::nullopt, stream);
        }).find("requires bits to be 8") != std::string::npos);
  CHECK(construction_error([&] {
          quantize(x, 32, 7, "affine", std::nullopt, stream);
        }).find("The requested number of bits 7 is not supported") !=
        std::string::npos);
  // Group sizes outside upstream's 32/64/128 set are rejected by the
  // pinned op-level validation before the backend gate runs.
  std::vector<float> matrix12(4 * 96, 0.5f);
  array x12(matrix12.begin(), Shape{4, 96}, float32);
  CHECK(construction_error([&] {
          quantize(x12, 12, 4, "affine", std::nullopt, stream);
        }).find("The requested group size 12 is not supported") !=
        std::string::npos);

  std::vector<uint32_t> words4(4 * 16, 0x33221100u);
  std::vector<uint8_t> scales_u8(4 * 4, 100);
  array w4(words4.begin(), Shape{4, 16}, uint32);
  array s8(scales_u8.begin(), Shape{4, 4}, uint8);
  // Note: no fp-mode backend refusal is reachable through the public
  // dequantize op - the mode-specific bits/group validation runs at the
  // op level, and float64 activations cannot exist on the GPU.

  std::vector<float> params4(4 * 4, 0.03125f);
  std::vector<float> params8(4 * 8, 0.03125f);
  std::vector<float> params96(4 * 8, 0.03125f);
  array sb96(params96.begin(), Shape{4, 8}, float32);
  std::vector<uint32_t> words12(4 * 12, 0x33221100u);
  array w12(words12.begin(), Shape{4, 12}, uint32);
  array sb8(params8.begin(), Shape{4, 8}, float32);
  // A bits/word-count combination whose packed shape cannot match the
  // scale grid is rejected by the pinned op-level word-shape math.
  CHECK(construction_error([&] {
          dequantize(
              w4,
              sb8,
              sb8,
              32,
              3,
              "affine",
              std::nullopt,
              std::nullopt,
              stream);
        }).find("does not match the matrix") != std::string::npos);
  CHECK(construction_error([&] {
          dequantize(
              w12,
              sb96,
              sb96,
              12,
              4,
              "affine",
              std::nullopt,
              std::nullopt,
              stream);
        }).find("Quantize group size") != std::string::npos);
}

TEST_CASE("fp qmm serves batched weight layouts against hand-packed oracles") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Hand-packed fp-mode weights. The fp4 e2m1 decode the kernels use is
  // codes 0..7 -> {0, 0.5, 1, 1.5, 2, 3, 4, 6} and the fp8 e4m3 byte
  // 0x38 decodes to 1.0; byte b of a packed word holds elements 2b (low
  // nibble) and 2b+1, so 0x11111111 is eight 0.5 elements (word sum 4.0)
  // and 0x38383838 is four 1.0 elements (word sum 4.0). Scales are one
  // byte per group: e8m0 for mxfp4/mxfp8 (0x7F = 1.0, 0x80 = 2.0) and
  // e4m3 for nvfp4 (0x38 = 1.0, 0x40 = 2.0, 0x30 = 0.5). Per-slice
  // scales give every batch slice its own weight result, so a
  // shared-weight bug cannot pass, and x rows carry (row + 1) so a
  // pairing bug that reuses slice zero's activation cannot pass either.
  constexpr uint32_t fp4_word = 0x11111111u;
  constexpr uint32_t fp8_word = 0x38383838u;

  auto run_batched = [&](bool transposed,
                         int bits,
                         int group_size,
                         int k,
                         int n,
                         const std::vector<uint8_t>& scale_codes,
                         const std::vector<float>& scale_values,
                         int x_rows_per_slice,
                         uint32_t word,
                         float element_value,
                         double epsilon) {
    CAPTURE(transposed);
    CAPTURE(bits);
    CAPTURE(group_size);
    CAPTURE(k);
    CAPTURE(n);
    CAPTURE(x_rows_per_slice);
    int batch = static_cast<int>(scale_values.size());
    int groups = k / group_size;
    int packed_outer = k * bits / 32;
    int weight_outer = transposed ? n : k;
    Shape w_shape{batch, weight_outer, transposed ? packed_outer : n * bits / 32};
    std::vector<uint32_t> words(
        static_cast<size_t>(batch) * weight_outer *
            (transposed ? packed_outer : n * bits / 32),
        word);
    std::vector<uint8_t> scale_bytes(
        static_cast<size_t>(batch) * weight_outer * groups);
    for (int b = 0; b < batch; ++b) {
      std::fill(
          scale_bytes.begin() + static_cast<size_t>(b) * weight_outer * groups,
          scale_bytes.begin() + static_cast<size_t>(b + 1) * weight_outer *
              groups,
          scale_codes[b]);
    }
    array w(words.begin(), w_shape, uint32);
    array sb(
        scale_bytes.begin(),
        Shape{batch, weight_outer, groups},
        uint8);

    // One row per slice rides the vec kernel; three rows ride the tile
    // kernel. x value (b + 1) forces the paired dispatch to advance x.
    std::vector<float> x_values(
        static_cast<size_t>(batch) * x_rows_per_slice * k);
    for (int b = 0; b < batch; ++b) {
      std::fill(
          x_values.begin() + static_cast<size_t>(b) * x_rows_per_slice * k,
          x_values.begin() + static_cast<size_t>(b + 1) * x_rows_per_slice * k,
          static_cast<float>(b + 1));
    }
    array x(x_values.begin(), Shape{batch, x_rows_per_slice, k}, float32);

    // Transposed: one output column sums every packed element of its
    // weight row. Non-transposed: column n reads element n % (32/bits)
    // from every k row; the uniform word makes every column agree.
    std::vector<float> expected;
    for (int b = 0; b < batch; ++b) {
      float slice_value = transposed
          ? static_cast<float>(packed_outer) * (element_value * (32 / bits)) *
              scale_values[b]
          : static_cast<float>(k) * element_value * scale_values[b];
      for (int r = 0; r < x_rows_per_slice; ++r) {
        for (int c = 0; c < n; ++c) {
          expected.push_back(slice_value * static_cast<float>(b + 1));
        }
      }
    }
    const char* mode = bits == 8 ? "mxfp8"
        : group_size == 16       ? "nvfp4"
                                 : "mxfp4";
    array y = quantized_matmul(
        x,
        w,
        sb,
        std::nullopt,
        transposed,
        group_size,
        bits,
        mode,
        stream);
    CHECK_EQ(y.shape(), Shape{batch, x_rows_per_slice, n});
    check_values(std::move(y), expected, stream, epsilon);
  };

  // Decode route (one row per slice): mxfp4, mxfp8, non-transposed
  // mxfp4, and nvfp4 with three differently-scaled slices.
  run_batched(
      true, 4, 32, 32, 2, {0x7F, 0x80}, {1.0f, 2.0f}, 1, fp4_word, 0.5f, 1e-6);
  run_batched(
      true, 8, 32, 32, 2, {0x7F, 0x80}, {1.0f, 2.0f}, 1, fp8_word, 1.0f, 1e-6);
  run_batched(
      false, 4, 32, 32, 32, {0x7F, 0x80}, {1.0f, 2.0f}, 1, fp4_word, 0.5f, 1e-6);
  run_batched(
      true,
      4,
      16,
      16,
      2,
      {0x38, 0x40, 0x30},
      {1.0f, 2.0f, 0.5f},
      1,
      fp4_word,
      0.5f,
      1e-6);
  // Prefill tile route (three rows per slice).
  run_batched(
      true, 4, 32, 32, 2, {0x7F, 0x80}, {1.0f, 2.0f}, 3, fp4_word, 0.5f, 1e-6);
  // General-kernel route with the tile engine disabled.
  setenv("MLX_OMARCHY_QMM_TILE", "0", 1);
  run_batched(
      true, 4, 32, 32, 2, {0x7F, 0x80}, {1.0f, 2.0f}, 1, fp4_word, 0.5f, 1e-6);
  unsetenv("MLX_OMARCHY_QMM_TILE");
}

TEST_CASE("affine quantize bfloat16 matches the f32-math bf16-store oracle") {
  if (!compute_available()) {
    return;
  }
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.storage_buffer_16bit_access ||
      !capabilities.shader_int16) {
    skip("Vulkan device lacks required BF16 storage features.");
    return;
  }
  Stream stream = gpu_stream();
  constexpr int rows = 4;
  constexpr int columns = 128;

  // bf16 round-to-nearest-even on the f32 bit pattern, the same
  // rounding STORE_VALUE performs in the USE_BF16 shader variant.
  auto bf16_round = [](float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) {
      return static_cast<uint16_t>((bits >> 16) | 0x40u);
    }
    return static_cast<uint16_t>(
        (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
  };

  // Inputs must be exactly representable in bf16 so the host oracle
  // and the device read identical values; dyadic mantissas guarantee
  // that regardless of the conversion rounding on the way in.
  std::mt19937 gen(31);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
  std::vector<float> matrix(static_cast<size_t>(rows) * columns);
  for (auto& value : matrix) {
    uint32_t bits;
    float raw = dist(gen);
    std::memcpy(&bits, &raw, sizeof(bits));
    bits = (static_cast<uint32_t>(bf16_round(raw)) << 16);
    std::memcpy(&value, &bits, sizeof(value));
  }
  // Force the interesting group shapes: min-dominant, all-equal, and
  // a group whose range is small enough to exercise the 1e-7 floor.
  for (int i = 0; i < 32; ++i) {
    matrix[i] = -static_cast<float>(i) * 0.25f;
    matrix[columns + i] = 1.0f;
  }

  for (auto [bits_int, group_size] : {std::pair<int, int>{4, 32}, {8, 64}}) {
    CAPTURE(bits_int);
    CAPTURE(group_size);
    // The device chain: bf16 quantize -> bf16 dequantize -> bf16 qmm.
    array input(matrix.begin(), Shape{rows, columns}, bfloat16);
    auto outputs = quantize(
        input, group_size, bits_int, "affine", std::nullopt, stream);
    REQUIRE_EQ(outputs.size(), 3);
    REQUIRE_EQ(outputs[0].dtype(), uint32);
    REQUIRE_EQ(outputs[1].dtype(), bfloat16);
    REQUIRE_EQ(outputs[2].dtype(), bfloat16);

    // Words pack from the unrounded f32 parameters, so the words must
    // equal the f32 oracle bit for bit.
    HostQuantizedWeights oracle = host_affine_quantize(
        matrix, rows, columns, group_size, bits_int);
    outputs[0].eval();
    omarchy::get_command_encoder(stream).synchronize();
    const uint32_t* words = outputs[0].data<uint32_t>();
    for (size_t index = 0; index < oracle.words.size(); ++index) {
      CHECK_EQ(words[index], oracle.words[index]);
    }
    // Stored parameters are bf16 roundings of the f32 oracle values.
    std::vector<float> device_scales = readback_f32(stream, outputs[1]);
    std::vector<float> device_biases = readback_f32(stream, outputs[2]);
    REQUIRE_EQ(device_scales.size(), oracle.scales.size());
    for (size_t index = 0; index < oracle.scales.size(); ++index) {
      uint32_t scale_bits = static_cast<uint32_t>(
          bf16_round(oracle.scales[index])) << 16;
      uint32_t bias_bits =
          static_cast<uint32_t>(bf16_round(oracle.biases[index])) << 16;
      float oracle_scale;
      float oracle_bias;
      std::memcpy(&oracle_scale, &scale_bits, sizeof(oracle_scale));
      std::memcpy(&oracle_bias, &bias_bits, sizeof(oracle_bias));
      CHECK_EQ(device_scales[index], oracle_scale);
      CHECK_EQ(device_biases[index], oracle_bias);
    }

    // Dequantize inverts through the stored bf16 parameters.
    array reconstructed = dequantize(
        outputs[0],
        outputs[1],
        outputs[2],
        group_size,
        bits_int,
        "affine",
        std::nullopt,
        std::nullopt,
        stream);
    std::vector<float> dequantized = readback_f32(stream, reconstructed);
    REQUIRE_EQ(dequantized.size(), matrix.size());
    const int groups = columns / group_size;
    for (size_t index = 0; index < dequantized.size(); ++index) {
      size_t group =
          (index / columns) * groups + (index % columns) / group_size;
      // Codes are chosen against the unrounded f32 scale; dequantize
      // reads the bf16 rounding, so the bound is half an f32 step plus
      // the worst bf16 parameter drift over the full code range.
      CHECK(std::abs(dequantized[index] - matrix[index]) <=
            std::abs(oracle.scales[group]) * 1.6f + 1e-3);
    }

    // The bf16 quantized matmul consumes the bf16 parameters: the
    // quantized rows are N (transpose=true), x is one constant row per
    // weight row, and the reference sums the dequantized weights.
    std::vector<float> x_values(static_cast<size_t>(rows) * columns, 0.25f);
    array x(x_values.begin(), Shape{rows, columns}, bfloat16);
    array y = quantized_matmul(
        x,
        outputs[0],
        outputs[1],
        outputs[2],
        true,
        group_size,
        bits_int,
        "affine",
        stream);
    REQUIRE_EQ(y.dtype(), bfloat16);
    REQUIRE_EQ(y.shape(), Shape{rows, rows});
    std::vector<float> y_values = readback_f32(stream, y);
    for (int m = 0; m < rows; ++m) {
      for (int n = 0; n < rows; ++n) {
        float reference = 0.0f;
        for (int kk = 0; kk < columns; ++kk) {
          reference += 0.25f *
              dequantized[static_cast<size_t>(n) * columns + kk];
        }
        CHECK(std::abs(y_values[static_cast<size_t>(m) * rows + n] -
                       reference) <=
              4e-2 * std::abs(reference) + 1e-2);
      }
    }
  }
}

TEST_CASE("rank-1 quantized mat-vec matches the rank-2 oracle") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Same hand-packed construction as the rank-2 non-transposed oracle:
  // word 0x33221100 holds LSB-first codes {0,0,1,1,2,2,3,3} per group
  // of eight elements, scale and bias 0.03125, x = 0.5 over K = 128,
  // so output column n is 0.5 * 128 * 0.03125 * (code + 1) =
  // 2 * (code + 1) with the byte code pattern {0,0,1,1,2,2,3,3}.
  std::vector<uint32_t> words(128 * 16, 0x33221100u);
  std::vector<float> params(128, 0.03125f);
  array w(words.begin(), Shape{128, 16}, uint32);
  array sb(params.begin(), Shape{128, 1}, float32);
  std::vector<float> x_values(128, 0.5f);
  array x(x_values.begin(), Shape{128}, float32);
  array y = quantized_matmul(x, w, sb, sb, false, 128, 4, "affine", stream);
  CHECK_EQ(y.shape(), Shape{128});
  constexpr int codes[8] = {0, 0, 1, 1, 2, 2, 3, 3};
  std::vector<float> expected;
  for (int c = 0; c < 128; ++c) {
    expected.push_back(2.0f * (codes[c % 8] + 1));
  }
  check_values(std::move(y), expected, stream, 1e-6);
}
// Non-affine quantization-mode conversions, pinned to the Metal
// fp4.h / fp8.h helpers bit for bit: the PyTorch round-to-nearest-even
// e4m3 encode with saturation to 448, the round-up e8m0 scale encode,
// the fp16-bit-trick fp4 decode, and the element domain
// w * (global_enc / decoded_scale).
uint8_t host_fp8_e4m3_encode(float f) {
  uint32_t fp8_max = 543u << 21;
  uint32_t denorm_mask = 141u << 23;
  uint32_t f_bits;
  std::memcpy(&f_bits, &f, sizeof(f_bits));
  uint32_t sign = f_bits & 0x80000000u;
  f_bits ^= sign;
  uint8_t bits;
  if (f_bits >= fp8_max) {
    bits = 0x7E;
  } else if (f_bits < (121u << 23)) {
    float biased = f + std::ldexp(1.0f, 141 - 127);
    uint32_t b_bits;
    std::memcpy(&b_bits, &biased, sizeof(b_bits));
    bits = static_cast<uint8_t>(b_bits - denorm_mask);
  } else {
    uint8_t mant_odd = (f_bits >> 20) & 1;
    f_bits += ((7u - 127u) << 23) + 0x7FFFF;
    f_bits += mant_odd;
    bits = static_cast<uint8_t>(f_bits >> 20);
  }
  return bits | static_cast<uint8_t>(sign >> 24);
}

float host_fp8_e4m3_decode(uint8_t b) {
  // The Metal decode rides fp16 bits scaled by 256; the arithmetic form
  // is exact for every code (0x7F/0xFF decode to 480, the same value
  // the fp16 path produces, and never occur in quantized output).
  int e = (b >> 3) & 0xF;
  int m = b & 0x7;
  float v = (e == 0)
      ? (static_cast<float>(m) * 0.125f) * 0.015625f
      : (1.0f + static_cast<float>(m) * 0.125f) *
          std::ldexp(1.0f, e - 7);
  return (b & 0x80) ? -v : v;
}

uint8_t host_e8m0_encode(float x) {
  if (!std::isfinite(x)) {
    return 0xFF;
  }
  if (x <= 0.0f) {
    return 0x00;
  }
  int n = static_cast<int>(std::round(std::log2(x)));
  n = std::clamp(n, -127, 127);
  uint8_t bits = static_cast<uint8_t>(n + 127);
  float decoded = std::ldexp(1.0f, bits == 0 ? -127 : bits - 127);
  if (bits < 0xFE && decoded < x) {
    bits = static_cast<uint8_t>(bits + 1);
  }
  return bits;
}

float host_e8m0_decode(uint8_t bits) {
  return std::ldexp(1.0f, bits == 0 ? -127 : static_cast<int>(bits) - 127);
}

uint8_t host_fp4_e2m1_encode(float x) {
  if (std::isnan(x)) {
    return 0x7;
  }
  uint8_t sign = (std::signbit(x)) ? 0x8 : 0x0;
  float a = std::abs(x);
  uint8_t m;
  if (a > 5.0f) {
    m = 0x7;
  } else if (a >= 3.5f) {
    m = 0x6;
  } else if (a > 2.5f) {
    m = 0x5;
  } else if (a >= 1.75f) {
    m = 0x4;
  } else if (a > 1.25f) {
    m = 0x3;
  } else if (a >= 0.75f) {
    m = 0x2;
  } else if (a > 0.25f) {
    m = 0x1;
  } else {
    m = 0x0;
  }
  return m | sign;
}

float host_fp4_e2m1_decode(uint8_t code) {
  float sign = (code & 0x8) ? -1.0f : 1.0f;
  int e = (code >> 1) & 0x3;
  int m = code & 0x1;
  float v = (e == 0)
      ? (static_cast<float>(m) * 0.5f)
      : ((1.0f + static_cast<float>(m) * 0.5f) * std::ldexp(1.0f, e - 1));
  return sign * v;
}


struct HostFpQuantized {
  std::vector<uint32_t> words;
  std::vector<uint8_t> scale_bytes;
};

// Mirror of the pinned Metal fp_quantize kernel: one group scale byte,
// byte-packed fp4/fp8 element codes in a little-endian uint32 word
// stream.
HostFpQuantized host_fp_quantize(
    const std::vector<float>& matrix,
    int rows,
    int cols,
    int group_size,
    int bits,
    bool has_global_scale = false,
    float global_scale = 1.0f) {
  HostFpQuantized result;
  float maxval = (bits == 8) ? 448.0f : 6.0f;
  float scale_enc =
      has_global_scale ? (448.0f * 6.0f) / global_scale : 1.0f;
  int groups = cols / group_size;
  int words_per_row = cols * bits / 32;
  result.words.assign(static_cast<size_t>(rows) * words_per_row, 0);
  result.scale_bytes.resize(static_cast<size_t>(rows) * groups);
  for (int row = 0; row < rows; ++row) {
    std::vector<uint8_t> stream(static_cast<size_t>(cols) * bits / 8, 0);
    auto pack_bits = [&](size_t bit_offset, uint32_t value) {
      for (int b = 0; b < bits; ++b) {
        if (((value >> b) & 1u) != 0u) {
          stream[(bit_offset + b) / 8] |=
              static_cast<uint8_t>(1u << ((bit_offset + b) % 8));
        }
      }
    };
    for (int group = 0; group < groups; ++group) {
      float amax = 0.0f;
      for (int i = 0; i < group_size; ++i) {
        amax = std::max(
            amax, std::abs(matrix[row * cols + group * group_size + i]));
      }
      float scale_dec = amax / maxval;
      uint8_t q_scale;
      float decoded;
      if (group_size == 16) {
        scale_dec *= scale_enc;
        q_scale = host_fp8_e4m3_encode(scale_dec);
        decoded = host_fp8_e4m3_decode(q_scale);
      } else {
        q_scale = host_e8m0_encode(scale_dec);
        decoded = host_e8m0_decode(q_scale);
      }
      float inv = (decoded == 0.0f) ? 0.0f : scale_enc / decoded;
      for (int i = 0; i < group_size; ++i) {
        float value = matrix[row * cols + group * group_size + i] * inv;
        uint32_t code = (bits == 8)
            ? host_fp8_e4m3_encode(value)
            : host_fp4_e2m1_encode(value);
        size_t bit_offset = static_cast<size_t>(group) * group_size + i;
        for (int b = 0; b < bits; ++b) {
          if (((code >> b) & 1u) != 0u) {
            size_t bit = bit_offset * bits + b;
            stream[bit / 8] |= static_cast<uint8_t>(1u << (bit % 8));
          }
        }
      }
    }
    for (int word = 0; word < words_per_row; ++word) {
      const uint8_t* bytes = stream.data() + word * 4;
      result.words[row * words_per_row + word] =
          static_cast<uint32_t>(bytes[0]) |
          (static_cast<uint32_t>(bytes[1]) << 8) |
          (static_cast<uint32_t>(bytes[2]) << 16) |
          (static_cast<uint32_t>(bytes[3]) << 24);
    }
  }
  return result;
}

// Mirror of the pinned Metal fp_dequantize kernel.
std::vector<float> host_fp_dequantize(
    const HostFpQuantized& w,
    int rows,
    int out_columns,
    int group_size,
    int bits,
    bool has_global_scale = false,
    float global_scale = 1.0f) {
  float inv_enc =
      has_global_scale ? global_scale / (448.0f * 6.0f) : 1.0f;
  int groups = out_columns / group_size;
  int words_per_row = out_columns * bits / 32;
  std::vector<float> out(static_cast<size_t>(rows) * out_columns);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < out_columns; ++column) {
      size_t bit =
          (static_cast<size_t>(row) * words_per_row * 32) + column * bits;
      uint32_t code = 0;
      for (int b = 0; b < bits; ++b) {
        if (((w.words[bit / 32] >> (bit % 32)) & 1u) != 0u) {
          code |= 1u << b;
        }
        ++bit;
      }
      uint8_t scale_byte = w.scale_bytes[row * groups + column / group_size];
      float scale = (group_size == 16)
          ? host_fp8_e4m3_decode(scale_byte)
          : host_e8m0_decode(scale_byte);
      if (has_global_scale) {
        scale *= inv_enc;
      }
      float value = (bits == 8)
          ? host_fp8_e4m3_decode(static_cast<uint8_t>(code))
          : host_fp4_e2m1_decode(static_cast<uint8_t>(code));
      out[static_cast<size_t>(row) * out_columns + column] = scale * value;
    }
  }
  return out;
}

void check_fp_mode(
    const char* mode,
    int group_size,
    int bits,
    bool has_global_scale,
    Stream& stream) {
  CAPTURE(mode);
  CAPTURE(group_size);
  CAPTURE(bits);
  CAPTURE(has_global_scale);
  constexpr int rows = 5;
  constexpr int columns = 128;
  std::mt19937 gen(31);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> matrix(static_cast<size_t>(rows) * columns);
  for (auto& value : matrix) {
    value = dist(gen);
  }
  // Exact LUT magnitudes and an all-zero group pin the encode edges.
  std::vector<float> lut_values(
      {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.5f, -3.0f, -6.0f});
  for (size_t index = 0; index < lut_values.size(); ++index) {
    matrix[index] = lut_values[index];
  }
  for (int column = columns - group_size; column < columns; ++column) {
    matrix[column] = 0.0f;
  }
  float global_scale = 6.0f;
  array w(matrix.begin(), Shape{rows, columns}, float32);

  HostFpQuantized host = host_fp_quantize(
      matrix,
      rows,
      columns,
      group_size,
      bits,
      has_global_scale,
      global_scale);
  std::optional<array> global;
  if (has_global_scale) {
    global = array(global_scale, float32);
  }
  auto outputs = quantize(
      w, group_size, bits, mode, global, stream);
  REQUIRE_EQ(outputs.size(), 2);
  REQUIRE_EQ(outputs[0].dtype(), uint32);
  REQUIRE_EQ(outputs[1].dtype(), uint8);
  outputs[0].eval();
  omarchy::get_command_encoder(stream).synchronize();
  const uint32_t* words = outputs[0].data<uint32_t>();
  REQUIRE_EQ(outputs[0].size(), host.words.size());
  for (size_t index = 0; index < host.words.size(); ++index) {
    CHECK_EQ(words[index], host.words[index]);
  }
  const uint8_t* scale_bytes = outputs[1].data<uint8_t>();
  REQUIRE_EQ(outputs[1].size(), host.scale_bytes.size());
  for (size_t index = 0; index < host.scale_bytes.size(); ++index) {
    CHECK_EQ(scale_bytes[index], host.scale_bytes[index]);
  }

  array round_out = dequantize(
      outputs[0],
      outputs[1],
      std::nullopt,
      group_size,
      bits,
      mode,
      global,
      float32,
      stream);
  std::string round_blocked = evaluation_error(round_out);
  REQUIRE(round_blocked.empty());
  std::vector<float> expected = host_fp_dequantize(
      host,
      rows,
      columns,
      group_size,
      bits,
      has_global_scale,
      global_scale);
  std::vector<float> round_values = readback_f32(stream, round_out);
  REQUIRE_EQ(round_values.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK(round_values[index] == doctest::Approx(expected[index]).epsilon(1e-6));
  }

  // Quantized matmul through both dispatch shapes: M=1 rides the fused
  // GEMV kernel, M=7 the m-tiled prefill kernel. The reference is the
  // host fp dequant dot.
  for (int m : {1, 7}) {
    std::vector<float> x_values(static_cast<size_t>(m) * columns);
    std::mt19937 xgen(41 + m);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    for (auto& value : x_values) {
      value = xdist(gen);
    }
    array x(x_values.begin(), Shape{m, columns}, float32);
    array out = quantized_matmul(
        x,
        outputs[0],
        outputs[1],
        std::nullopt,
        /*transpose=*/true,
        group_size,
        bits,
        mode,
        stream);
    std::string blocked = evaluation_error(out);
    REQUIRE(blocked.empty());
    std::vector<float> device_values = readback_f32(stream, out);
    double tolerance = 1e-4;
    for (int row = 0; row < m; ++row) {
      for (int column = 0; column < rows; ++column) {
        double acc = 0.0;
        for (int inner = 0; inner < columns; ++inner) {
          size_t bit =
              (static_cast<size_t>(column) * (columns * bits / 32) * 32) +
              inner * bits;
          uint32_t code = 0;
          for (int b = 0; b < bits; ++b) {
            if (((host.words[bit / 32] >> (bit % 32)) & 1u) != 0u) {
              code |= 1u << b;
            }
            ++bit;
          }
          uint8_t scale_byte =
              host.scale_bytes[column * (columns / group_size) +
                  inner / group_size];
          float scale = (group_size == 16)
              ? host_fp8_e4m3_decode(scale_byte)
              : host_e8m0_decode(scale_byte);
          if (has_global_scale) {
            scale *= global_scale / (448.0f * 6.0f);
          }
          float value = (bits == 8)
              ? host_fp8_e4m3_decode(static_cast<uint8_t>(code))
              : host_fp4_e2m1_decode(static_cast<uint8_t>(code));
          acc += static_cast<double>(x_values[row * columns + inner]) *
              (scale * value);
        }
        CHECK(
            device_values[row * rows + column] ==
            doctest::Approx(acc).epsilon(tolerance));
      }
    }
  }
}


// Host direct convolution reference copied from the upstream CPU path
// slow_conv_2D for the forward groups==1, flip=false, input_dilation==1
// case: out[n, oh, ow, o] sums in[n, ih, iw, c] * wt[o, wh, ww, c] over
// in-bounds taps, with ih = oh*sh - plo_h + wh*dh and iw likewise.
std::vector<float> host_conv2d_nhwc(
    const std::vector<float>& input,
    Shape in_shape,
    const std::vector<float>& weight,
    Shape wt_shape,
    std::pair<int, int> stride,
    std::pair<int, int> pad_lo,
    std::pair<int, int> pad_hi,
    std::pair<int, int> dilation) {
  int n = in_shape[0], ih = in_shape[1], iw = in_shape[2], c = in_shape[3];
  int o = wt_shape[0], kh = wt_shape[1], kw = wt_shape[2];
  int oh = (ih + pad_lo.first + pad_hi.first - dilation.first * (kh - 1) - 1) /
      stride.first +
      1;
  int ow = (iw + pad_lo.second + pad_hi.second - dilation.second * (kw - 1) -
            1) /
      stride.second +
      1;
  std::vector<float> output(n * oh * ow * o, 0.0f);
  for (int batch = 0; batch < n; ++batch) {
    for (int row = 0; row < oh; ++row) {
      for (int column = 0; column < ow; ++column) {
        for (int out_channel = 0; out_channel < o; ++out_channel) {
          float accumulator = 0.0f;
          for (int ky = 0; ky < kh; ++ky) {
            int in_row = row * stride.first - pad_lo.first + ky * dilation.first;
            if (in_row < 0 || in_row >= ih) {
              continue;
            }
            for (int kx = 0; kx < kw; ++kx) {
              int in_column =
                  column * stride.second - pad_lo.second + kx * dilation.second;
              if (in_column < 0 || in_column >= iw) {
                continue;
              }
              for (int channel = 0; channel < c; ++channel) {
                accumulator += input[((batch * ih + in_row) * iw + in_column) *
                        c +
                    channel] *
                    weight[(
                               (out_channel * kh + ky) * kw + kx) *
                        c +
                        channel];
              }
            }
          }
          output[((batch * oh + row) * ow + column) * o + out_channel] =
              accumulator;
        }
      }
    }
  }
  return output;
}


TEST_CASE("Convolution matches host references through Vulkan compute") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // 3x3 identity kernel, stride 1, pad 1: the output equals the input.
  std::vector<float> identity_in(9);
  for (size_t index = 0; index < identity_in.size(); ++index) {
    identity_in[index] = dist(rng);
  }
  std::vector<float> identity_wt(9, 0.0f);
  identity_wt[4] = 1.0f;
  array identity_input(
      identity_in.begin(), Shape{1, 3, 3, 1}, float32);
  array identity_weight(identity_wt.begin(), Shape{1, 3, 3, 1}, float32);
  check_values(
      conv2d(
          identity_input,
          identity_weight,
          {1, 1},
          {1, 1},
          {1, 1},
          1,
          stream),
      identity_in,
      stream,
      1e-6);

  // Random 1x1 kernels are per-position channel matmuls.
  Shape in_shape_1x1 = {2, 5, 5, 3};
  std::vector<float> in_1x1(2 * 5 * 5 * 3);
  for (auto& value : in_1x1) {
    value = dist(rng);
  }
  std::vector<float> wt_1x1(4 * 1 * 1 * 3);
  for (auto& value : wt_1x1) {
    value = dist(rng);
  }
  auto expected_1x1 = host_conv2d_nhwc(
      in_1x1, in_shape_1x1, wt_1x1, Shape{4, 1, 1, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1});
  check_values(
      conv2d(
          array(in_1x1.begin(), in_shape_1x1, float32),
          array(wt_1x1.begin(), Shape{4, 1, 1, 3}, float32),
          {1, 1},
          {0, 0},
          {1, 1},
          1,
          stream),
      expected_1x1,
      stream,
      1e-5);

  // Random 3x3 kernel, stride 2, pad 1 exercises the window walk and
  // zero padding guards.
  Shape in_shape_3x3 = {1, 7, 7, 2};
  std::vector<float> in_3x3(1 * 7 * 7 * 2);
  for (auto& value : in_3x3) {
    value = dist(rng);
  }
  std::vector<float> wt_3x3(3 * 3 * 3 * 2);
  for (auto& value : wt_3x3) {
    value = dist(rng);
  }
  auto expected_3x3 = host_conv2d_nhwc(
      in_3x3, in_shape_3x3, wt_3x3, Shape{3, 3, 3, 2}, {2, 2}, {1, 1}, {1, 1}, {1, 1});
  check_values(
      conv2d(
          array(in_3x3.begin(), in_shape_3x3, float32),
          array(wt_3x3.begin(), Shape{3, 3, 3, 2}, float32),
          {2, 2},
          {1, 1},
          {1, 1},
          1,
          stream),
      expected_3x3,
      stream,
      1e-5);

  // Dilation 2 widens the window without touching the output grid.
  Shape in_shape_dil = {1, 9, 9, 1};
  std::vector<float> in_dil(1 * 9 * 9 * 1);
  for (auto& value : in_dil) {
    value = dist(rng);
  }
  std::vector<float> wt_dil(1 * 3 * 3 * 1);
  for (auto& value : wt_dil) {
    value = dist(rng);
  }
  auto expected_dil = host_conv2d_nhwc(
      in_dil, in_shape_dil, wt_dil, Shape{1, 3, 3, 1}, {1, 1}, {2, 2}, {2, 2}, {2, 2});
  check_values(
      conv2d(
          array(in_dil.begin(), in_shape_dil, float32),
          array(wt_dil.begin(), Shape{1, 3, 3, 1}, float32),
          {1, 1},
          {2, 2},
          {2, 2},
          1,
          stream),
      expected_dil,
      stream,
      1e-5);

  // Asymmetric padding and a bias add ride conv_general plus a broadcast.
  Shape in_shape_pad = {1, 4, 4, 2};
  std::vector<float> in_pad(1 * 4 * 4 * 2);
  for (auto& value : in_pad) {
    value = dist(rng);
  }
  std::vector<float> wt_pad(2 * 2 * 2 * 2);
  for (auto& value : wt_pad) {
    value = dist(rng);
  }
  std::vector<float> bias(2);
  for (auto& value : bias) {
    value = dist(rng);
  }
  auto conv_pad = conv_general(
      array(in_pad.begin(), in_shape_pad, float32),
      array(wt_pad.begin(), Shape{2, 2, 2, 2}, float32),
      {1, 1},
      {0, 1},
      {2, 0},
      {1, 1},
      {1, 1},
      1,
      false,
      stream);
  auto expected_pad = host_conv2d_nhwc(
      in_pad, in_shape_pad, wt_pad, Shape{2, 2, 2, 2}, {1, 1}, {0, 1}, {2, 0}, {1, 1});
  std::vector<float> expected_bias(expected_pad.size());
  for (size_t index = 0; index < expected_bias.size(); ++index) {
    expected_bias[index] = expected_pad[index] + bias[index % 2];
  }
  check_values(
      add(conv_pad, array(bias.begin(), Shape{2}, float32), stream),
      expected_bias,
      stream,
      1e-5);
}

TEST_CASE("FP16 Convolution matches host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }
  Shape in_shape = {1, 5, 5, 2};
  std::vector<float> input_values(1 * 5 * 5 * 2);
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& value : input_values) {
    value = dist(rng);
  }
  std::vector<float> weight_values(2 * 3 * 3 * 2);
  for (auto& value : weight_values) {
    value = dist(rng);
  }
  auto expected = host_conv2d_nhwc(
      input_values,
      in_shape,
      weight_values,
      Shape{2, 3, 3, 2},
      {1, 1},
      {1, 1},
      {1, 1},
      {1, 1});
  check_values(
      astype(
          conv2d(
              astype(
                  array(input_values.begin(), in_shape, float32),
                  float16,
                  stream),
              astype(
                  array(weight_values.begin(), Shape{2, 3, 3, 2}, float32),
                  float16,
                  stream),
              {1, 1},
              {1, 1},
              {1, 1},
              1,
              stream),
          float32,
          stream),
      expected,
      stream,
      1e-3);
}
TEST_CASE("mx.compile evaluates a four-op elementwise chain") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {-0.4f, 0.0f, 0.3f, 0.9f};
  array x(xv.begin(), Shape{4}, float32);
  std::vector<float> expected(4);
  for (size_t index = 0; index < expected.size(); ++index) {
    expected[index] = std::sqrt(std::exp(2.0f * xv[index]) + 1.0f);
  }
  using VectorFn = std::function<std::vector<array>(const std::vector<array>&)>;

  set_compile_mode(CompileMode::enabled);
  VectorFn chain_fun = [&](const std::vector<array>& inputs) {
    auto scaled = multiply(inputs[0], array(2.0f), stream);
    auto raised = exp(scaled, stream);
    auto shifted = add(raised, array(1.0f), stream);
    return std::vector<array>{sqrt(shifted, stream)};
  };
  auto chain = compile(chain_fun);
  check_values(chain({x})[0], expected, stream, 1e-5);
  set_compile_mode(CompileMode::enabled);
}

TEST_CASE("mx.compile runs the bf16 tape bit-exact against eager") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> xv = {0.0f, 0.1f, 0.2f, 0.3f};
  std::vector<float> yv = {1.0f, 0.5f, 2.0f, 0.25f};
  array x(xv.begin(), Shape{4}, bfloat16);
  array y(yv.begin(), Shape{4}, bfloat16);
  using VectorFn = std::function<std::vector<array>(const std::vector<array>&)>;

  // The bf16 tape fuses through the chain's own bf16 kernels, which
  // round every intermediate to the storage dtype exactly like
  // per-node dispatch, so compiled output must equal eager's bf16
  // bits exactly.
  set_compile_mode(CompileMode::enabled);
  VectorFn fused_fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        multiply(add(inputs[0], inputs[1], stream), inputs[0], stream)};
  };
  auto fused = compile(fused_fun);

  set_compile_mode(CompileMode::disabled);
  std::vector<array> eager_bf16 = fused_fun({x, y});
  for (auto& out : eager_bf16) {
    out.eval();
  }
  omarchy::get_command_encoder(stream).synchronize();

  std::vector<array> compiled_bf16 = fused({x, y});
  for (auto& out : compiled_bf16) {
    out.eval();
  }
  omarchy::get_command_encoder(stream).synchronize();
  set_compile_mode(CompileMode::disabled);
  const uint16_t* eager_data = eager_bf16[0].data<uint16_t>();
  const uint16_t* compiled_data = compiled_bf16[0].data<uint16_t>();
  for (size_t index = 0; index < x.size(); ++index) {
    INFO("bf16 bits mismatch at ", index, " eager=0x", std::hex,
         eager_data[index], " compiled=0x", compiled_data[index], std::dec);
    CHECK_EQ(eager_data[index], compiled_data[index]);
  }

  // f16 and f32 tapes keep running.
  std::vector<float> expected(4);
  for (size_t index = 0; index < expected.size(); ++index) {
    expected[index] = (xv[index] + yv[index]) * xv[index];
  }
  array x16(xv.begin(), Shape{4}, float16);
  array y16(yv.begin(), Shape{4}, float16);
  // check_values reads a float32 buffer, so the f16 result is cast first.
  // F16 arithmetic keeps the host reference within 8e-3.
  check_values(
      astype(fused({x16, y16})[0], float32, stream),
      expected,
      stream,
      8e-3);
  array x32(xv.begin(), Shape{4}, float32);
  array y32(yv.begin(), Shape{4}, float32);
  check_values(fused({x32, y32})[0], expected, stream, 1e-5);
  set_compile_mode(CompileMode::enabled);
}

// ---------------------------------------------------------------------------
// Wave 3: inverse trig, hyperbolic, expm1/log1p, the erf family,
// rounding, the binary ArcTan2 / LogAddExp, and the complex-only trio.
// ---------------------------------------------------------------------------

namespace {

// Combined absolute + relative check against a host double reference.
// doctest::Approx is purely relative and cannot express a tolerance at
// expected values at or near zero (sin(0), log1p(-1e-7), ...).
void check_wave3_close(
    array value,
    const std::vector<float>& expected,
    const Stream& stream,
    double abs_tolerance = 1e-6,
    double rel_tolerance = 1e-6) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  const float* values = value.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    const double got = values[index];
    const double want = expected[index];
    const double tolerance = abs_tolerance + rel_tolerance * std::abs(want);
    CHECK_MESSAGE(
        std::abs(got - want) <= tolerance,
        "index " << index << " got " << got << " want " << want);
  }
}

void check_wave3_all_nan(array value, const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* values = value.data<float>();
  for (size_t index = 0; index < value.size(); ++index) {
    CHECK_MESSAGE(
        std::isnan(values[index]), "index " << index << " is not NaN");
  }
}

// std::erfinv is missing from libstdc++; bracketing std::erf in double
// serves as the host reference. Two hundred halvings of [-40, 40]
// exhaust double precision, so the result is deterministic.
double host_erfinv(double a) {
  double low = -40.0;
  double high = 40.0;
  for (int step = 0; step < 200; ++step) {
    double mid = 0.5 * (low + high);
    if (std::erf(mid) < a) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return 0.5 * (low + high);
}

} // namespace

TEST_CASE("Wave3 ArcCos ArcSin ArcTan ArcTan2 and hyperbolic match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // acos / asin: domain edges at +/-1, negatives, near zero.
  std::vector<float> domain = {
      -1.0f,
      -0.99999f,
      -0.75f,
      -0.25f,
      0.0f,
      0.25f,
      0.75f,
      0.99999f,
      1.0f};
  std::vector<float> acos_expected;
  std::vector<float> asin_expected;
  for (float value : domain) {
    acos_expected.push_back((float)std::acos((double)value));
    asin_expected.push_back((float)std::asin((double)value));
  }
  array x(domain.begin(), Shape{static_cast<int>(domain.size())}, float32);
  check_wave3_close(arccos(x, stream), acos_expected, stream);
  check_wave3_close(arcsin(x, stream), asin_expected, stream);
  // Outside the domain the host keeps NaN.
  array outside({-2.0f, 1.5f}, float32);
  check_wave3_all_nan(arccos(outside, stream), stream);
  check_wave3_all_nan(arcsin(outside, stream), stream);

  std::vector<float> wide = {-8.0f, -1.0f, -0.25f, 0.0f, 0.25f, 1.0f, 8.0f};
  std::vector<float> atan_expected;
  for (float value : wide) {
    atan_expected.push_back((float)std::atan((double)value));
  }
  array w(wide.begin(), Shape{static_cast<int>(wide.size())}, float32);
  check_wave3_close(arctan(w, stream), atan_expected, stream);

  // ArcTan2 quadrant anchors; upstream is atan2(lhs, rhs) with the
  // first operand as y, matching std::atan2 and np.arctan2.
  std::vector<float> yv = {
      1.0f,
      -1.0f,
      -1.0f,
      1.0f,
      0.0f,
      1.0f,
      0.0f,
      -1.0f,
      3.0f,
      -3.0f};
  std::vector<float> xv = {
      1.0f,
      1.0f,
      -1.0f,
      -1.0f,
      1.0f,
      0.0f,
      -1.0f,
      0.0f,
      -4.0f,
      -4.0f};
  std::vector<float> atan2_expected;
  for (size_t index = 0; index < yv.size(); ++index) {
    atan2_expected.push_back(
        (float)std::atan2((double)yv[index], (double)xv[index]));
  }
  array y(yv.begin(), Shape{static_cast<int>(yv.size())}, float32);
  array x2(xv.begin(), Shape{static_cast<int>(xv.size())}, float32);
  check_wave3_close(arctan2(y, x2, stream), atan2_expected, stream);

  // acosh: domain edge at 1, NaN below 1.
  std::vector<float> positive = {1.0f, 1.0001f, 1.5f, 3.0f, 42.0f};
  std::vector<float> acosh_expected;
  for (float value : positive) {
    acosh_expected.push_back((float)std::acosh((double)value));
  }
  array p(positive.begin(), Shape{static_cast<int>(positive.size())}, float32);
  check_wave3_close(arccosh(p, stream), acosh_expected, stream);
  array below({0.5f}, float32);
  check_wave3_all_nan(arccosh(below, stream), stream);

  std::vector<float> spread = {
      -42.0f,
      -4.0f,
      -1.0f,
      -0.25f,
      0.0f,
      0.25f,
      1.0f,
      4.0f,
      42.0f};
  std::vector<float> asinh_expected;
  std::vector<float> cosh_expected;
  std::vector<float> sinh_expected;
  std::vector<float> tanh_expected;
  for (float value : spread) {
    asinh_expected.push_back((float)std::asinh((double)value));
    cosh_expected.push_back((float)std::cosh((double)value));
    sinh_expected.push_back((float)std::sinh((double)value));
    tanh_expected.push_back((float)std::tanh((double)value));
  }
  array s(spread.begin(), Shape{static_cast<int>(spread.size())}, float32);
  check_wave3_close(arcsinh(s, stream), asinh_expected, stream);
  // cosh(42) is 8.7e17: one float32 ulp there is 6.4e10, so the
  // relative tolerance widens to two ulps of the largest value.
  check_wave3_close(cosh(s, stream), cosh_expected, stream, 1e-6, 2e-6);
  // sinh(-42) and sinh(42) sit at -/+8.7e17 like cosh and carry the
  // same widened two-ulp relative bound.
  check_wave3_close(sinh(s, stream), sinh_expected, stream, 1e-6, 2e-6);
  check_wave3_close(tanh(s, stream), tanh_expected, stream);

  std::vector<float> tan_domain = {
      -3.0f,
      -0.5f,
      -0.25f,
      0.0f,
      0.25f,
      0.5f,
      3.0f};
  std::vector<float> tan_expected;
  for (float value : tan_domain) {
    tan_expected.push_back((float)std::tan((double)value));
  }
  array t(tan_domain.begin(), Shape{static_cast<int>(tan_domain.size())}, float32);
  check_wave3_close(tan(t, stream), tan_expected, stream);
}

TEST_CASE("Wave3 Expm1 Log1p Erf ErfInv match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // expm1: near-zero cancellation, negatives, and large positives.
  std::vector<float> xv = {
      -88.0f,
      -20.0f,
      -1.0f,
      -1e-7f,
      -0.25f,
      0.0f,
      1e-7f,
      0.25f,
      1.0f,
      3.5f,
      20.0f,
      80.0f};
  std::vector<float> expm1_expected;
  for (float value : xv) {
    expm1_expected.push_back((float)std::expm1((double)value));
  }
  array x(xv.begin(), Shape{static_cast<int>(xv.size())}, float32);
  check_wave3_close(expm1(x, stream), expm1_expected, stream);

  // log1p: near -1 cancellation and near zero.
  std::vector<float> lv = {
      -0.99999f,
      -0.999f,
      -0.5f,
      -1e-7f,
      0.0f,
      1e-7f,
      0.5f,
      1.0f,
      20.0f};
  std::vector<float> log1p_expected;
  for (float value : lv) {
    log1p_expected.push_back((float)std::log1p((double)value));
  }
  array l(lv.begin(), Shape{static_cast<int>(lv.size())}, float32);
  check_wave3_close(log1p(l, stream), log1p_expected, stream);
  // -1 keeps -inf and below -1 keeps NaN, matching the host.
  array edge({-1.0f}, float32);
  array edge_out = log1p(edge, stream);
  edge_out.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK(std::isinf(edge_out.data<float>()[0]));
  CHECK(edge_out.data<float>()[0] < 0.0f);
  array below({-1.5f}, float32);
  check_wave3_all_nan(log1p(below, stream), stream);

  // erf: dense grid across the central and tail branches.
  std::vector<float> grid;
  for (float value = -6.0f; value <= 6.0f; value += 0.1f) {
    grid.push_back(value);
  }
  std::vector<float> erf_expected;
  for (float value : grid) {
    erf_expected.push_back((float)std::erf((double)value));
  }
  array g(grid.begin(), Shape{static_cast<int>(grid.size())}, float32);
  check_wave3_close(erf(g, stream), erf_expected, stream);

  // erfinv against the double bracketing reference.
  std::vector<float> av = {
      -0.9999f,
      -0.999f,
      -0.99f,
      -0.9f,
      -0.5f,
      -1e-7f,
      0.0f,
      1e-7f,
      0.5f,
      0.9f,
      0.99f,
      0.999f,
      0.9999f};
  std::vector<float> erfinv_expected;
  for (float value : av) {
    erfinv_expected.push_back((float)host_erfinv((double)value));
  }
  array a(av.begin(), Shape{static_cast<int>(av.size())}, float32);
  // The erfinv argument transform squares the input; the software
  // pipeline's log accuracy in log(1 - a^2) lands the tail (a = +/-0.9999,
  // y = +/-2.75) at 3.2e-6 relative on this device, so the tolerance
  // carries the measured bound.
  check_wave3_close(erfinv(a, stream), erfinv_expected, stream, 1e-6, 5e-6);
  array sat({-1.0f, 1.0f}, float32);
  array sat_out = erfinv(sat, stream);
  sat_out.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK(std::isinf(sat_out.data<float>()[0]));
  CHECK(sat_out.data<float>()[0] < 0.0f);
  CHECK(std::isinf(sat_out.data<float>()[1]));
  CHECK(sat_out.data<float>()[1] > 0.0f);
  check_wave3_all_nan(erfinv(array({1.5f}, float32), stream), stream);

  // Upstream anchor: the erfinv(erf(x)) round trip. The kernel erf
  // holds about 1e-7 absolute, and d erfinv / d erf =
  // (sqrt(pi)/2) exp(x^2), so the round-trip error grows with x^2;
  // 1e-5 covers |x| <= 2 with margin.
  std::vector<float> rv = {
      -2.0f,
      -1.5f,
      -1.0f,
      -0.75f,
      -0.25f,
      0.0f,
      0.25f,
      0.75f,
      1.0f,
      1.5f,
      2.0f};
  std::vector<float> roundtrip;
  for (float value : rv) {
    roundtrip.push_back(value);
  }
  array r(rv.begin(), Shape{static_cast<int>(rv.size())}, float32);
  check_wave3_close(erfinv(erf(r, stream), stream), roundtrip, stream, 1e-5);

  // The dispatch is dtype-generic: float16 keeps the same kernel with
  // float32 arithmetic and a looser storage grid.
  const auto& capabilities = omarchy::device(0).capabilities();
  if (!capabilities.shader_float16 ||
      !capabilities.storage_buffer_16bit_access) {
    skip("Vulkan device lacks required FP16 shader and storage features.");
    return;
  }
  std::vector<float> hv = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
  std::vector<float> htanh;
  std::vector<float> herf;
  for (float value : hv) {
    htanh.push_back((float)std::tanh((double)value));
    herf.push_back((float)std::erf((double)value));
  }
  array h(hv.begin(), Shape{static_cast<int>(hv.size())}, float16);
  check_wave3_close(
      astype(tanh(h, stream), float32, stream), htanh, stream, 8e-3);
  check_wave3_close(
      astype(erf(h, stream), float32, stream), herf, stream, 8e-3);
}

TEST_CASE("Wave3 Ceil Floor Round and LogAddExp match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  std::vector<float> xv = {
      -2.5f,
      -1.5f,
      -1.25f,
      -0.5f,
      0.0f,
      0.5f,
      1.25f,
      1.5f,
      2.5f,
      3.75f};
  std::vector<float> ceil_expected;
  std::vector<float> floor_expected;
  std::vector<float> round_expected;
  for (float value : xv) {
    ceil_expected.push_back(std::ceil(value));
    floor_expected.push_back(std::floor(value));
    // Upstream Round is rint on CPU and Metal: halfway cases round to
    // the nearest even integer, which is the default nearbyint mode.
    round_expected.push_back(std::nearbyint((double)value));
  }
  array x(xv.begin(), Shape{static_cast<int>(xv.size())}, float32);
  check_wave3_close(ceil(x, stream), ceil_expected, stream);
  check_wave3_close(floor(x, stream), floor_expected, stream);
  check_wave3_close(round(x, stream), round_expected, stream);

  // Half-even anchors: +/-0.5 stays at zero, +/-1.5 and +/-2.5 go to
  // the even neighbor.
  std::vector<float> halves = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f};
  std::vector<float> half_expected;
  for (float value : halves) {
    half_expected.push_back(std::nearbyint((double)value));
  }
  array h(halves.begin(), Shape{static_cast<int>(halves.size())}, float32);
  check_wave3_close(round(h, stream), half_expected, stream);

  // LogAddExp: max + log1p(exp(-|a-b|)) in double on the host.
  std::vector<float> av = {
      1.0f, -1.0f, 1000.0f, 1000.0f, 70000.0f, -745.0f};
  std::vector<float> bv = {
      2.0f, 2.0f, 1000.0f, -1000.0f, 69999.5f, -746.0f};
  std::vector<float> lae_expected;
  for (size_t index = 0; index < av.size(); ++index) {
    double m = std::max((double)av[index], (double)bv[index]);
    lae_expected.push_back(
        (float)(m + std::log1p(std::exp(-std::abs((double)av[index] -
                                                   (double)bv[index])))));
  }
  array a(av.begin(), Shape{static_cast<int>(av.size())}, float32);
  array b(bv.begin(), Shape{static_cast<int>(bv.size())}, float32);
  check_wave3_close(logaddexp(a, b, stream), lae_expected, stream);

  // Special values: one -inf keeps the other side, both -inf keep
  // -inf, +inf wins over any finite, and NaN propagates.
  std::vector<float> ninf_v{-std::numeric_limits<float>::infinity()};
  std::vector<float> pinf_v{std::numeric_limits<float>::infinity()};
  array ninf(ninf_v.begin(), Shape{1}, float32);
  array pinf(pinf_v.begin(), Shape{1}, float32);
  array five({5.0f}, float32);
  auto lae_at = [&](array lhs, array rhs) {
    array value = logaddexp(lhs, rhs, stream);
    value.eval();
    omarchy::get_command_encoder(stream).synchronize();
    return value.data<float>()[0];
  };
  CHECK_EQ(lae_at(ninf, five), 5.0f);
  CHECK_EQ(lae_at(five, ninf), 5.0f);
  CHECK(std::isinf(lae_at(ninf, ninf)));
  CHECK(lae_at(ninf, ninf) < 0.0f);
  CHECK(std::isinf(lae_at(pinf, ninf)));
  CHECK(lae_at(pinf, ninf) > 0.0f);
  array nan_v(std::numeric_limits<float>::quiet_NaN());
  CHECK(std::isnan(lae_at(nan_v, five)));
}

TEST_CASE("Wave3 Conjugate Real Imag mirror upstream on real dtypes and compute on complex") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Upstream never constructs the Conjugate, Real, or Imag primitives
  // for real dtypes: conjugate() and real() return the input array
  // itself and imag() returns zeros_like (mlx/ops.cpp:6527, :6743,
  // :6751). The API-level mirror is identity, identity, zeros.
  std::vector<float> xv = {1.0f, -2.0f, 3.5f};
  array x(xv.begin(), Shape{3}, float32);
  check_wave3_close(conjugate(x, stream), xv, stream);
  check_wave3_close(real(x, stream), xv, stream);
  check_wave3_close(imag(x, stream), {0.0f, 0.0f, 0.0f}, stream);

  // Complex input reaches the backend primitives, and they compute now:
  // complex64 transport landed on 2026-09-02, so conjugate flips the
  // imaginary sign, real and imag project the components, and none of
  // the three refuses. This case previously pinned those refusals; the
  // pin went stale the moment the primitives were implemented, which is
  // the same rot that had already hit the error-contract suite twice.
  array c = array(complex64_t{1.0f, 2.0f});
  array conj_c = conjugate(c, stream);
  array real_c = real(c, stream);
  array imag_c = imag(c, stream);
  conj_c.eval();
  real_c.eval();
  imag_c.eval();
  CHECK(conj_c.dtype() == complex64);
  CHECK(real_c.dtype() == float32);
  CHECK(imag_c.dtype() == float32);
  complex64_t conj_value = conj_c.data<complex64_t>()[0];
  CHECK(conj_value.real() == doctest::Approx(1.0f));
  CHECK(conj_value.imag() == doctest::Approx(-2.0f));
  CHECK(real_c.data<float>()[0] == doctest::Approx(1.0f));
  CHECK(imag_c.data<float>()[0] == doctest::Approx(2.0f));
}

TEST_CASE("Wave3 Conjugate Real Imag mirror is exact for real dtypes across views and broadcasts") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& capabilities = omarchy::device(0).capabilities();

  // Upstream builds Conjugate, Real, and Imag only on the complex64
  // side of their op gates (mlx/ops.cpp:6529, :6744, :6751): real
  // dtypes return the input array for conjugate() and real() and
  // full_like zeros for imag(). No real-dtype path constructs the
  // primitive, so the backend refusals stay, and the observable
  // contract is the exact mirror pinned here. Identity results share
  // the input storage, so the identity checks read both arrays over
  // the base buffer span and compare exactly.
  auto eval_and_sync = [&](array value) {
    value.eval();
    omarchy::get_command_encoder(stream).synchronize();
  };
  auto check_mirror_identity = [&](array base, array result, auto tag) {
    using T = decltype(tag);
    eval_and_sync(result);
    CHECK_EQ(result.dtype(), base.dtype());
    const T* got = result.data<T>();
    const T* want = base.data<T>();
    for (size_t index = 0; index < base.size(); ++index) {
      CHECK_EQ(got[index], want[index]);
    }
  };
  auto check_mirror_zeros = [&](array result, Dtype dtype, auto tag, int count) {
    using T = decltype(tag);
    eval_and_sync(result);
    CHECK_EQ(result.dtype(), dtype);
    CHECK_EQ(result.size(), static_cast<size_t>(count));
    const T* got = result.data<T>();
    for (int index = 0; index < count; ++index) {
      CHECK_EQ(got[index], T(0));
    }
  };

  // float32: identity, identity, exact zeros.
  std::vector<float> fv = {1.0f, -2.0f, 3.5f, 0.25f};
  array f(fv.begin(), Shape{4}, float32);
  array f_conjugate = conjugate(f, stream);
  check_mirror_identity(f, f_conjugate, float{});
  CHECK_EQ(f_conjugate.size(), 4);
  array f_real = real(f, stream);
  check_mirror_identity(f, f_real, float{});
  CHECK_EQ(f_real.size(), 4);
  array f_imag = imag(f, stream);
  check_mirror_zeros(f_imag, float32, float{}, 4);

  // int32 and uint32: the same mirror through the typed buffers.
  std::vector<int32_t> iv = {5, -7, 0, 123456};
  array i(iv.begin(), Shape{4}, int32);
  check_mirror_identity(i, conjugate(i, stream), int32_t{});
  check_mirror_identity(i, real(i, stream), int32_t{});
  check_mirror_zeros(imag(i, stream), int32, int32_t{}, 4);

  std::vector<uint32_t> uv = {9u, 4294967290u, 0u, 123456u};
  array u(uv.begin(), Shape{4}, uint32);
  check_mirror_identity(u, conjugate(u, stream), uint32_t{});
  check_mirror_identity(u, real(u, stream), uint32_t{});
  check_mirror_zeros(imag(u, stream), uint32, uint32_t{}, 4);

  // float16 and bfloat16 behind the shader and 16-bit storage gates;
  // bfloat16 adds the int16 storage requirement. The chosen values are
  // exact in both formats, so the f32 readback is bit-exact.
  if (capabilities.shader_float16 && capabilities.storage_buffer_16bit_access) {
    std::vector<float> hv = {1.0f, -2.0f, 3.5f, 0.25f};
    for (Dtype dtype : {float16, bfloat16}) {
      if (dtype == bfloat16 && !capabilities.shader_int16) {
        continue;
      }
      array h(hv.begin(), Shape{4}, dtype);
      auto check_cast_exact = [&](array result, const std::vector<float>& want) {
        array wide = astype(result, float32, stream);
        eval_and_sync(wide);
        CHECK_EQ(wide.dtype(), float32);
        const float* got = wide.data<float>();
        for (size_t index = 0; index < want.size(); ++index) {
          CHECK_EQ(got[index], want[index]);
        }
      };
      check_cast_exact(conjugate(h, stream), hv);
      check_cast_exact(real(h, stream), hv);
      check_cast_exact(imag(h, stream), std::vector<float>(hv.size(), 0.0f));
    }
  }

  // Transposed view: identity returns the same strided view (shape
  // {3, 2}, base storage equal); imag() fills exact zeros of the view
  // shape.
  std::vector<float> tv = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  array t(tv.begin(), Shape{2, 3}, float32);
  array t_view = transpose(t, stream);
  array vt_conjugate = conjugate(t_view, stream);
  check_mirror_identity(t, vt_conjugate, float{});
  CHECK_EQ(vt_conjugate.shape(0), 3);
  CHECK_EQ(vt_conjugate.shape(1), 2);
  array vt_real = real(t_view, stream);
  check_mirror_identity(t, vt_real, float{});
  CHECK_EQ(vt_real.shape(0), 3);
  CHECK_EQ(vt_real.shape(1), 2);
  array vt_imag = imag(t_view, stream);
  check_mirror_zeros(vt_imag, float32, float{}, 6);
  CHECK_EQ(vt_imag.shape(0), 3);
  CHECK_EQ(vt_imag.shape(1), 2);

  // Broadcast view: identity keeps the stride-0 view over the same
  // base storage; imag() fills broadcast-shaped zeros without reading
  // the view data.
  array bc_base(fv.begin(), Shape{4}, float32);
  array bc = broadcast_to(bc_base, Shape{2, 4}, stream);
  array bc_conjugate = conjugate(bc, stream);
  check_mirror_identity(bc_base, bc_conjugate, float{});
  CHECK_EQ(bc_conjugate.shape(0), 2);
  CHECK_EQ(bc_conjugate.shape(1), 4);
  array bc_real = real(bc, stream);
  check_mirror_identity(bc_base, bc_real, float{});
  CHECK_EQ(bc_real.shape(0), 2);
  CHECK_EQ(bc_real.shape(1), 4);
  array bc_imag = imag(bc, stream);
  check_mirror_zeros(bc_imag, float32, float{}, 8);
  CHECK_EQ(bc_imag.shape(0), 2);
  CHECK_EQ(bc_imag.shape(1), 4);
}

namespace {

// Host references for the wave-2 family, mirroring the upstream CPU
// functors (mlx/backend/cpu/simd/base_simd.h and binary.cpp) one
// formula at a time.
int host_python_mod(int a, int b) {
  int r = a % b;
  if (r != 0 && (r < 0) != (b < 0)) {
    r += b;
  }
  return r;
}

int host_floor_div(int a, int b) {
  int q = a / b;
  int r = a % b;
  if (r != 0 && (r < 0) != (b < 0)) {
    q -= 1;
  }
  return q;
}

float host_float_mod(float a, float b) {
  float r = std::fmod(a, b);
  if (r != 0.0f && (r < 0.0f) != (b < 0.0f)) {
    r += b;
  }
  return r;
}

void check_bool(
    array value,
    const std::vector<bool>& expected,
    const Stream& stream) {
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  REQUIRE_EQ(value.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(value.data<bool>()[index], expected[index]);
  }
}

float host_sign(float x) {
  return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f);
}

} // namespace

TEST_CASE("Less, LessEqual, Greater, and NotEqual match host references and the NaN matrix") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Float32 rows mixing negatives, zero, and NaN against one host
  // comparator loop. C++ semantics: ordered comparisons of NaN are
  // false and NaN != x is true.
  std::vector<float> xv = {-3.0f, 0.5f, 2.0f, std::nanf("")};
  std::vector<float> yv = {0.0f, 0.5f, -1.5f, 1.0f};
  array x(xv.begin(), Shape{4}, float32);
  array y(yv.begin(), Shape{4}, float32);
  // Reference comparisons land in plain bool storage. std::vector<bool>
  // is bit-packed: every write goes through a read-modify-write proxy,
  // and GCC at -O1 and above on aarch64 miscompiled four interleaved
  // comparisons through it (NaN != 1.0f folded to false). The proxy is
  // rebuilt from the plain values only at the call.
  std::array<bool, 4> lt{}, le{}, gt{}, ne{};
  for (size_t i = 0; i < 4; ++i) {
    lt[i] = xv[i] < yv[i];
    le[i] = xv[i] <= yv[i];
    gt[i] = xv[i] > yv[i];
    ne[i] = xv[i] != yv[i];
  }
  check_bool(less(x, y, stream), std::vector<bool>(lt.begin(), lt.end()), stream);
  check_bool(less_equal(x, y, stream), std::vector<bool>(le.begin(), le.end()), stream);
  check_bool(greater(x, y, stream), std::vector<bool>(gt.begin(), gt.end()), stream);
  check_bool(not_equal(x, y, stream), std::vector<bool>(ne.begin(), ne.end()), stream);

  // The NaN matrix: NaN against NaN and NaN against one. Every
  // ordered comparison and Equal stay false, NotEqual alone is true.
  std::vector<float> nav = {std::nanf(""), std::nanf(""), 1.0f};
  std::vector<float> nbv = {std::nanf(""), 1.0f, std::nanf("")};
  array na(nav.begin(), Shape{3}, float32);
  array nb(nbv.begin(), Shape{3}, float32);
  check_bool(less(na, nb, stream), {false, false, false}, stream);
  check_bool(less_equal(na, nb, stream), {false, false, false}, stream);
  check_bool(greater(na, nb, stream), {false, false, false}, stream);
  // GreaterEqual now serves the full comparison dtype table, so the
  // float NaN row follows C++ ordering: every ordered comparison of a
  // NaN stays false.
  check_bool(greater_equal(na, nb, stream), {false, false, false}, stream);
  check_bool(equal(na, nb, stream), {false, false, false}, stream);
  check_bool(not_equal(na, nb, stream), {true, true, true}, stream);
  std::vector<int32_t> iv = {-2, 0, 7};
  std::vector<int32_t> jv = {-1, 0, 3};
  array i(iv.begin(), Shape{3}, int32);
  array j(jv.begin(), Shape{3}, int32);
  check_bool(greater(i, j, stream), {false, false, true}, stream);
  check_bool(less_equal(i, j, stream), {true, true, false}, stream);
  check_bool(not_equal(i, j, stream), {true, false, true}, stream);

  // Row against scalar and a leading-axis broadcast go through the
  // modulo and stride transports.
  check_bool(less(x, array(0.5f), stream), {true, false, false, false}, stream);
  std::vector<float> wide_v = {1.0f, 3.0f, 3.0f, 3.0f};
  std::vector<float> tall_v = {2.0f, 3.0f, 4.0f, 1.0f, 3.0f, 4.0f, 1.0f, 5.0f};
  array wide(wide_v.begin(), Shape{1, 4}, float32);
  array tall(tall_v.begin(), Shape{2, 4}, float32);
  check_bool(greater(tall, wide, stream),
      {true, false, true, false, true, true, false, true},
      stream);

  // Float16 and bfloat16 keep their storage variants.
  std::vector<float> hv = {0.5f, -2.0f};
  array h(hv.begin(), Shape{2}, float16);
  check_bool(greater(h, array(0.5f, float16), stream), {false, false}, stream);
  check_bool(not_equal(h, array(0.5f, float16), stream), {false, true}, stream);
  check_bool(less(h, array(-1.0f, float16), stream), {false, true}, stream);
  array bf(hv.begin(), Shape{2}, bfloat16);
  check_bool(not_equal(bf, array(0.5f, bfloat16), stream), {false, true}, stream);

  // Unsigned comparisons run the unsigned shader variant: identical
  // self-comparison is all-false, and 3 < 0 is false the way the
  // unsigned reference is.
  std::vector<uint32_t> uv = {3u, 0u};
  array u(uv.begin(), Shape{2}, uint32);
  check_bool(less(u, u, stream), {false, false}, stream);
}

TEST_CASE("LogicalAnd and LogicalNot match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // 33 elements cross the 32-bit word packing boundary. The pattern and
  // reference land in plain bool storage: interleaved writes through the
  // bit-packed std::vector<bool> proxy are a known aarch64 miscompile
  // shape at -O1 and above (see the comparison test above).
  std::array<bool, 33> xv{}, yv{}, and_expected{};
  for (size_t i = 0; i < 33; ++i) {
    xv[i] = (i % 3) != 0;
    yv[i] = (i % 5) != 0;
    and_expected[i] = xv[i] && yv[i];
  }
  array x(xv.begin(), Shape{33}, bool_);
  array y(yv.begin(), Shape{33}, bool_);
  array landed = logical_and(x, y, stream);
  landed.eval();
  omarchy::get_command_encoder(stream).synchronize();
  for (size_t i = 0; i < 33; ++i) {
    CHECK_EQ(landed.data<bool>()[i], and_expected[i]);
  }

  // A broadcast row against a scalar condition.
  array scalar_true = logical_and(x, array(true), stream);
  scalar_true.eval();
  omarchy::get_command_encoder(stream).synchronize();
  for (size_t i = 0; i < 33; ++i) {
    CHECK_EQ(scalar_true.data<bool>()[i], xv[i]);
  }

  // LogicalNot is unary: single input, same word transport.
  std::vector<bool> nv = {true, false, true, true, false};
  array n(nv.begin(), Shape{5}, bool_);
  array flipped = logical_not(n, stream);
  flipped.eval();
  omarchy::get_command_encoder(stream).synchronize();
  for (size_t i = 0; i < 5; ++i) {
    CHECK_EQ(flipped.data<bool>()[i], !nv[i]);
  }
  array scalar_flip = logical_not(array(false), stream);
  scalar_flip.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(scalar_flip.data<bool>()[0], true);
}

TEST_CASE("BitwiseBinary and BitwiseInvert match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Signed values, including negatives for the arithmetic shift.
  std::vector<int32_t> av = {-1, 12, 5, -8};
  std::vector<int32_t> bv = {10, 6, -3, 3};
  array a(av.begin(), Shape{4}, int32);
  array b(bv.begin(), Shape{4}, int32);
  check_int32_values(bitwise_and(a, b, stream), {av[0] & bv[0], av[1] & bv[1], av[2] & bv[2], av[3] & bv[3]}, stream);
  check_int32_values(bitwise_or(a, b, stream), {av[0] | bv[0], av[1] | bv[1], av[2] | bv[2], av[3] | bv[3]}, stream);
  check_int32_values(bitwise_xor(a, b, stream), {av[0] ^ bv[0], av[1] ^ bv[1], av[2] ^ bv[2], av[3] ^ bv[3]}, stream);

  // Shifts: signed right shift is arithmetic, matching the upstream
  // C++ operator on this platform.
  std::vector<int32_t> sv = {1, 2, -8};
  std::vector<int32_t> cv = {2, 1, 1};
  array s(sv.begin(), Shape{3}, int32);
  array c(cv.begin(), Shape{3}, int32);
  check_int32_values(left_shift(s, c, stream), {4, 4, -16}, stream);
  std::vector<int32_t> rv = {-8, -1, 64};
  std::vector<int32_t> dv = {1, 1, 3};
  array r(rv.begin(), Shape{3}, int32);
  array d(dv.begin(), Shape{3}, int32);
  check_int32_values(right_shift(r, d, stream), {-4, -1, 8}, stream);

  // Unsigned: large magnitudes and a logical right shift.
  std::vector<uint32_t> uv = {0xFFFFFFFFu, 0x80000000u, 0xF0F0F0F0u};
  std::vector<uint32_t> vv = {0x0F0F0F0Fu, 1u, 4u};
  array u(uv.begin(), Shape{3}, uint32);
  array v(vv.begin(), Shape{3}, uint32);
  check_uint32_values(
      bitwise_or(u, v, stream), {uv[0] | vv[0], uv[1] | vv[1], uv[2] | vv[2]}, stream);
  // Shift counts stay inside the width: counts at or above it are as
  // undefined here as they are in the upstream C++ reference.
  std::vector<uint32_t> suv = {0xFFFFFFFFu, 0x80000000u, 0xF0F0F0F0u};
  std::vector<uint32_t> scv = {20u, 1u, 4u};
  array s32(suv.begin(), Shape{3}, uint32);
  array c32(scv.begin(), Shape{3}, uint32);
  check_uint32_values(
      right_shift(s32, c32, stream), {4095u, 1073741824u, 252645135u}, stream);
  check_uint32_values(left_shift(v, array(4u, uint32), stream), {0xF0F0F0F0u, 16u, 64u}, stream);
  // Invert covers both signednesses.
  check_int32_values(bitwise_invert(a, stream), {~av[0], ~av[1], ~av[2], ~av[3]}, stream);
  check_uint32_values(bitwise_invert(u, stream), {~uv[0], ~uv[1], ~uv[2]}, stream);

  // Boolean inputs compute as the logical ops on the byte lanes: the
  // truth tables land exactly, never a non-0/1 byte.
  std::array<bool, 4> tbv = {true, false, true, false};
  std::array<bool, 4> tbv2 = {true, true, false, false};
  array bt(tbv.begin(), Shape{4}, bool_);
  array bt2(tbv2.begin(), Shape{4}, bool_);
  auto check_bool = [&](array value, std::array<bool, 4> expected) {
    value.eval();
    omarchy::get_command_encoder(stream).synchronize();
    const bool* words = value.data<bool>();
    for (size_t i = 0; i < 4; ++i) {
      CHECK_EQ(words[i], expected[i]);
    }
  };
  check_bool(bitwise_and(bt, bt2, stream), {true, false, false, false});
  check_bool(bitwise_or(bt, bt2, stream), {true, true, true, false});
  check_bool(bitwise_xor(bt, bt2, stream), {false, true, true, false});
  // Upstream bool add is the logical or, and bool max/min keep {0,1}.
  check_bool(add(bt, bt2, stream), {true, true, true, false});
  check_bool(maximum(bt, bt2, stream), {true, true, true, false});
  check_bool(minimum(bt, bt2, stream), {true, false, false, false});
  // Float inputs are rejected one level up by the upstream op gate,
  // which throws at graph-build time rather than eval time.
  REQUIRE_THROWS_AS(
      bitwise_xor(array(1.0f), array(2.0f), stream), std::runtime_error);
}
TEST_CASE("integer Remainder, DivMod, Power, Sign, and Abs match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Remainder takes the divisor's sign: negatives both ways.
  std::vector<int32_t> av = {7, -7, 7, -7, 5, 0};
  std::vector<int32_t> bv = {3, 3, -3, -3, 5, 3};
  array a(av.begin(), Shape{6}, int32);
  array b(bv.begin(), Shape{6}, int32);
  std::vector<int32_t> mod_expected(6);
  std::vector<int32_t> quot_expected(6);
  for (size_t i = 0; i < 6; ++i) {
    mod_expected[i] = host_python_mod(av[i], bv[i]);
    quot_expected[i] = host_floor_div(av[i], bv[i]);
  }
  check_int32_values(remainder(a, b, stream), mod_expected, stream);

  // DivMod returns the floor-division quotient and remainder as two
  // outputs sharing one evaluation.
  auto quot_and_rem = divmod(a, b, stream);
  REQUIRE_EQ(quot_and_rem.size(), 2u);
  check_int32_values(quot_and_rem[0], quot_expected, stream);
  check_int32_values(quot_and_rem[1], mod_expected, stream);

  // Unsigned remainder and divmod.
  std::vector<uint32_t> uv = {7u, 0xFFFFFFFFu};
  std::vector<uint32_t> vv = {3u, 16u};
  array u(uv.begin(), Shape{2}, uint32);
  array v(vv.begin(), Shape{2}, uint32);
  check_uint32_values(remainder(u, v, stream), {1u, 15u}, stream);
  auto udivmod = divmod(u, v, stream);
  check_uint32_values(udivmod[0], {2u, 0x0FFFFFFFu}, stream);
  check_uint32_values(udivmod[1], {1u, 15u}, stream);

  // Power: squaring loop, negative base, zero exponent, and the
  // negative-signed-exponent yields-zero rule.
  std::vector<int32_t> pv = {2, -2, -2, 5, 2, 0};
  std::vector<int32_t> qv = {10, 3, 2, 0, -1, 0};
  array p(pv.begin(), Shape{6}, int32);
  array q(qv.begin(), Shape{6}, int32);
  check_int32_values(
      power(p, q, stream), {1024, -8, 4, 1, 0, 1}, stream);
  std::vector<uint32_t> pvu = {3, 7};
  std::vector<uint32_t> qvu = {4, 2};
  array pu(pvu.begin(), Shape{2}, uint32);
  array qu(qvu.begin(), Shape{2}, uint32);
  check_uint32_values(power(pu, qu, stream), {81u, 49u}, stream);

  // Sign: three-way for signed, 0/1 for unsigned.
  std::vector<int32_t> sv = {-5, 0, 7};
  array s(sv.begin(), Shape{3}, int32);
  check_int32_values(sign(s, stream), {-1, 0, 1}, stream);
  check_uint32_values(sign(u, stream), {1u, 1u}, stream);

  // Abs over signed values and the INT_MIN edge: upstream never
  // special-cases it, so the negation wraps to itself exactly as the
  // C++ reference does on this platform.
  std::vector<int32_t> wv = {-5, 5, 0,
      std::numeric_limits<int32_t>::min()};
  array w(wv.begin(), Shape{4}, int32);
  check_int32_values(
      abs(w, stream),
      {5, 5, 0, std::numeric_limits<int32_t>::min()},
      stream);
  check_uint32_values(abs(u, stream), uv, stream);

  // The 64-bit family rides the widened kernel: remainder and divmod
  // match the host fixups sign-for-sign.
  std::vector<int64_t> lv = {7, -7, 7, -7, 5};
  std::vector<int64_t> lv2 = {3, 3, -3, -3, 5};
  std::vector<int64_t> lmod_expected(5);
  std::vector<int64_t> lquot_expected(5);
  for (size_t i = 0; i < 5; ++i) {
    lmod_expected[i] =
        host_python_mod(static_cast<int>(lv[i]), static_cast<int>(lv2[i]));
    lquot_expected[i] =
        host_floor_div(static_cast<int>(lv[i]), static_cast<int>(lv2[i]));
  }
  array l(lv.begin(), Shape{5}, int64);
  array l2(lv2.begin(), Shape{5}, int64);
  check_int64_values(remainder(l, l2, stream), lmod_expected, stream);
  auto ldivmod = divmod(l, l2, stream);
  REQUIRE_EQ(ldivmod.size(), 2u);
  check_int64_values(ldivmod[0], lquot_expected, stream);
  check_int64_values(ldivmod[1], lmod_expected, stream);
  std::vector<uint64_t> ulv = {7ull, 0xFFFFFFFFFFull};
  std::vector<uint64_t> ulv2 = {3ull, 16ull};
  array ul(ulv.begin(), Shape{2}, uint64);
  array ul2(ulv2.begin(), Shape{2}, uint64);
  check_uint64_values(remainder(ul, ul2, stream), {1ull, 15ull}, stream);
  auto uldivmod = divmod(ul, ul2, stream);
  check_uint64_values(uldivmod[0], {2ull, 0xFFFFFFFFFull}, stream);
  check_uint64_values(uldivmod[1], {1ull, 15ull}, stream);
}

TEST_CASE("float Remainder, Power, Sign, Abs, and DivMod match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Remainder takes the divisor's sign; values are exactly
  // representable so the host fmod reference is bit-comparable.
  std::vector<float> av = {5.5f, -5.5f, 5.5f, -5.5f, 7.25f};
  std::vector<float> bv = {2.0f, 2.0f, -2.0f, -2.0f, 1.0f};
  array a(av.begin(), Shape{5}, float32);
  array b(bv.begin(), Shape{5}, float32);
  std::vector<float> mod_expected(5);
  std::vector<float> quot_expected(5);
  for (size_t i = 0; i < 5; ++i) {
    mod_expected[i] = host_float_mod(av[i], bv[i]);
    quot_expected[i] = std::floor(av[i] / bv[i]);
  }
  check_values(remainder(a, b, stream), mod_expected, stream, 1e-6);

  // Float divmod: floor-division quotient plus Python-style remainder.
  auto quot_and_rem = divmod(a, b, stream);
  REQUIRE_EQ(quot_and_rem.size(), 2u);
  check_values(quot_and_rem[0], quot_expected, stream, 1e-6);
  check_values(quot_and_rem[1], mod_expected, stream, 1e-6);

  // Power: positive bases against std::pow, negative bases with
  // integer and non-integer exponents, and a negative exponent.
  std::vector<float> pv = {2.0f, 3.0f, -2.0f, -2.0f, -8.0f, 0.5f};
  std::vector<float> qv = {0.5f, 4.0f, 3.0f, 2.0f, 1.0f / 3.0f, -2.0f};
  array p(pv.begin(), Shape{6}, float32);
  array q(qv.begin(), Shape{6}, float32);
  array pow_out = power(p, q, stream);
  pow_out.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* pow_values = pow_out.data<float>();
  CHECK(pow_values[0] == doctest::Approx(std::pow(2.0f, 0.5f)).epsilon(1e-5));
  CHECK(pow_values[1] == doctest::Approx(81.0f).epsilon(1e-5));
  CHECK_EQ(pow_values[2], -8.0f);
  CHECK_EQ(pow_values[3], 4.0f);
  // A negative base with a non-integer exponent is NaN, the way the
  // upstream std::pow reference is.
  CHECK(std::isnan(pow_values[4]));
  CHECK(pow_values[5] == doctest::Approx(4.0f).epsilon(1e-5));

  // Sign: NaN compares false on both sides and maps to zero, exactly
  // like the host comparator formula.
  std::vector<float> sv = {-3.5f, 0.0f, 2.5f, std::nanf("")};
  array s(sv.begin(), Shape{4}, float32);
  std::vector<float> sign_expected(4);
  for (size_t i = 0; i < 4; ++i) {
    sign_expected[i] = host_sign(sv[i]);
  }
  check_values(sign(s, stream), sign_expected, stream, 1e-6);

  // Abs clears the sign bit: negative zero flips to positive and NaN
  // stays NaN.
  std::vector<float> wv = {-2.5f, 0.0f, -0.0f, std::nanf("")};
  array w(wv.begin(), Shape{4}, float32);
  array abs_out = abs(w, stream);
  abs_out.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* abs_values = abs_out.data<float>();
  CHECK_EQ(abs_values[0], 2.5f);
  CHECK_EQ(abs_values[1], 0.0f);
  CHECK_EQ(abs_values[2], 0.0f);
  CHECK(std::signbit(abs_values[2]) == false);
  CHECK(std::isnan(abs_values[3]));


  // Half-precision storage keeps its variant; checks read back through
  // a float32 cast the way the existing FP16 tests do.
  std::vector<float> hv = {1.5f, -2.0f};
  array h(hv.begin(), Shape{2}, float16);
  check_values(
      astype(remainder(h, array(1.0f, float16), stream), float32, stream),
      {0.5f, 0.0f},
      stream,
      1e-6);
  check_values(
      astype(power(array(2.0f, float16), array(3.0f, float16), stream), float32, stream),
      {8.0f},
      stream,
      1e-6);
  check_values(
      astype(sign(h, stream), float32, stream), {1.0f, -1.0f}, stream, 1e-6);
  array bf(hv.begin(), Shape{2}, bfloat16);
  check_values(astype(abs(bf, stream), float32, stream), {1.5f, 2.0f}, stream, 1e-6);
}

TEST_CASE("Power keeps the host libm zero-base and integral contract") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // The Power vjp forms a^(b-1): d/dx (x-y)^x at x = 1, y = 1 is
  // b * a^(b-1) = 1 * 0^0, so the whole gradient depends on pow
  // delivering the IEEE 0^0 = 1 the host libm computes. Raw GLSL pow
  // maps 0^0 to 0 on this backend and the gradient collapsed to 0.
  array one(1.0f, float32);
  auto zero_fun = [&](const std::vector<array>& inputs) {
    return power(
        subtract(inputs[0], inputs[1], stream), inputs[0], stream);
  };
  auto [zero_value, zero_grads] =
      value_and_grad(zero_fun, std::vector<int>{0})({one, one});
  zero_grads.at(0).eval();
  omarchy::get_command_encoder(stream).synchronize();
  CHECK_EQ(zero_grads.at(0).data<float>()[0], 1.0f);

  // Integral exponents are exact wherever the result is exactly
  // representable: raw GLSL pow returned 3.0000002 for 3^1 and
  // 124.99998 for 5^3, errors the vjp multiplies straight into every
  // integer-power gradient.
  std::vector<float> pv = {3.0f, 5.0f, 2.0f, 7.5f, 1.0f, 0.0f};
  std::vector<float> qv = {1.0f, 3.0f, 11.0f, 2.0f, -4.0f, 0.0f};
  array p(pv.begin(), Shape{6}, float32);
  array q(qv.begin(), Shape{6}, float32);
  array pow_out = power(p, q, stream);
  pow_out.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* pow_values = pow_out.data<float>();
  CHECK_EQ(pow_values[0], 3.0f);
  CHECK_EQ(pow_values[1], 125.0f);
  CHECK_EQ(pow_values[2], 2048.0f);
  CHECK_EQ(pow_values[3], 56.25f);
  CHECK_EQ(pow_values[4], 1.0f);
  CHECK_EQ(pow_values[5], 1.0f);
}

TEST_CASE("int8 comparisons read one byte per element") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // A comparison dispatched to a wider kernel reads past the int8
  // buffer into recycled pages; isolated probes on fresh zero pages
  // passed while in-context gguf and scatter compares failed. Warm the
  // allocator with non-zero float words first so a wrong-width read
  // cannot pass by luck.
  {
    array warm = full({256}, 1.5f, float32, stream);
    warm.eval();
    omarchy::get_command_encoder(stream).synchronize();
  }
  std::vector<int8_t> av = {-128, -1, 0, 1, 2, 3, 100, 127, 5, -5, 7, -7, 9, 11, 13, 127};
  std::vector<int8_t> bv = av;
  bv[3] = 2;
  bv[13] = -11;
  array a(av.begin(), Shape{16}, int8);
  array b(bv.begin(), Shape{16}, int8);
  std::vector<uint8_t> expected_eq(16, 1);
  expected_eq[3] = 0;
  expected_eq[13] = 0;
  array eq = equal(a, b, stream);
  eq.eval();
  omarchy::get_command_encoder(stream).synchronize();
  for (int i = 0; i < 16; ++i) {
    INFO("index ", i);
    CHECK_EQ(static_cast<int>(eq.data<bool>()[i]), static_cast<int>(expected_eq[i]));
  }
  array lt = less(a, b, stream);
  lt.eval();
  omarchy::get_command_encoder(stream).synchronize();
  for (int i = 0; i < 16; ++i) {
    INFO("index ", i);
    CHECK_EQ(static_cast<int>(lt.data<bool>()[i]), static_cast<int>(av[i] < bv[i]));
  }
  CHECK(all(equal(a, a, stream), stream).item<bool>());
}
