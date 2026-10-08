// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 4 reduction and scan coverage: integer dtypes for Sum/Prod/Min/Max,
// Any/All, arbitrary-axis reduction, the full cumsum/cumprod/cummax/cummin
// flag matrix, and Hadamard. Every value test runs against a host reference
// or exact constants; the empty-reduction identities follow the upstream
// Limits<> seeds in mlx/backend/cpu/reduce.cpp.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
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

void sync(const Stream& stream) {
  omarchy::get_command_encoder(stream).synchronize();
}

void check_values(
    array value,
    const std::vector<float>& expected,
    const Stream& stream,
    double epsilon = 1e-5) {
  value.eval();
  sync(stream);
  REQUIRE_EQ(value.size(), expected.size());
  const float* values = value.data<float>();
  for (size_t index = 0; index < expected.size(); ++index) {
    if (std::isinf(expected[index]) || std::isnan(expected[index]) ||
        std::isinf(values[index]) || std::isnan(values[index])) {
      // doctest::Approx cannot compare infinities; exact equality is the
      // right check for inf and NaN identities.
      CHECK_EQ(values[index], expected[index]);
    } else {
      CHECK(values[index] == doctest::Approx(expected[index]).epsilon(epsilon));
    }
  }
}

void check_int32_values(
    array value,
    const std::vector<int32_t>& expected,
    const Stream& stream) {
  value.eval();
  sync(stream);
  REQUIRE_EQ(value.dtype(), int32);
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
  sync(stream);
  REQUIRE_EQ(value.dtype(), uint32);
  REQUIRE_EQ(value.size(), expected.size());
  const uint32_t* values = value.data<uint32_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_int8_values(
    array value,
    const std::vector<int8_t>& expected,
    const Stream& stream) {
  value.eval();
  sync(stream);
  REQUIRE_EQ(value.dtype(), int8);
  REQUIRE_EQ(value.size(), expected.size());
  const int8_t* values = value.data<int8_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_uint8_values(
    array value,
    const std::vector<uint8_t>& expected,
    const Stream& stream) {
  value.eval();
  sync(stream);
  REQUIRE_EQ(value.dtype(), uint8);
  REQUIRE_EQ(value.size(), expected.size());
  const uint8_t* values = value.data<uint8_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_int64_values(
    array value,
    const std::vector<int64_t>& expected,
    const Stream& stream) {
  value.eval();
  sync(stream);
  REQUIRE_EQ(value.dtype(), int64);
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
  sync(stream);
  REQUIRE_EQ(value.dtype(), uint64);
  REQUIRE_EQ(value.size(), expected.size());
  const uint64_t* values = value.data<uint64_t>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
  }
}

void check_bool_values(
    array value,
    const std::vector<bool>& expected,
    const Stream& stream) {
  value.eval();
  sync(stream);
  REQUIRE_EQ(value.dtype(), bool_);
  REQUIRE_EQ(value.size(), expected.size());
  const bool* values = value.data<bool>();
  for (size_t index = 0; index < expected.size(); ++index) {
    CHECK_EQ(values[index], expected[index]);
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

// Host reference for one scan family: op 0 sum, 1 prod, 2 min, 3 max.
std::vector<float> host_scan(
    const std::vector<float>& in,
    size_t lines,
    size_t length,
    int op,
    bool reverse,
    bool inclusive) {
  std::vector<float> out(in.size());
  for (size_t line = 0; line < lines; ++line) {
    float acc = op == 0 ? 0.0f
        : op == 1     ? 1.0f
        : op == 2     ? std::numeric_limits<float>::infinity()
                      : -std::numeric_limits<float>::infinity();
    for (size_t t = 0; t < length; ++t) {
      size_t j = reverse ? length - 1 - t : t;
      float v = in[line * length + j];
      float next = op == 0   ? acc + v
          : op == 1          ? acc * v
          : op == 2          ? std::min(acc, v)
                             : std::max(acc, v);
      if (inclusive) {
        acc = next;
        out[line * length + j] = acc;
      } else {
        out[line * length + j] = acc;
        acc = next;
      }
    }
  }
  return out;
}

// Independent Hadamard reference: the transform is the Kronecker product
// H_m (x) W_2^k applied to each row, times the scale, where W_2 is the
// unnormalized [[1,1],[1,-1]].
std::vector<float> kron(
    const std::vector<float>& a,
    int am,
    int an,
    const std::vector<float>& b,
    int bm,
    int bn) {
  std::vector<float> out(static_cast<size_t>(am * bm * an * bn));
  for (int i = 0; i < am; ++i) {
    for (int j = 0; j < an; ++j) {
      for (int k = 0; k < bm; ++k) {
        for (int l = 0; l < bn; ++l) {
          out[static_cast<size_t>((i * bm + k) * (an * bn) + j * bn + l)] =
              a[static_cast<size_t>(i * an + j)] *
              b[static_cast<size_t>(k * bn + l)];
        }
      }
    }
  }
  return out;
}

std::vector<float> sylvester_power(int k) {
  std::vector<float> w{1.0f, 1.0f, 1.0f, -1.0f};
  std::vector<float> result{1.0f};
  int size = 1;
  for (int step = 0; step < k; ++step) {
    result = kron(result, size, size, w, 2, 2);
    size *= 2;
  }
  return result;
}

// The h12 matrix, parsed from the same string mlx/backend/common/hadamard.h
// embeds, so no hand transcription can drift. '+' is +1.
std::vector<float> hadamard_12() {
  static const char* rows[12] = {
      "+-++++++++++", "--+-+-+-+-+-", "+++-++----++", "+---+--+-++-",
      "+++++-++----", "+-+---+--+-+", "++--+++-++--", "+--++---+--+",
      "++----+++-++", "+--+-++---+-", "++++----+++-", "+-+--+-++---"};
  std::vector<float> matrix;
  for (int j = 0; j < 12; ++j) {
    for (int k = 0; k < 12; ++k) {
      matrix.push_back(rows[j][k] == '+' ? 1.0f : -1.0f);
    }
  }
  REQUIRE_EQ(matrix.size(), 144u);
  return matrix;
}

std::vector<float> hadamard_reference(
    const std::vector<float>& x,
    int rows,
    int n,
    int m,
    float scale) {
  std::vector<float> w = sylvester_power(n == 1 ? 0 : [&] {
    int k = 0;
    for (int v = n; v > 1; v /= 2) {
      ++k;
    }
    return k;
  }());
  std::vector<float> h;
  if (m == 1) {
    h = {1.0f};
  } else {
    h = hadamard_12();
  }
  std::vector<float> full = kron(h, m, m, w, n, n);
  std::vector<float> out(static_cast<size_t>(rows) * m * n);
  for (int row = 0; row < rows; ++row) {
    for (int i = 0; i < m * n; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < m * n; ++j) {
        sum += full[static_cast<size_t>(i * (m * n) + j)] *
            x[static_cast<size_t>(row) * m * n + j];
      }
      out[static_cast<size_t>(row) * m * n + i] = sum * scale;
    }
  }
  return out;
}

} // namespace

TEST_CASE("int32 Sum, Prod, Min, Max wrap exactly like the CPU contract") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Suffix reduction through the fast path stays exact for integers.
  array suffix(
      {1, 2, 3, 4, 5, 6, 7, 8}, {2, 2, 2}, int32);
  check_int32_values(
      sum(suffix, std::vector<int>{1, 2}, false, stream), {10, 26}, stream);
  check_int32_values(
      prod(suffix, std::vector<int>{1, 2}, false, stream), {24, 1680}, stream);
  check_int32_values(
      min(suffix, std::vector<int>{1, 2}, false, stream), {1, 5}, stream);
  check_int32_values(
      max(suffix, std::vector<int>{1, 2}, false, stream), {4, 8}, stream);

  // Overflow-adjacent sums: upstream accumulates int32 with wraparound.
  array edge({2147483647, 1}, {2}, int32);
  check_int32_values(
      sum(edge, std::vector<int>{0}, false, stream), {-2147483648}, stream);
  array triple({2147483647, 1, -2147483647 - 1}, {3}, int32);
  check_int32_values(
      sum(triple, std::vector<int>{0}, false, stream), {0}, stream);

  // Overflow-adjacent products: 2^16 * 2^16 wraps to 0.
  array square({65536, 65536}, {2}, int32);
  check_int32_values(
      prod(square, std::vector<int>{0}, false, stream), {0}, stream);
  array mixed({3, -5, 7}, {3}, int32);
  check_int32_values(
      prod(mixed, std::vector<int>{0}, false, stream), {-105}, stream);

  // Extremes participate in min and max.
  array extremes({0, -2147483647 - 1, 2147483647, 5}, {4}, int32);
  check_int32_values(
      min(extremes, std::vector<int>{0}, false, stream), {-2147483648},
      stream);
  check_int32_values(
      max(extremes, std::vector<int>{0}, false, stream), {2147483647},
      stream);
}

TEST_CASE("uint32 Sum, Prod, Min, Max order and wrap unsigned") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array values(
      {0u, 1u, 4294967295u, 2147483648u, 7u, 42u}, {2, 3}, uint32);

  check_uint32_values(
      sum(values, std::vector<int>{1}, false, stream),
      {0u, 2147483697u},
      stream);
  check_uint32_values(
      prod(values, std::vector<int>{1}, false, stream),
      {0u, 2147483648u * 7u * 42u % 4294967296u}, stream);
  check_uint32_values(
      min(values, std::vector<int>{1}, false, stream), {0u, 7u}, stream);
  check_uint32_values(
      max(values, std::vector<int>{1}, false, stream),
      {4294967295u, 2147483648u}, stream);

  array wrap({4294967295u, 1u}, {2}, uint32);
  check_uint32_values(
      sum(wrap, std::vector<int>{0}, false, stream), {0u}, stream);
  check_uint32_values(
      prod(wrap, std::vector<int>{0}, false, stream), {4294967295u}, stream);
}

TEST_CASE("integer reductions over leading and middle axes match host loops") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // (2, 3, 4) int32 with values 1..24.
  std::vector<int32_t> raw(24);
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<int32_t>(i) + 1;
  }
  array grid(raw.begin(), Shape{2, 3, 4}, int32);

  // Axis 0 (leading): out[j][k] = sum_i (12*i + 4*j + k + 1).
  std::vector<int32_t> axis0;
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 4; ++k) {
      axis0.push_back(12 + 2 * (4 * j + k + 1));
    }
  }
  check_int32_values(
      sum(grid, std::vector<int>{0}, false, stream), axis0, stream);
  check_int32_values(
      sum(grid, std::vector<int>{0}, true, stream), axis0, stream);
  CHECK_EQ(
      sum(grid, std::vector<int>{0}, true, stream).shape(),
      Shape{1, 3, 4});

  // Axis 1 (middle): out[i][k] = sum_j (12*i + 4*j + k + 1).
  std::vector<int32_t> axis1;
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 4; ++k) {
      axis1.push_back(12 + 3 * (12 * i + k + 1));
    }
  }
  check_int32_values(
      sum(grid, std::vector<int>{1}, false, stream), axis1, stream);

  // Axis 2 stays exact through the general kernel for integers.
  std::vector<int32_t> axis2;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      int base = 12 * i + 4 * j;
      axis2.push_back(4 * base + 10);
    }
  }
  check_int32_values(
      sum(grid, std::vector<int>{2}, false, stream), axis2, stream);

  // uint32 over a leading axis.
  std::vector<uint32_t> uraw(24);
  for (size_t i = 0; i < uraw.size(); ++i) {
    uraw[i] = static_cast<uint32_t>(i) + 1;
  }
  array ugrid(uraw.begin(), Shape{2, 3, 4}, uint32);
  std::vector<uint32_t> uaxis0;
  for (int j = 0; j < 3; ++j) {
    for (int k = 0; k < 4; ++k) {
      uaxis0.push_back(static_cast<uint32_t>(12 + 2 * (4 * j + k + 1)));
    }
  }
  check_uint32_values(
      sum(ugrid, std::vector<int>{0}, false, stream), uaxis0, stream);
}

TEST_CASE("rank-4 float reductions hit every axis position and keepdims") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // (2, 3, 4, 5) with values 1..120.
  std::vector<float> raw(120);
  for (size_t i = 0; i < raw.size(); ++i) {
    raw[i] = static_cast<float>(i) + 1.0f;
  }
  array grid(raw.begin(), Shape{2, 3, 4, 5}, float32);
  auto at = [](int i, int j, int k, int l) {
    return static_cast<float>(((i * 3 + j) * 4 + k) * 5 + l + 1);
  };

  for (int axis = 0; axis < 4; ++axis) {
    std::vector<float> expected;
    int di[4] = {2, 3, 4, 5};
    di[axis] = 1;
    for (int i = 0; i < di[0]; ++i) {
      for (int j = 0; j < di[1]; ++j) {
        for (int k = 0; k < di[2]; ++k) {
          for (int l = 0; l < di[3]; ++l) {
            float total = 0.0f;
            int extent = axis == 0 ? 2 : axis == 1 ? 3 : axis == 2 ? 4 : 5;
            for (int t = 0; t < extent; ++t) {
              int r[4] = {i, j, k, l};
              r[axis] = t;
              total += at(r[0], r[1], r[2], r[3]);
            }
            expected.push_back(total);
          }
        }
      }
    }
    array reduced = sum(grid, std::vector<int>{axis}, false, stream);
    check_values(reduced, expected, stream);
    array kept = sum(grid, std::vector<int>{axis}, true, stream);
    Shape kept_shape = grid.shape();
    kept_shape[axis] = 1;
    CHECK_EQ(kept.shape(), kept_shape);
    check_values(kept, expected, stream);
  }

  // A non-suffix pair of axes in one pass.
  std::vector<float> pair_expected;
  for (int j = 0; j < 3; ++j) {
    for (int l = 0; l < 5; ++l) {
      float total = 0.0f;
      for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
          total += at(i, j, k, l);
        }
      }
      pair_expected.push_back(total);
    }
  }
  check_values(
      sum(grid, std::vector<int>{0, 2}, false, stream), pair_expected, stream);

  // Float Prod and Min over a leading axis route through the general
  // kernel too.
  std::vector<float> prod_expected;
  for (int j = 0; j < 3; ++j) {
    for (int l = 0; l < 5; ++l) {
      float total = 1.0f;
      for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 4; ++k) {
          total *= at(i, j, k, l);
        }
      }
      prod_expected.push_back(total);
    }
  }
  check_values(
      prod(grid, std::vector<int>{0, 2}, false, stream), prod_expected,
      stream);
}

TEST_CASE("float reductions over strided views match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array base(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
       12.0f},
      {3, 4},
      float32);
  array flipped = transpose(base, stream);

  // Sum over axis 1 of the transpose (shape 4x3): one value per i, each
  // walking the physically strided column of the base.
  std::vector<float> column_sums;
  for (int i = 0; i < 4; ++i) {
    float total = 0.0f;
    for (int j = 0; j < 3; ++j) {
      total += static_cast<float>(j * 4 + i + 1);
    }
    column_sums.push_back(total);
  }
  check_values(
      sum(flipped, std::vector<int>{1}, false, stream), column_sums, stream);

  // A sliced view: rows 1..2, columns 1..3.
  array window = slice(base, {1, 1}, {3, 4}, {1, 1}, stream);
  std::vector<float> window_expected;
  for (int j = 1; j < 4; ++j) {
    float total = 0.0f;
    for (int i = 1; i < 3; ++i) {
      total += static_cast<float>(i * 4 + j + 1);
    }
    window_expected.push_back(total);
  }
  check_values(
      sum(window, std::vector<int>{0}, false, stream), window_expected,
      stream);
}

TEST_CASE("empty reductions return the upstream identity values") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  array floats = zeros({2, 0, 3}, float32, stream);
  // Upstream throws in the op layer at construction time for min/max over
  // a size-0 axis (op_without_identity). Sum is 0 and prod is 1.
  auto construction_error = [](auto&& build) -> std::string {
    try {
      build();
    } catch (const std::exception& error) {
      return error.what();
    }
    return {};
  };
  CHECK(construction_error(
            [&] { return min(floats, std::vector<int>{1}, false, stream); })
            .find("Cannot min reduce") != std::string::npos);
  CHECK(construction_error(
            [&] { return max(floats, std::vector<int>{1}, false, stream); })
            .find("Cannot max reduce") != std::string::npos);

  // Integer identities: sum 0, prod 1.
  array ints = zeros({2, 0, 3}, int32, stream);
  check_int32_values(
      sum(ints, std::vector<int>{1}, false, stream),
      {0, 0, 0, 0, 0, 0}, stream);
  check_int32_values(
      prod(ints, std::vector<int>{1}, false, stream),
      {1, 1, 1, 1, 1, 1}, stream);
  CHECK(construction_error(
            [&] { return min(ints, std::vector<int>{1}, false, stream); })
            .find("Cannot min reduce") != std::string::npos);
  CHECK(construction_error(
            [&] { return max(ints, std::vector<int>{1}, false, stream); })
            .find("Cannot max reduce") != std::string::npos);
  array uints = zeros({2, 0, 3}, uint32, stream);
  CHECK(construction_error(
            [&] { return min(uints, std::vector<int>{1}, false, stream); })
            .find("Cannot min reduce") != std::string::npos);
  CHECK(construction_error(
            [&] { return max(uints, std::vector<int>{1}, false, stream); })
            .find("Cannot max reduce") != std::string::npos);
  // Upstream #4546 class: unsigned sum/prod identities over empty axes must
  // not abort for a missing init kernel. The general reduce path covers
  // every width; pin uint32 and uint64 explicitly.
  check_uint32_values(
      sum(uints, std::vector<int>{1}, false, stream),
      {0, 0, 0, 0, 0, 0}, stream);
  check_uint32_values(
      prod(uints, std::vector<int>{1}, false, stream),
      {1, 1, 1, 1, 1, 1}, stream);
  array uints64 = zeros({2, 0, 3}, uint64, stream);
  check_uint64_values(
      sum(uints64, std::vector<int>{1}, false, stream),
      {0, 0, 0, 0, 0, 0}, stream);
  check_uint64_values(
      prod(uints64, std::vector<int>{1}, false, stream),
      {1, 1, 1, 1, 1, 1}, stream);

  // Any is false and all is true over an empty axis.
  array flags = zeros({2, 0, 3}, bool_, stream);
  check_bool_values(
      any(flags, std::vector<int>{1}, false, stream),
      {false, false, false, false, false, false}, stream);
  check_bool_values(
      all(flags, std::vector<int>{1}, false, stream),
      {true, true, true, true, true, true}, stream);

  // keepdims keeps the reduced axis at size 1.
  CHECK_EQ(sum(floats, std::vector<int>{1}, true, stream).shape(),
           Shape{2, 1, 3});
  check_values(
      sum(floats, std::vector<int>{1}, true, stream),
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, stream);
}

TEST_CASE("any and all reduce bool, integer, and float truthiness") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  std::vector<bool> flag_values = {true, false, true, false};
  array flags(flag_values.begin(), Shape{4}, bool_);
  check_bool_values(
      any(flags, std::vector<int>{0}, false, stream), {true}, stream);
  check_bool_values(
      all(flags, std::vector<int>{0}, false, stream), {false}, stream);
  std::vector<bool> none_values = {false, false};
  array none(none_values.begin(), Shape{2}, bool_);
  check_bool_values(
      any(none, std::vector<int>{0}, false, stream), {false}, stream);
  std::vector<bool> every_values = {true, true, true};
  array every(every_values.begin(), Shape{3}, bool_);
  check_bool_values(
      all(every, std::vector<int>{0}, false, stream), {true}, stream);

  // Non-word-multiple sizes exercise the partial-word bool write.
  std::vector<bool> five_values = {false, true, false, false, true};
  array five(five_values.begin(), Shape{5}, bool_);
  check_bool_values(
      any(five, std::vector<int>{0}, false, stream), {true}, stream);
  check_bool_values(
      all(five, std::vector<int>{0}, false, stream), {false}, stream);

  // Integer truthiness includes negative values.
  array ints({0, -3, 0}, {3}, int32);
  check_bool_values(
      any(ints, std::vector<int>{0}, false, stream), {true}, stream);
  check_bool_values(
      all(ints, std::vector<int>{0}, false, stream), {false}, stream);
  array uints({0u, 7u}, {2}, uint32);
  check_bool_values(
      any(uints, std::vector<int>{0}, false, stream), {true}, stream);

  // Float truthiness: -0.0 is zero, NaN is truthy.
  array zeros({0.0f, -0.0f}, {2}, float32);
  check_bool_values(
      any(zeros, std::vector<int>{0}, false, stream), {false}, stream);
  array nan_mixed({0.0f, std::nanf("")}, {2}, float32);
  check_bool_values(
      any(nan_mixed, std::vector<int>{0}, false, stream), {true}, stream);

  // Over a middle axis with keepdims.
  std::vector<bool> grid_values = {true, false, false, true};
  array grid(grid_values.begin(), Shape{2, 2}, bool_);
  check_bool_values(
      any(grid, std::vector<int>{1}, true, stream), {true, true}, stream);
}

TEST_CASE("bool Min, Max, and Prod follow the upstream logical mapping") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Upstream maps bool max to any and bool min to all, each
  // returning bool (mlx/backend/metal/reduce.cpp remap_reduce_types).
  // Host reference: elementwise or / and over the reduced axis.
  // Bool prod keeps the upstream int32 output (mlx/ops.cpp) and runs
  // the existing bool-numeric path; its values are 0/1 all-products.
  std::vector<bool> values = {true, true, false, true, true, true};
  array x(values.begin(), Shape{2, 3}, bool_);
  check_bool_values(
      max(x, std::vector<int>{0}, false, stream), {true, true, true},
      stream);
  check_bool_values(
      max(x, std::vector<int>{1}, false, stream), {true, true}, stream);
  check_bool_values(
      min(x, std::vector<int>{0}, false, stream), {true, true, false},
      stream);
  check_bool_values(
      min(x, std::vector<int>{1}, false, stream), {false, true}, stream);
  check_int32_values(
      prod(x, std::vector<int>{0}, false, stream), {1, 1, 0}, stream);
  check_int32_values(
      prod(x, std::vector<int>{1}, false, stream), {0, 1}, stream);

  // All-axes reduces over a non-word-multiple length: a kernel that
  // reads a falsy byte past the packed word boundary flips any/all.
  std::vector<bool> five_values = {false, true, false, true, true};
  array five(five_values.begin(), Shape{5}, bool_);
  check_bool_values(max(five, false, stream), {true}, stream);
  check_bool_values(min(five, false, stream), {false}, stream);
  check_int32_values(prod(five, false, stream), {0}, stream);

  // The min/max output dtype is bool, not the int32 of a bool sum.
  array reduced = min(x, std::vector<int>{0}, false, stream);
  reduced.eval();
  sync(stream);
  CHECK_EQ(reduced.dtype(), bool_);
}

TEST_CASE("bool Any and All stay exact across word and chunk boundaries") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();

  // Sizes straddle every packed-bool word boundary: 4/5 is the first
  // second-word read, 8/9 and 32/33 the next two, 64/65 and 256/257
  // later words, 4096/4097 the chunk split into the scratch and
  // combine phases. All-true All and single-true Any past word 0 are
  // the discriminating polarities: a kernel that reads a falsy byte
  // past word 0 flips All and drops Any exactly here, while sizes
  // 1-4 still pass.
  const std::vector<size_t> sizes = {
      4, 5, 8, 9, 32, 33, 64, 65, 256, 257, 4096, 4097};
  auto host_all = [](const std::vector<bool>& values) {
    bool result = true;
    for (bool value : values) {
      result = result && value;
    }
    return result;
  };
  auto host_any = [](const std::vector<bool>& values) {
    bool result = false;
    for (bool value : values) {
      result = result || value;
    }
    return result;
  };
  for (size_t n : sizes) {
    int dim = static_cast<int>(n);
    std::vector<bool> ones(n, true);
    std::vector<bool> zeros(n, false);
    // One false inside the first word must always be honoured.
    std::vector<bool> false_in_word0(n, true);
    false_in_word0[1] = false;
    // One false in the first byte of the second and third words: the
    // exact reads that miscompiles drop.
    std::vector<bool> false_outside(n, true);
    if (n >= 5) {
      false_outside[4] = false;
    }
    if (n >= 9) {
      false_outside[8] = false;
    }
    // Single true in the first byte of the second word: Any must find
    // it; a falsy read past word 0 turns this false.
    std::vector<bool> true_outside(n, false);
    if (n >= 5) {
      true_outside[4] = true;
    }

    array on(ones.begin(), Shape{dim}, bool_);
    check_bool_values(all(on, std::vector<int>{0}, false, stream), {true}, stream);
    check_bool_values(any(on, std::vector<int>{0}, false, stream), {true}, stream);
    array off(zeros.begin(), Shape{dim}, bool_);
    check_bool_values(all(off, std::vector<int>{0}, false, stream), {false}, stream);
    check_bool_values(any(off, std::vector<int>{0}, false, stream), {false}, stream);

    array in_word0(false_in_word0.begin(), Shape{dim}, bool_);
    check_bool_values(
        all(in_word0, std::vector<int>{0}, false, stream),
        {host_all(false_in_word0)}, stream);
    check_bool_values(
        any(in_word0, std::vector<int>{0}, false, stream),
        {host_any(false_in_word0)}, stream);
    array outside(false_outside.begin(), Shape{dim}, bool_);
    check_bool_values(
        all(outside, std::vector<int>{0}, false, stream),
        {host_all(false_outside)}, stream);
    check_bool_values(
        any(outside, std::vector<int>{0}, false, stream),
        {host_any(false_outside)}, stream);
    array lone(true_outside.begin(), Shape{dim}, bool_);
    check_bool_values(
        any(lone, std::vector<int>{0}, false, stream),
        {host_any(true_outside)}, stream);
    check_bool_values(
        all(lone, std::vector<int>{0}, false, stream),
        {host_all(true_outside)}, stream);

    // The same sizes through the axis reduce: two rows of a (2, n)
    // array reduce along axis 1. For odd n the second row starts
    // mid-word, so the kernel decodes a nonzero kept offset plus a
    // packed input index.
    std::vector<bool> row0(n, true);
    std::vector<bool> row1 = false_outside;
    std::vector<bool> grid_values = row0;
    grid_values.insert(grid_values.end(), row1.begin(), row1.end());
    array grid(grid_values.begin(), Shape{2, dim}, bool_);
    std::vector<bool> grid_all = {host_all(row0), host_all(row1)};
    std::vector<bool> grid_any = {host_any(row0), host_any(row1)};
    check_bool_values(all(grid, std::vector<int>{1}, false, stream), grid_all, stream);
    check_bool_values(any(grid, std::vector<int>{1}, false, stream), grid_any, stream);

    // Typed control: the int32 AnyAll variants carry one element per
    // word, no packing, so a regression here is bool-specific.
    std::vector<int32_t> int_values(n, 7);
    int_values[0] = 0;
    array ints(int_values.begin(), Shape{dim}, int32);
    check_bool_values(
        all(ints, std::vector<int>{0}, false, stream), {false}, stream);
    check_bool_values(
        any(ints, std::vector<int>{0}, false, stream), {true}, stream);
  }
}

TEST_CASE("float scans match host references across all flag combinations") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> row{3.0f, -1.0f, 2.0f, 0.5f, -2.0f, 4.0f};
  array x(row.begin(), Shape{6}, float32);
  const char* op_names[4] = {"sum", "prod", "min", "max"};

  for (int op = 0; op < 4; ++op) {
    for (int reverse = 0; reverse < 2; ++reverse) {
      for (int inclusive = 0; inclusive < 2; ++inclusive) {
        auto expected =
            host_scan(row, 1, 6, op, reverse != 0, inclusive != 0);
        array scanned = op == 0
            ? cumsum(x, 0, reverse != 0, inclusive != 0, stream)
            : op == 1
            ? cumprod(x, 0, reverse != 0, inclusive != 0, stream)
            : op == 2
            ? cummin(x, 0, reverse != 0, inclusive != 0, stream)
            : cummax(x, 0, reverse != 0, inclusive != 0, stream);
        INFO(
            "op=", op_names[op], " reverse=", reverse,
            " inclusive=", inclusive);
        check_values(scanned, expected, stream, 1e-6);
      }
    }
  }

  // Multi-row arrays scan independently along a middle axis.
  std::vector<float> grid_raw;
  for (int i = 0; i < 24; ++i) {
    grid_raw.push_back(
        static_cast<float>((static_cast<int>(i) * 7) % 11 - 5));
  }
  array grid(grid_raw.begin(), Shape{2, 3, 4}, float32);
  // Host reference in output memory order, exclusive variants writing the
  // accumulator before it absorbs the current element.
  std::vector<float> expected_sum(24);
  std::vector<float> expected_max(24);
  std::vector<float> expected_min(24);
  for (int i = 0; i < 2; ++i) {
    for (int k = 0; k < 4; ++k) {
      float sum_acc = 0.0f;
      // cumsum forward inclusive.
      for (int j = 0; j < 3; ++j) {
        sum_acc += grid_raw[(i * 3 + j) * 4 + k];
        expected_sum[static_cast<size_t>((i * 3 + j) * 4 + k)] = sum_acc;
      }
      // cummax reverse exclusive: running value covers later elements.
      float max_running = -std::numeric_limits<float>::infinity();
      for (int j = 2; j >= 0; --j) {
        size_t flat = static_cast<size_t>((i * 3 + j) * 4 + k);
        expected_max[flat] = max_running;
        max_running =
            std::max(max_running, grid_raw[(i * 3 + j) * 4 + k]);
      }
      // cummin forward exclusive: running value covers earlier elements.
      float min_running = std::numeric_limits<float>::infinity();
      for (int j = 0; j < 3; ++j) {
        size_t flat = static_cast<size_t>((i * 3 + j) * 4 + k);
        expected_min[flat] = min_running;
        min_running =
            std::min(min_running, grid_raw[(i * 3 + j) * 4 + k]);
      }
    }
  }
  check_values(cumsum(grid, 1, false, true, stream), expected_sum, stream, 1e-6);
  check_values(cummax(grid, 1, true, false, stream), expected_max, stream, 1e-6);
  check_values(cummin(grid, 1, false, false, stream), expected_min, stream, 1e-6);
}

TEST_CASE("int32 scans wrap and stay exact") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array x({2147483647, 1, -5}, {3}, int32);
  check_int32_values(
      cumsum(x, 0, false, true, stream),
      {2147483647, -2147483648, 2147483643}, stream);
  check_int32_values(
      cumsum(x, 0, false, false, stream), {0, 2147483647, -2147483648},
      stream);
  array values({1, 2, 3}, {3}, int32);
  check_int32_values(
      cumsum(values, 0, true, false, stream), {5, 3, 0}, stream);
  check_int32_values(
      cumprod(values, 0, true, true, stream), {6, 6, 3}, stream);
}

TEST_CASE("scans walk strided views correctly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array base(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f,
       12.0f},
      {3, 4},
      float32);
  array flipped = transpose(base, stream);

  // cumsum along axis 0 of the transpose walks the physically strided
  // columns of the base.
  std::vector<float> expected;
  float running[3] = {0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      running[j] += static_cast<float>(j * 4 + i + 1);
      expected.push_back(running[j]);
    }
  }
  check_values(cumsum(flipped, 0, false, true, stream), expected, stream);

  // A sliced window scans its own values.
  array window = slice(base, {1, 1}, {3, 4}, {1, 1}, stream);
  std::vector<float> window_expected;
  for (int i = 1; i < 3; ++i) {
    float running = 0.0f;
    for (int j = 1; j < 4; ++j) {
      running += static_cast<float>(i * 4 + j + 1);
      window_expected.push_back(running);
    }
  }
  check_values(cumsum(window, 1, false, true, stream), window_expected,
               stream);
}

TEST_CASE("hadamard matches an independent Kronecker reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Sylvester sizes 2, 4, 8 with an explicit scale.
  for (int n : {2, 4, 8}) {
    std::vector<float> raw(static_cast<size_t>(n));
    for (size_t i = 0; i < raw.size(); ++i) {
      raw[i] = static_cast<float>(i) - static_cast<float>(n) / 2.0f;
    }
    array x(raw.begin(), Shape{n}, float32);
    auto expected = hadamard_reference(raw, 1, n, 1, 2.0f);
    check_values(hadamard_transform(x, 2.0f, stream), expected, stream, 1e-5);
  }

  // A 2-D batch transforms every row.
  std::vector<float> batch_raw(16);
  for (size_t i = 0; i < batch_raw.size(); ++i) {
    batch_raw[i] = static_cast<float>(i % 7) - 3.0f;
  }
  array batch(batch_raw.begin(), Shape{2, 8}, float32);
  auto batch_expected = hadamard_reference(batch_raw, 2, 8, 1, 2.0f);
  check_values(
      hadamard_transform(batch, 2.0f, stream), batch_expected, stream, 1e-5);

  // The default scale is 1/sqrt(N).
  std::vector<float> raw8{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  array x8(raw8.begin(), Shape{8}, float32);
  auto scaled_expected = hadamard_reference(
      raw8, 1, 8, 1, 1.0f / std::sqrt(8.0f));
  check_values(hadamard_transform(x8, std::nullopt, stream), scaled_expected,
               stream, 1e-5);

  // Power of two large enough to cross multiple workgroup dispatches.
  int n = 1024;
  std::vector<float> wide_raw(static_cast<size_t>(n));
  for (size_t i = 0; i < wide_raw.size(); ++i) {
    wide_raw[i] = static_cast<float>(i % 17) - 8.0f;
  }
  array wide(wide_raw.begin(), Shape{n}, float32);
  auto wide_expected = hadamard_reference(wide_raw, 1, n, 1, 1.0f);
  check_values(
      hadamard_transform(wide, 1.0f, stream), wide_expected, stream, 1e-4);
}

TEST_CASE("size-one Hadamard honors a custom scale") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> raw{3.0f};
  array x(raw.begin(), Shape{1}, float32);
  check_values(hadamard_transform(x, 2.0f, stream), {6.0f}, stream);
}

TEST_CASE("large Hadamard rows complete every butterfly stage") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  for (int n : {16384, 32768, 65536}) {
    std::vector<float> input(static_cast<size_t>(n), 1.0f);
    array x(input.begin(), Shape{1, n}, float32);
    auto y = hadamard_transform(x, std::nullopt, stream);
    y.eval();
    sync(stream);
    REQUIRE_EQ(y.size(), static_cast<size_t>(n));
    const float* values = y.data<float>();
    CHECK(values[0] == doctest::Approx(std::sqrt(static_cast<float>(n)))
                           .epsilon(1e-5));
    float max_tail = 0.0f;
    for (int i = 1; i < n; ++i) {
      max_tail = std::max(max_tail, std::fabs(values[i]));
    }
    CHECK_EQ(max_tail, 0.0f);
  }
}

TEST_CASE("hadamard order-12 rows match the reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // N = 12: pure H_12 (n = 1).
  std::vector<float> raw12(12);
  for (size_t i = 0; i < raw12.size(); ++i) {
    raw12[i] = static_cast<float>(i) - 5.5f;
  }
  array x12(raw12.begin(), Shape{12}, float32);
  auto expected12 = hadamard_reference(raw12, 1, 1, 12, 1.0f);
  check_values(hadamard_transform(x12, 1.0f, stream), expected12, stream,
               1e-4);

  // N = 48 = 12 * 4: the mixed component transform.
  std::vector<float> raw48(48);
  for (size_t i = 0; i < raw48.size(); ++i) {
    raw48[i] = static_cast<float>(i % 9) - 4.0f;
  }
  array x48(raw48.begin(), Shape{48}, float32);
  auto expected48 = hadamard_reference(raw48, 1, 4, 12, 1.0f);
  check_values(hadamard_transform(x48, 1.0f, stream), expected48, stream,
               1e-4);
}

TEST_CASE("FP16 hadamard matches the reference within half precision") {
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
  std::vector<float> raw{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  array half = astype(array(raw.begin(), Shape{8}, float32), float16, stream);
  auto expected = hadamard_reference(raw, 1, 8, 1, 1.0f);
  check_values(
      astype(hadamard_transform(half, 1.0f, stream), float32, stream),
      expected,
      stream,
      1e-2);
}

TEST_CASE("rank-5 general reductions preserve every operation") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  Shape shape{2, 2, 2, 2, 2};
  std::vector<int32_t> values(32);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<int32_t>(i) + 1;
  }
  array x(values.begin(), shape, int32);
  std::vector<int32_t> sums;
  std::vector<int32_t> mins;
  std::vector<int32_t> maxs;
  for (int i1 = 0; i1 < 2; ++i1) {
    for (int i3 = 0; i3 < 2; ++i3) {
      int32_t total = 0;
      int32_t smallest = std::numeric_limits<int32_t>::max();
      int32_t largest = std::numeric_limits<int32_t>::min();
      for (int i0 = 0; i0 < 2; ++i0) {
        for (int i2 = 0; i2 < 2; ++i2) {
          for (int i4 = 0; i4 < 2; ++i4) {
            int index = (((i0 * 2 + i1) * 2 + i2) * 2 + i3) * 2 + i4;
            total += values[index];
            smallest = std::min(smallest, values[index]);
            largest = std::max(largest, values[index]);
          }
        }
      }
      sums.push_back(total);
      mins.push_back(smallest);
      maxs.push_back(largest);
    }
  }
  std::vector<int> axes{0, 2, 4};
  check_int32_values(sum(x, axes, false, stream), sums, stream);
  check_int32_values(min(x, axes, false, stream), mins, stream);
  check_int32_values(max(x, axes, false, stream), maxs, stream);
  array kept = sum(x, axes, true, stream);
  CHECK_EQ(kept.shape(), Shape{1, 2, 1, 2, 1});
  check_int32_values(kept, sums, stream);

  std::vector<int32_t> factors(32, 1);
  for (int i0 = 0; i0 < 2; ++i0) {
    for (int i1 = 0; i1 < 2; ++i1) {
      for (int i2 = 0; i2 < 2; ++i2) {
        for (int i3 = 0; i3 < 2; ++i3) {
          for (int i4 = 0; i4 < 2; ++i4) {
            if (i0 == 1) {
              int index = (((i0 * 2 + i1) * 2 + i2) * 2 + i3) * 2 + i4;
              factors[index] = 2;
            }
          }
        }
      }
    }
  }
  array factor_grid(factors.begin(), shape, int32);
  check_int32_values(prod(factor_grid, axes, false, stream),
                     {16, 16, 16, 16}, stream);
  const auto& capabilities = omarchy::device(0).capabilities();
  if (capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access) {
    std::vector<float> half_values(4097, 1.0f);
    half_values[4000] = 2.0f;
    array half = astype(
        array(half_values.begin(), Shape{1, 1, 1, 1, 4097}, float32),
        float16,
        stream);
    check_values(
        astype(
            prod(half, std::vector<int>{0, 1, 2, 3, 4}, false, stream),
            float32,
            stream),
        {2.0f},
        stream,
        1e-3);
  }

  std::vector<bool> flags(32, true);
  for (int i0 = 0; i0 < 2; ++i0) {
    for (int i2 = 0; i2 < 2; ++i2) {
      for (int i4 = 0; i4 < 2; ++i4) {
        int index = (((i0 * 2) * 2 + i2) * 2) * 2 + i4;
        flags[index] = false;
      }
    }
  }
  array flag_grid(flags.begin(), shape, bool_);
  check_bool_values(any(flag_grid, axes, false, stream),
                    {false, true, true, true}, stream);
  check_bool_values(all(flag_grid, axes, false, stream),
                    {false, true, true, true}, stream);
}

TEST_CASE("rank-5 strided and broadcast reductions preserve logical order") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  std::vector<float> values(48);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(i) + 1.0f;
  }
  array base(values.begin(), Shape{2, 3, 2, 2, 2}, float32);
  array view = transpose(base, {4, 1, 3, 0, 2});
  std::vector<float> expected;
  for (int i1 = 0; i1 < 3; ++i1) {
    for (int i3 = 0; i3 < 2; ++i3) {
      float total = 0.0f;
      for (int i0 = 0; i0 < 2; ++i0) {
        for (int i2 = 0; i2 < 2; ++i2) {
          for (int i4 = 0; i4 < 2; ++i4) {
            int index = (((i3 * 3 + i1) * 2 + i4) * 2 + i2) * 2 + i0;
            total += values[index];
          }
        }
      }
      expected.push_back(total);
    }
  }
  check_values(sum(view, std::vector<int>{0, 2, 4}, false, stream),
               expected, stream);

  array scalar(3, int32);
  array expanded = broadcast_to(scalar, {2, 3, 2, 2, 2});
  check_int32_values(sum(expanded, std::vector<int>{0, 2, 4}, false, stream),
                     {24, 24, 24, 24, 24, 24}, stream);
}

TEST_CASE("rank-6 empty reductions retain upstream identities") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  Shape shape{2, 1, 0, 3, 1, 2};
  std::vector<int> axes{2, 4};
  array ints = zeros(shape, int32, stream);
  check_int32_values(sum(ints, axes, false, stream),
                     std::vector<int32_t>(12, 0), stream);
  check_int32_values(prod(ints, axes, false, stream),
                     std::vector<int32_t>(12, 1), stream);
  array flags = zeros(shape, bool_, stream);
  check_bool_values(any(flags, axes, false, stream),
                    std::vector<bool>(12, false), stream);
  check_bool_values(all(flags, axes, false, stream),
                    std::vector<bool>(12, true), stream);
}
TEST_CASE("bool sum and product preserve numeric promotion across layouts") {
  if (!compute_available()) return;
  auto stream = gpu_stream();
  std::vector<uint8_t> raw{0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  array storage(raw.begin(), Shape{16}, bool_);
  array x = reshape(slice(storage, {1}, {16}, stream), {3, 5}, stream);
  check_int32_values(sum(x, 1, false, stream), {3, 0, 5}, stream);
  check_int32_values(prod(x, 1, false, stream), {0, 0, 1}, stream);
  auto transposed = transpose(x, stream);
  check_int32_values(sum(transposed, 1, false, stream), {2, 1, 2, 1, 2}, stream);
  check_int32_values(prod(transposed, 1, false, stream), {0, 0, 0, 0, 0}, stream);
  auto broadcast = broadcast_to(array(true), {2, 3, 2, 2, 2}, stream);
  check_int32_values(sum(broadcast, std::vector<int>{0, 2, 4}, false, stream),
                     std::vector<int32_t>(6, 8), stream);
  check_int32_values(prod(broadcast, std::vector<int>{0, 2, 4}, false, stream),
                     std::vector<int32_t>(6, 1), stream);
  auto empty = zeros({2, 0, 3}, bool_, stream);
  check_int32_values(sum(empty, 1, false, stream), std::vector<int32_t>(6, 0), stream);
  check_int32_values(prod(empty, 1, false, stream), std::vector<int32_t>(6, 1), stream);
  std::vector<uint8_t> large(5000, 1);
  large[4501] = 0;
  array split(large.begin(), Shape{5000}, bool_);
  check_int32_values(sum(split, 0, false, stream), {4999}, stream);
  check_int32_values(prod(split, 0, false, stream), {0}, stream);
}

TEST_CASE("out-of-scope dtypes and shapes keep their named errors") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // int64 Sum/Prod reduce in their own width behind shaderInt64.
  array wide({1, 2, 3}, {3}, int64);
  check_int64_values(
      prod(wide, std::vector<int>{0}, false, stream), {6}, stream);
  check_int32_values(
      sum(array({1, 2, 3}, {3}, int32), 0, false, stream), {6}, stream);

  // LogAddExp scans accumulate through the float log-sum-exp.
  array x({1.0f, 2.0f}, {2}, float32);
  check_values(
      logcumsumexp(x, 0, false, true, stream),
      {1.0f, 2.0f + std::log1p(std::exp(-1.0f))},
      stream,
      1e-5);
  check_int32_values(
      sum(array({4, 5, 6}, {3}, int32), 0, false, stream), {15}, stream);

  // Hadamard only supports n = m*2^k for m in (1, 12, 20, 28).
  std::vector<float> bad_values(7, 1.0f);
  array bad(bad_values.begin(), Shape{7}, float32);
  CHECK(evaluation_error(hadamard_transform(bad, std::nullopt, stream))
            .find("Hadamard size") != std::string::npos);
  check_int32_values(
      sum(array({7, 8, 9}, {3}, int32), 0, false, stream), {24}, stream);

}

TEST_CASE("logcumsumexp promotes integer and bool inputs to float (#4625)") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // Running log-add-exp over the scan direction; the promotion lands before
  // the Scan, so the accumulation is float log-space on the GPU route.
  auto running_logaddexp = [](const std::vector<float>& values) {
    std::vector<float> out;
    float acc = values.front();
    for (size_t i = 0; i < values.size(); ++i) {
      if (i > 0) {
        acc = std::max(acc, values[i]) +
            std::log1p(std::exp(-std::abs(values[i] - acc)));
      }
      out.push_back(acc);
    }
    return out;
  };

  // uint8: a non-promoted scan returns the input dtype with cumulative
  // maxima; the promoted scan returns float log-space values. Reverse
  // inclusive walks the suffix: out[i] = logaddexp(a[i..n-1]).
  std::vector<uint8_t> raw = {3, 5, 2, 7};
  array u(raw.begin(), Shape{4}, uint8);
  std::vector<float> expected = running_logaddexp({3.0f, 5.0f, 2.0f, 7.0f});
  CHECK(std::abs(expected[1] - 5.0f) > 1e-3);
  array lu = logcumsumexp(u, 0, false, true, stream);
  REQUIRE_EQ(lu.dtype(), float32);
  check_values(lu, expected, stream, 1e-5);
  // Suffix scan for {3, 5, 2, 7}: out[3]=7, out[2]=lax(2,7), out[1]=lax(5, out[2]),
  // out[0]=lax(3, out[1]).
  std::vector<float> reverse_expected = {7.0f, 0.0f, 0.0f, 0.0f};
  float acc = raw.back();
  reverse_expected[3] = acc;
  for (int i = 2; i >= 0; --i) {
    acc = std::max(acc, static_cast<float>(raw[i])) +
        std::log1p(std::exp(-std::abs(static_cast<float>(raw[i]) - acc)));
    reverse_expected[i] = acc;
  }
  check_values(
      logcumsumexp(u, 0, true, true, stream), reverse_expected, stream, 1e-5);

  // bool: same promotion through the 0/1 floats.
  std::vector<bool> raw_b = {true, false, true, true};
  array b(raw_b.begin(), Shape{4}, bool_);
  array lb = logcumsumexp(b, 0, false, true, stream);
  REQUIRE_EQ(lb.dtype(), float32);
  check_values(
      lb, running_logaddexp({1.0f, 0.0f, 1.0f, 1.0f}), stream, 1e-5);
}


TEST_CASE("int32 multi-axis sum visits every reduced element") {
  if (!compute_available()) return;
  auto stream = gpu_stream();
  // Shape (65,65,1,65): the kept axis 2 contributes one output cell per
  // combination of reduced axes. The reduction span is 65*65*65 = 274625
  // which exceeds the per-invocation serial-loop trip cap that the old
  // single-pass shader hit silently; the tiled two-phase reduction must
  // visit every element regardless of the trip-cap boundary.
  const int N0 = 65, N1 = 65, N2 = 1, N3 = 65;
  std::vector<int32_t> ones(N0 * N1 * N2 * N3, 1);
  array x(ones.begin(), Shape{N0, N1, N2, N3}, int32);

  // Three-axis sum over (0,1,3) keeps axis 2 (size 1 -> one output cell).
  array g013 = sum(x, std::vector<int>{0, 1, 3});
  g013.eval();
  sync(stream);
  REQUIRE_EQ(g013.size(), 1);
  CHECK_EQ(g013.data<int32_t>()[0], int32_t(N0 * N1 * N2 * N3));

  // Two-axis (0,1) keeps (2,3) - 65 cells, each summing 65*65 = 4225 ones.
  array g01 = sum(x, std::vector<int>{0, 1});
  g01.eval();
  sync(stream);
  REQUIRE_EQ(g01.size(), 65);
  const int32_t* g01p = g01.data<int32_t>();
  for (int i = 0; i < 65; ++i) {
    CHECK_EQ(g01p[i], 4225);
  }

  // Single-axis over axis 0 keeps (1,2,3), giving 65*1*65 = 4225 cells,
  // each summing the 65 reduced elements in axis 0. Per cell = 65.
  array g0 = sum(x, std::vector<int>{0});
  g0.eval();
  sync(stream);
  CHECK_EQ(g0.size(), N1 * N2 * N3);
  const int32_t* g0p = g0.data<int32_t>();
  for (int i = 0; i < g0.size(); ++i) {
    CHECK_EQ(g0p[i], 65);
  }
}

TEST_CASE("int multi-axis sum agrees with int64 reference across ranks") {
  if (!compute_available()) return;
  auto stream = gpu_stream();
  std::mt19937 gen(42);
  std::normal_distribution<float> dist(0.0f, 128.0f);
  for (int axis_subset_bits : {0b0111, 0b1011, 0b1101, 0b1110, 0b1111}) {
    Shape shape{8, 7, 5, 4};
    std::vector<int32_t> data(8 * 7 * 5 * 4);
    for (auto& v : data) v = int32_t(std::lround(dist(gen)));
    array x(data.begin(), shape, int32);
    std::vector<int> axes;
    for (int bit = 0; bit < 4; ++bit) {
      if (axis_subset_bits & (1 << bit)) axes.push_back(bit);
    }
    std::vector<int> kept;
    for (int d = 0; d < 4; ++d) {
      if (!(axis_subset_bits & (1 << d))) kept.push_back(d);
    }
    Shape kept_shape;
    for (int d : kept) kept_shape.push_back(shape[d]);
    if (kept_shape.empty()) kept_shape.push_back(1);
    size_t ksize = 1;
    for (int v : kept_shape) ksize *= v;
    std::vector<int64_t> want(ksize, 0);
    for (int i0 = 0; i0 < shape[0]; ++i0)
      for (int i1 = 0; i1 < shape[1]; ++i1)
        for (int i2 = 0; i2 < shape[2]; ++i2)
          for (int i3 = 0; i3 < shape[3]; ++i3) {
            std::vector<int> kc;
            for (int d : kept)
              kc.push_back(d == 0 ? i0 : d == 1 ? i1 : d == 2 ? i2 : i3);
            size_t kf = 0;
            for (int i = (int)kept.size() - 1; i >= 0; --i) {
              kf = kf * kept_shape[i] + kc[i];
            }
            want[kf] +=
                data[((i0 * shape[1] + i1) * shape[2] + i2) * shape[3] + i3];
          }
    auto got = sum(x, axes);
    got.eval();
    sync(stream);
    REQUIRE_EQ(got.size(), ksize);
    const int32_t* gp = got.data<int32_t>();
    for (size_t k = 0; k < ksize; ++k) {
      int64_t w = want[k];
      int32_t expected = int32_t(w);
      CHECK_EQ(gp[k], expected);
    }
  }
}

TEST_CASE("uint32 multi-axis sum covers the trip-cap boundary") {
  if (!compute_available()) return;
  auto stream = gpu_stream();
  std::vector<uint32_t> ones(32768, 1);
  array x(ones.begin(), Shape{32768, 1, 1, 1}, uint32);
  auto g = sum(x, std::vector<int>{0, 1, 2, 3});
  g.eval();
  sync(stream);
  REQUIRE_EQ(g.size(), 1);
  CHECK_EQ(g.data<uint32_t>()[0], 32768u);

  std::vector<uint32_t> big(65536, 1);
  array y(big.begin(), Shape{65536, 1, 1, 1}, uint32);
  auto gy = sum(y, std::vector<int>{0, 1, 2, 3});
  gy.eval();
  sync(stream);
  CHECK_EQ(gy.data<uint32_t>()[0], 65536u);
}

TEST_CASE("float non-suffix reductions propagate NaN through Min and Max") {
  if (!compute_available()) return;
  auto stream = gpu_stream();
  // Shape (2,3) with NaN at (0,1). Column-wise reduce along axis 0:
  //   col 0 = [1, 4]            -> max = 4    (no NaN)
  //   col 1 = [NaN, 5]          -> max = NaN  (NaN poisons)
  //   col 2 = [3, 6]            -> max = 6    (no NaN)
  // Row-wise reduce along axis 1: row 0 contains NaN -> NaN; row 1
  //   [4, 5, 6] -> max = 6.
  std::vector<float> data = {1.0f, NAN, 3.0f, 4.0f, 5.0f, 6.0f};
  array x(data.begin(), Shape{2, 3}, float32);

  auto m0 = max(x, 0);
  m0.eval();
  sync(stream);
  REQUIRE_EQ(m0.size(), 3);
  CHECK_EQ(m0.data<float>()[0], 4.0f);
  CHECK(std::isnan(m0.data<float>()[1]));
  CHECK_EQ(m0.data<float>()[2], 6.0f);

  auto mn0 = min(x, 0);
  mn0.eval();
  sync(stream);
  CHECK_EQ(mn0.data<float>()[0], 1.0f);
  CHECK(std::isnan(mn0.data<float>()[1]));
  CHECK_EQ(mn0.data<float>()[2], 3.0f);

  auto m1 = max(x, 1);
  m1.eval();
  sync(stream);
  REQUIRE_EQ(m1.size(), 2);
  CHECK(std::isnan(m1.data<float>()[0]));
  CHECK_EQ(m1.data<float>()[1], 6.0f);

  auto mn1 = min(x, 1);
  mn1.eval();
  sync(stream);
  CHECK(std::isnan(mn1.data<float>()[0]));
  CHECK_EQ(mn1.data<float>()[1], 4.0f);

}

TEST_CASE("cummax and cummin hold NaN once it appears") {
  if (!compute_available()) return;
  auto stream = gpu_stream();

  // Forward cummax: [1, 3, NaN, 5, 4] -> [1, 3, NaN, NaN, NaN] (sticky).
  std::vector<float> a = {1.0f, 3.0f, NAN, 5.0f, 4.0f};
  array xa(a.begin(), Shape{5}, float32);
  auto cm = cummax(xa);
  cm.eval();
  sync(stream);
  const float* p = cm.data<float>();
  CHECK_EQ(p[0], 1.0f);
  CHECK_EQ(p[1], 3.0f);
  CHECK(std::isnan(p[2]));
  CHECK(std::isnan(p[3]));
  CHECK(std::isnan(p[4]));

  // Reverse cummax: walks the line from the right. With NaN at position
  // 2, positions 2..0 hold NaN and positions 3 and 4 stay clean.
  auto cmr = cummax(xa, true);
  cmr.eval();
  sync(stream);
  const float* pr = cmr.data<float>();
  CHECK(std::isnan(pr[0]));
  CHECK(std::isnan(pr[1]));
  CHECK(std::isnan(pr[2]));
  CHECK_EQ(pr[3], 5.0f);
  CHECK_EQ(pr[4], 4.0f);

  // Cummin: [5, 4, NaN, 2, 3] -> [5, 4, NaN, NaN, NaN] (sticky).
  std::vector<float> b = {5.0f, 4.0f, NAN, 2.0f, 3.0f};
  array xb(b.begin(), Shape{5}, float32);
  auto mi = cummin(xb);
  mi.eval();
  sync(stream);
  const float* q = mi.data<float>();
  CHECK_EQ(q[0], 5.0f);
  CHECK_EQ(q[1], 4.0f);
  CHECK(std::isnan(q[2]));
  CHECK(std::isnan(q[3]));
  CHECK(std::isnan(q[4]));

  // NaN at the FIRST position: poison enters immediately.
  std::vector<float> first = {NAN, 1.0f, 2.0f};
  array xf(first.begin(), Shape{3}, float32);
  auto cmf = cummax(xf);
  cmf.eval();
  sync(stream);
  const float* pf = cmf.data<float>();
  CHECK(std::isnan(pf[0]));
  CHECK(std::isnan(pf[1]));
  CHECK(std::isnan(pf[2]));

  // NaN at the LAST position: the prior running max is unaffected.
  std::vector<float> last = {2.0f, 5.0f, 1.0f, NAN};
  array xl(last.begin(), Shape{4}, float32);
  auto cml = cummax(xl);
  cml.eval();
  sync(stream);
  const float* pl = cml.data<float>();
  CHECK_EQ(pl[0], 2.0f);
  CHECK_EQ(pl[1], 5.0f);
  CHECK_EQ(pl[2], 5.0f);
  CHECK(std::isnan(pl[3]));
}

TEST_CASE("Prod, Any, and All cover the full reduction span") {
  if (!compute_available()) return;
  auto stream = gpu_stream();
  // Prod over (65536,1,1,1) with a non-trivial 2 at position 40000
  // catches silent truncation: any visit-skip leaves the product at 1.
  std::vector<int32_t> data(65536, 1);
  data[40000] = 2;
  array x(data.begin(), Shape{65536, 1, 1, 1}, int32);
  auto g = prod(x, std::vector<int>{0, 1, 2, 3});
  g.eval();
  sync(stream);
  CHECK_EQ(g.data<int32_t>()[0], 2);

  // Any/All over (50000,1,1,1) with a single zero at position 30000.
  std::vector<uint8_t> bool_data(50000, 1);
  bool_data[30000] = 0;
  array xb(bool_data.begin(), Shape{50000, 1, 1, 1}, bool_);
  auto ga = any(xb, std::vector<int>{0, 1, 2, 3});
  ga.eval();
  sync(stream);
  REQUIRE_EQ(ga.dtype(), bool_);
  CHECK_EQ(ga.data<bool>()[0], true);
  auto gl = all(xb, std::vector<int>{0, 1, 2, 3});
  gl.eval();
  sync(stream);
  CHECK_EQ(gl.data<bool>()[0], false);
}

TEST_CASE("suffix reductions cover rows larger than the driver trip cap") {
  if (!compute_available()) return;
  auto stream = gpu_stream();
  // Rows that exceed the suffix kernel's per-invocation serial-loop
  // trip cap (roughly 2^16 on lavapipe) used to silently truncate; the
  // chunked two-phase dispatch now visits every element.
  std::vector<float> f32_ones(200000, 1.0f);
  array xf(f32_ones.begin(), Shape{200000}, float32);
  auto sf = sum(xf);
  eval(sf);
  sync(stream);
  REQUIRE_EQ(sf.size(), 1);
  CHECK_EQ(sf.data<float>()[0], 200000.0f);

  std::vector<int32_t> i32_ones(200000, 1);
  array xi(i32_ones.begin(), Shape{200000}, int32);
  auto si = sum(xi);
  eval(si);
  sync(stream);
  REQUIRE_EQ(si.size(), 1);
  CHECK_EQ(si.data<int32_t>()[0], 200000);

  // Sum of (64,64,64,1) full reduce over 262144 elements - well past the
  // old 65535 cap.
  size_t big = 64 * 64 * 64;
  std::vector<float> big_ones(big, 1.0f);
  array xb(big_ones.begin(), Shape{64, 64, 64, 1}, float32);
  auto sb = sum(xb);
  eval(sb);
  sync(stream);
  REQUIRE_EQ(sb.size(), 1);
  CHECK_EQ(sb.data<float>()[0], float(big));

  // Suffix Max NaN propagation across the chunked path: a NaN at index
 // 50000 in a 100000-long row poisons the entire row.
  std::vector<float> with_nan(100000, 5.0f);
  with_nan[50000] = NAN;
  array xm(with_nan.begin(), Shape{100000}, float32);
  auto mm = max(xm);
  eval(mm);
  sync(stream);
  CHECK(std::isnan(mm.data<float>()[0]));
}

TEST_CASE("narrow integers sum in 32 bits and keep Min/Max width") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The upstream contract accumulates narrow sums in int32: 120 + 120
  // wraps an int8 accumulator but stays exact through the int32 lane.
  std::vector<int8_t> narrow_bits = {120, 120};
  array narrow(narrow_bits.begin(), Shape{2}, int8);
  check_int32_values(sum(narrow, 0, false, stream), {240}, stream);
  std::vector<int8_t> prod_bits = {3, 5};
  array prod_vals(prod_bits.begin(), Shape{2}, int8);
  check_int32_values(prod(prod_vals, 0, false, stream), {15}, stream);
  // Min/Max keep the input width, negative values included.
  std::vector<int8_t> signed_bits = {-128, 3, 127};
  array signed_vals(signed_bits.begin(), Shape{3}, int8);
  check_int8_values(min(signed_vals, 0, false, stream), {-128}, stream);
  check_int8_values(max(signed_vals, 0, false, stream), {127}, stream);
  std::vector<uint8_t> unsigned_bits = {200, 3, 250};
  array unsigned_vals(unsigned_bits.begin(), Shape{3}, uint8);
  check_uint8_values(min(unsigned_vals, 0, false, stream), {3}, stream);
  check_uint8_values(max(unsigned_vals, 0, false, stream), {250}, stream);
  std::vector<uint8_t> sum_bits = {250, 10};
  array sum_vals(sum_bits.begin(), Shape{2}, uint8);
  check_uint32_values(sum(sum_vals, 0, false, stream), {260}, stream);
}

TEST_CASE("64-bit and complex64 reductions match host references") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<int64_t> big_bits = {4, 2, 1};
  array big(big_bits.begin(), Shape{3}, int64);
  check_int64_values(sum(big, 0, false, stream), {7}, stream);
  check_int64_values(
      max(array({-4, 9, 2}, {3}, int64), 0, false, stream), {9}, stream);
  std::vector<uint64_t> ubits = {1ull << 40u, 3};
  array uvals(ubits.begin(), Shape{2}, uint64);
  auto usum = sum(uvals, 0, false, stream);
  usum.eval();
  sync(stream);
  CHECK_EQ(usum.data<uint64_t>()[0], (1ull << 40u) + 3u);

  // complex64 Sum/Prod/Min/Max: lexicographic compare, numpy NaN rule.
  std::vector<float> ca_bits = {1.0f, 1.0f, 2.0f, -1.0f, 0.5f, 2.0f};
  array ca_flat(ca_bits.begin(), Shape{6}, float32);
  array ca = view(ca_flat, complex64, stream);
  ca.eval();
  sync(stream);
  array csum = sum(ca, 0, false, stream);
  csum.eval();
  sync(stream);
  auto packed = csum.data<float>();
  CHECK_EQ(packed[0], doctest::Approx(3.5f));
  CHECK_EQ(packed[1], doctest::Approx(2.0f));
  array cmax = max(ca, 0, false, stream);
  cmax.eval();
  sync(stream);
  auto maxp = cmax.data<float>();
  CHECK_EQ(maxp[0], doctest::Approx(2.0f));
  CHECK_EQ(maxp[1], doctest::Approx(-1.0f));
}

TEST_CASE("integer argreduces pick exact native comparisons") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array i16_vals({-300, 5, -301}, {3}, int16);
  check_uint32_values(argmin(i16_vals, 0, false, stream), {2}, stream);
  check_uint32_values(argmax(i16_vals, 0, false, stream), {1}, stream);
  array i64_vals({5, -9, 9}, {3}, int64);
  check_uint32_values(argmin(i64_vals, 0, false, stream), {1}, stream);
  check_uint32_values(argmax(i64_vals, 0, false, stream), {2}, stream);
  array u64_vals({7, 1, 3}, {3}, uint64);
  check_uint32_values(argmin(u64_vals, 0, false, stream), {1}, stream);
}

TEST_CASE("float argreduces propagate NaN like the reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  // mlx reduction semantics: NaN propagates and the FIRST NaN index wins,
  // for argmin and argmax alike (upstream "test arg reduce NaN").
  array mixed({3.0f, nan, 1.0f, 5.0f, nan, -1.0f}, {2, 3}, float32);
  check_uint32_values(argmin(mixed, 1, false, stream), {1, 1}, stream);
  check_uint32_values(argmax(mixed, 1, false, stream), {1, 1}, stream);
  std::vector<float> wide(1024, 0.0f);
  wide[17] = nan;
  wide[529] = nan;
  array wide_arr(wide.begin(), Shape{1024}, float32);
  check_uint32_values(argmin(wide_arr, 0, false, stream), {17}, stream);
  check_uint32_values(argmax(wide_arr, 0, false, stream), {17}, stream);
}

TEST_CASE("scans cover int64, complex64, and bool cumsum widths") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<int64_t> wide_bits = {1, 2, 3, -4};
  array wide(wide_bits.begin(), Shape{4}, int64);
  check_int64_values(cumsum(wide, 0, false, true, stream), {1, 3, 6, 2},
                     stream);
  check_int64_values(cumprod(wide, 0, false, true, stream), {1, 2, 6, -24},
                     stream);

  std::vector<float> ca_bits = {1.0f, 0.0f, 2.0f, 1.0f};
  array ca_flat(ca_bits.begin(), Shape{4}, float32);
  array ca = view(ca_flat, complex64, stream);
  array cc = cumsum(ca, 0, false, true, stream);
  cc.eval();
  sync(stream);
  const float* got = cc.data<float>();
  CHECK_EQ(got[0], doctest::Approx(1.0f));
  CHECK_EQ(got[1], doctest::Approx(0.0f));
  CHECK_EQ(got[2], doctest::Approx(3.0f));
  CHECK_EQ(got[3], doctest::Approx(1.0f));

  std::vector<int32_t> bbits = {1, 1, 0, 1};
  array braw(bbits.begin(), Shape{4}, int32);
  array bvals = astype(braw, bool_, stream);
  check_int32_values(
      cumsum(bvals, 0, false, true, stream), {1, 2, 2, 3}, stream);
  check_bool_values(
      cummax(bvals, 0, false, true, stream), {true, true, true, true},
      stream);
}

TEST_CASE("reductions over flip views materialize negative strides") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array base({1.0f, 2.0f, 3.0f, 4.0f}, {4}, float32);
  array flipped = flip(base, 0, stream);
  check_values(sum(flipped, 0, false, stream), {10.0f}, stream);
  check_values(max(flipped, 0, false, stream), {4.0f}, stream);
  check_values(min(flipped, 0, false, stream), {1.0f}, stream);
}
