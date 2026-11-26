// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Select (where) general-layout coverage. Every case compares device
// values element-for-element against a host-computed reference. tril and
// triu lower to where(), whose general-layout Select variant gates
// composed linalg::lu, so these cases pin the transports the composed
// chain rides on: broadcast conditions, strided and transposed operands,
// mixed operand shapes, non-contiguous conditions, offset dense views,
// word-boundary sizes, and the six value dtypes. Bool arrays are built
// through integer comparisons because the backend deliberately refuses
// dtype-converting bool copies. The composed linalg cases at the bottom
// verify lu, solve, and pinv end to end against host references.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/device.h"
#include "mlx/linalg.h"
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

bool compute_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

// Reads any device array back as float32 host values. Test values stay
// small integers and int32 magnitudes below 2^24, so every dtype
// round-trips through float32 without loss.
std::vector<float> readback_f32(const Stream& stream, array value) {
  value = astype(value, float32, stream);
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = value.data<float>();
  return std::vector<float>(data, data + value.size());
}

void expect_values(
    const std::vector<float>& device,
    const std::vector<double>& expected,
    double tol,
    const std::string& label) {
  REQUIRE_EQ(device.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    double diff = std::abs(static_cast<double>(device[i]) - expected[i]);
    CHECK_MESSAGE(
        diff <= tol + tol * std::abs(expected[i]),
        label,
        " index ",
        i,
        ": device ",
        device[i],
        " vs host ",
        expected[i]);
  }
}

// Builds a device array of any test dtype from host doubles. Integral
// dtypes construct directly from typed host data because the backend
// refuses dtype-converting copies for uint32; float16 and bfloat16 ride
// the implemented float casts. Values stay integral and small, so every
// conversion is exact.
array device_values(
    const Stream& stream,
    const std::vector<double>& values,
    Shape shape,
    Dtype dtype) {
  if (dtype == uint32) {
    std::vector<uint32_t> flat(values.begin(), values.end());
    return array(flat.begin(), shape, uint32);
  }
  if (dtype == int32) {
    std::vector<int32_t> flat(values.begin(), values.end());
    return array(flat.begin(), shape, int32);
  }
  std::vector<float> flat(values.begin(), values.end());
  array out(flat.begin(), shape, float32);
  return astype(out, dtype, stream);
}

// Flat int32 ramp reshaped to shape; the base for comparison-built bool
// patterns and for value arrays.
array index_ramp(const Stream& stream, Shape shape) {
  int64_t total = 1;
  for (int axis = 0; axis < static_cast<int>(shape.size()); ++axis) {
    total *= shape[axis];
  }
  return reshape(
      arange(0, static_cast<double>(total), 1, int32, stream), shape, stream);
}

// Dense bool mask with mask[i * n + j] = (i + j) % 2 == 1. Parity of a
// sum is bit 0 of the bitwise XOR, which is implemented for int32.
array parity_mask(const Stream& stream, int m, int n) {
  array rows = expand_dims(arange(0, m, 1, int32, stream), 1, stream);
  array cols = expand_dims(arange(0, n, 1, int32, stream), 0, stream);
  array low_bit =
      bitwise_and(bitwise_xor(rows, cols, stream), array(1, int32), stream);
  return equal(low_bit, array(1, int32), stream);
}

// Strictly diagonally dominant square matrix, nonsingular by
// construction.
std::vector<double> host_diag_dominant(std::mt19937& gen, int n) {
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> a(n * n);
  for (int i = 0; i < n; ++i) {
    double row = 0.0;
    for (int j = 0; j < n; ++j) {
      if (j == i) {
        continue;
      }
      a[i * n + j] = dist(gen);
      row += std::abs(a[i * n + j]);
    }
    a[i * n + i] = row + 1.0;
  }
  return a;
}

std::vector<double> host_matmul(
    const std::vector<double>& a,
    const std::vector<double>& b,
    int rows_a,
    int inner,
    int cols_b) {
  std::vector<double> out(rows_a * cols_b, 0.0);
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      double sum = 0.0;
      for (int l = 0; l < inner; ++l) {
        sum += a[i * inner + l] * b[l * cols_b + j];
      }
      out[i * cols_b + j] = sum;
    }
  }
  return out;
}

TEST_CASE("where picks transposed operands for the tril and triu pair") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int n = 5;
  std::mt19937 gen(7);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::vector<double> x_host(n * n);
  for (auto& value : x_host) {
    value = dist(gen);
  }
  array x = device_values(stream, x_host, Shape{n, n}, float32);
  array xt = transpose(x, std::vector<int>{1, 0}, stream);
  // mask[i][j] = j <= i, the upstream tri(k=0) predicate, built through
  // comparisons.
  array rows = expand_dims(arange(0, n, 1, int32, stream), 1, stream);
  array cols = expand_dims(arange(0, n, 1, int32, stream), 0, stream);
  array mask = less_equal(cols, rows, stream);
  // tril lowers to where(mask, x, 0): keep x below the diagonal.
  {
    std::vector<double> expected(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (j <= i) {
          // xt[i][j] = x[j][i]
          expected[i * n + j] = x_host[j * n + i];
        }
      }
    }
    auto got = readback_f32(
        stream, where(mask, xt, array(0, float32), stream));
    expect_values(got, expected, 1e-6, "tril transposed");
  }
  // triu lowers to where(mask, 0, x): keep x above the diagonal; this
  // is the scalar-truthy general-layout select that composed lu rides.
  {
    std::vector<double> expected(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        expected[i * n + j] = x_host[j * n + i];
      }
    }
    auto got = readback_f32(
        stream, where(mask, array(0, float32), xt, stream));
    expect_values(got, expected, 1e-6, "triu transposed");
  }
  // Dense operands stay on the flat transport; the same masks pin the
  // regression path.
  {
    std::vector<double> lower(n * n, 0.0);
    std::vector<double> upper(n * n, 0.0);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (j <= i) {
          lower[i * n + j] = x_host[i * n + j];
        } else {
          upper[i * n + j] = x_host[i * n + j];
        }
      }
    }
    auto got_lower = readback_f32(
        stream, where(mask, x, array(0, float32), stream));
    expect_values(got_lower, lower, 1e-6, "tril dense");
    auto got_upper = readback_f32(
        stream, where(mask, array(0, float32), x, stream));
    expect_values(got_upper, upper, 1e-6, "triu dense");
  }
}

TEST_CASE("where with a broadcast condition") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int m = 9;
  int n = 129;
  std::mt19937 gen(11);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::vector<double> x_host(m * n);
  std::vector<double> y_host(m * n);
  for (auto& value : x_host) {
    value = dist(gen);
  }
  for (auto& value : y_host) {
    value = dist(gen);
  }
  // One row of the condition, broadcast over m rows by where(); the
  // row pattern is (j % 3) < 2.
  std::vector<double> expected(m * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      bool pick_x = (j % 3) < 2;
      expected[i * n + j] =
          pick_x ? x_host[i * n + j] : y_host[i * n + j];
    }
  }
  array cond_row = less(
      remainder(
          index_ramp(stream, Shape{1, n}), array(3, int32), stream),
      array(2, int32),
      stream);
  array x = device_values(stream, x_host, Shape{m, n}, float32);
  array y = device_values(stream, y_host, Shape{m, n}, float32);
  auto got = readback_f32(stream, where(cond_row, x, y, stream));
  expect_values(got, expected, 1e-6, "broadcast condition");
}

TEST_CASE("where with mixed operand broadcasts") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int m = 4;
  int n = 5;
  std::mt19937 gen(13);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  std::vector<double> x_host(m);
  std::vector<double> y_host(n);
  for (auto& value : x_host) {
    value = dist(gen);
  }
  for (auto& value : y_host) {
    value = dist(gen);
  }
  std::vector<double> expected(m * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      expected[i * n + j] = (i + j) % 2 == 1 ? x_host[i] : y_host[j];
    }
  }
  array mask = parity_mask(stream, m, n);
  array x = device_values(stream, x_host, Shape{m, 1}, float32);
  array y = device_values(stream, y_host, Shape{1, n}, float32);
  auto got = readback_f32(stream, where(mask, x, y, stream));
  expect_values(got, expected, 1e-6, "mixed broadcast");
}

// A broadcast bool operand materializes through logical_or.comp, which
// covers one element per invocation with no grid-stride loop. Past
// 65,535 groups * 256 = 16,776,960 elements a single dispatch left the
// tail unwritten (false), which is how upstream
// test_sdpa_full_head_dim_256's mx.where(causal_mask, p, min) over
// [8, 2048, 2048] scores lost its last four heads. The output stays a
// cheap [m, 1] x [1, n] broadcast so only the bool transport is large;
// mismatches are counted on the host instead of asserted per element.
TEST_CASE("broadcast bool operands past the 65,535-group dispatch limit") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  const int m = 3;
  const int n = 6000000; // m * n = 18,000,000 > 16,776,960
  // Condition row (j % 3) < 2, broadcast over m rows; x[i] = i + 1.
  array cond_row = less(
      remainder(index_ramp(stream, Shape{1, n}), array(3, int32), stream),
      array(2, int32),
      stream);
  array x = device_values(stream, {1.0, 2.0, 3.0}, Shape{m, 1}, float32);
  auto count_mismatches = [&](const std::vector<float>& got,
                              auto expected,
                              const std::string& label) {
    REQUIRE_EQ(got.size(), size_t(m) * n);
    size_t mismatches = 0;
    size_t first = got.size();
    for (size_t index = 0; index < got.size(); ++index) {
      if (got[index] != expected(index / n, index % n)) {
        if (mismatches++ == 0) {
          first = index;
        }
      }
    }
    CHECK_MESSAGE(
        mismatches == 0, label, ": ", mismatches, " mismatches, first at ", first);
  };
  count_mismatches(
      readback_f32(stream, where(cond_row, x, array(0.0f), stream)),
      [](size_t i, size_t j) { return (j % 3) < 2 ? float(i + 1) : 0.0f; },
      "where broadcast condition");
  // The logical family shares the dispatch: and(row, column) broadcast
  // over the same extent, with column[i] = i is even, so the last row
  // (the one past the limit) expects ones where the row pattern holds.
  array col_even = equal(
      bitwise_and(index_ramp(stream, Shape{m, 1}), array(1, int32), stream),
      array(0, int32),
      stream);
  count_mismatches(
      readback_f32(stream, logical_and(cond_row, col_even, stream)),
      [](size_t i, size_t j) { return ((j % 3) < 2 && (i % 2) == 0) ? 1.0f : 0.0f; },
      "logical_and broadcast operands");
}

// The chunk cursor advances by min(remaining, cap), never a bare
// += cap: from 256*cap a bare add wraps 4,294,901,760 + 16,776,960 to
// 16,711,424 and loops forever on counts a 4 GiB bool tensor can
// reach. This exercises the real constexpr boundary helper as pure
// host arithmetic -- no GPU, no allocation.
TEST_CASE("logical dispatch chunk cursor terminates across the uint32 boundary") {
  const uint32_t cap = omarchy::kLogicalChunkElements;
  const std::vector<uint32_t> counts = {
      1,
      cap - 1,
      cap,
      cap + 1,
      2 * cap + 7,
      255 * cap,
      256 * cap - 1,
      256 * cap,
      256 * cap + 1,
      4294967295u};
  for (uint32_t count : counts) {
    uint32_t first = 0;
    uint32_t steps = 0;
    while (first < count) {
      const uint32_t end = omarchy::next_logical_chunk_end(first, count);
      CHECK_MESSAGE(end > first, "cursor stalled for count ", count);
      CHECK_MESSAGE(end <= count, "cursor wrapped past count ", count);
      first = end;
      steps++;
    }
    CHECK_MESSAGE(
        steps == (uint64_t(count) + cap - 1) / cap,
        "wrong chunk count ", steps, " for count ", count);
  }
}

TEST_CASE("where with a strided condition") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int m = 4;
  int n = 6;
  std::mt19937 gen(17);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  // The condition is a [m,n]-shaped transposed view: base [n,m] built
  // through comparisons with cond_t[i][j] = ((j * m + i) % 3) == 0, then
  // transposed. data_size == size but the flat index is no longer the
  // memory index, so the host materializes it before the kernel's word
  // read.
  array cond_t = transpose(
      equal(
          remainder(index_ramp(stream, Shape{n, m}), array(3, int32), stream),
          array(0, int32),
          stream),
      std::vector<int>{1, 0},
      stream);
  std::vector<double> x_host(m * n);
  std::vector<double> y_host(m * n);
  for (auto& value : x_host) {
    value = dist(gen);
  }
  for (auto& value : y_host) {
    value = dist(gen);
  }
  std::vector<double> expected(m * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      bool pick_x = ((j * m + i) % 3) == 0;
      expected[i * n + j] =
          pick_x ? x_host[i * n + j] : y_host[i * n + j];
    }
  }
  array x = device_values(stream, x_host, Shape{m, n}, float32);
  array y = device_values(stream, y_host, Shape{m, n}, float32);
  auto got = readback_f32(stream, where(cond_t, x, y, stream));
  expect_values(got, expected, 1e-6, "strided condition");
}

TEST_CASE("where with an offset dense condition view") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int m = 4;
  int n = 9;  // Odd width makes the byte offset land mid-word.
  std::vector<double> x_host(m * n);
  std::vector<double> y_host(m * n);
  std::mt19937 gen(19);
  std::uniform_real_distribution<double> dist(-2.0, 2.0);
  for (auto& value : x_host) {
    value = dist(gen);
  }
  for (auto& value : y_host) {
    value = dist(gen);
  }
  // full[i][j] = ((i * n + j) % 4) == 1, built through comparisons; the
  // condition is rows 1..m, a dense view at byte offset n (odd, so
  // mid-word).
  array full = equal(
      remainder(index_ramp(stream, Shape{m + 1, n}), array(4, int32), stream),
      array(1, int32),
      stream);
  array cond =
      slice(full, Shape{1, 0}, Shape{m + 1, n}, Shape{1, 1}, stream);
  std::vector<double> expected(m * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      bool pick_x = (((i + 1) * n + j) % 4) == 1;
      expected[i * n + j] =
          pick_x ? x_host[i * n + j] : y_host[i * n + j];
    }
  }
  array x = device_values(stream, x_host, Shape{m, n}, float32);
  array y = device_values(stream, y_host, Shape{m, n}, float32);
  auto got = readback_f32(stream, where(cond, x, y, stream));
  expect_values(got, expected, 1e-6, "offset condition");
}
TEST_CASE("where sweeps value dtypes over a transposed operand") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int m = 4;
  int n = 6;
  struct DtypeCase {
    Dtype dtype;
    double tol;
    const char* label;
  };
  std::vector<DtypeCase> cases{
      {float32, 1e-6, "f32"},
      {float16, 1e-2, "f16"},
      {bfloat16, 1e-2, "bf16"},
      {int32, 0.0, "i32"},
      {uint32, 0.0, "u32"},
      {bool_, 0.0, "bool"},
  };
  for (auto& [dtype, tol, label] : cases) {
    // Desired operand values at logical [i][j]; the device array is a
    // transposed view of a [n,m] base so the shape stays [m,n] while the
    // strides stay strided.
    std::vector<double> base_host(n * m);
    std::vector<double> expected(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        double value = static_cast<double>(i * n + j);
        if (dtype == bool_) {
          // The bool base is built from the [n,m] flat ramp, so the
          // logical value rides the base's own flat index.
          value = (j * m + i) % 2;
        } else if (dtype == int32) {
          // Negative magnitudes ride the raw 32-bit transport.
          value = -(100000.0 + value);
        }
        base_host[j * m + i] = value;
        if ((i + j) % 2 == 1) {
          expected[i * n + j] = value;
        }
      }
    }
    array base = (dtype == bool_)
        ? equal(
              remainder(index_ramp(stream, Shape{n, m}),
                        array(2, int32),
                        stream),
              array(1, int32),
              stream)
        : device_values(stream, base_host, Shape{n, m}, dtype);
    array xt = transpose(base, std::vector<int>{1, 0}, stream);
    array mask = parity_mask(stream, m, n);
    auto got = readback_f32(
        stream, where(mask, xt, array(0, dtype), stream));
    expect_values(got, expected, tol, std::string("dtype sweep ") + label);
  }
}

TEST_CASE("where stays exact across word boundaries") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  // 33 and 63 elements: both cross 32-bit word lanes at odd offsets, the
  // shape class where packed-bool code diverged on hardware.
  for (auto [m, n] : {std::pair<int, int>{3, 11}, std::pair<int, int>{7, 9}}) {
    std::vector<double> base_host(n * m);
    std::vector<double> expected(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        base_host[j * m + i] = -(5000.0 + i * n + j);
        if ((i + j) % 2 == 1) {
          expected[i * n + j] = base_host[j * m + i];
        }
      }
    }
    array base = device_values(stream, base_host, Shape{n, m}, int32);
    array xt = transpose(base, std::vector<int>{1, 0}, stream);
    array mask = parity_mask(stream, m, n);
    auto got = readback_f32(
        stream, where(mask, xt, array(0, int32), stream));
    expect_values(
        got, expected, 0.0, "word boundary " + std::to_string(m * n));
  }
}

TEST_CASE("where selects between packed-bool operands") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int m = 5;
  int n = 7;  // 35 elements crosses word boundaries.
  // x[i][j] = (i*n+j) % 3 == 0, y[i][j] = (i*n+j) % 4 == 2, both built
  // through comparisons.
  array idx = index_ramp(stream, Shape{m, n});
  array x = equal(
      remainder(idx, array(3, int32), stream), array(0, int32), stream);
  array y = equal(
      remainder(idx, array(4, int32), stream), array(2, int32), stream);
  std::vector<double> expected(m * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      bool pick_x = (i + j) % 2 == 1;
      double xv = (i * n + j) % 3 == 0 ? 1.0 : 0.0;
      double yv = (i * n + j) % 4 == 2 ? 1.0 : 0.0;
      expected[i * n + j] = pick_x ? xv : yv;
    }
  }
  array mask = parity_mask(stream, m, n);
  auto got = readback_f32(stream, where(mask, x, y, stream));
  expect_values(got, expected, 0.0, "packed-bool where");
}

TEST_CASE("composed lu computes and reconstructs the input") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  for (auto [m, n] : {std::pair<int, int>{5, 5}, std::pair<int, int>{8, 5}}) {
    std::mt19937 gen(static_cast<unsigned int>(23 + m));
    std::vector<double> a_host(m * n);
    if (m == n) {
      a_host = host_diag_dominant(gen, n);
    } else {
      std::uniform_real_distribution<double> dist(-1.0, 1.0);
      for (auto& value : a_host) {
        value = dist(gen);
      }
    }
    array a = device_values(stream, a_host, Shape{m, n}, float32);
    std::vector<array> parts;
    try {
      parts = linalg::lu(a, stream);
    } catch (const std::exception& caught) {
      FAIL("composed lu refused: ", std::string(caught.what()));
    }
    CHECK_EQ(parts.size(), 3u);
    parts.at(0).eval();
    parts.at(1).eval();
    parts.at(2).eval();
    omarchy::get_command_encoder(stream).synchronize();
    int k = std::min(m, n);
    const uint32_t* pivot_data = parts.at(0).data<uint32_t>();
    std::vector<float> l_flat = readback_f32(stream, parts.at(1));
    std::vector<float> u_flat = readback_f32(stream, parts.at(2));
    // L is m x k unit-lower; U is k x n upper.
    std::vector<double> l_double(l_flat.size());
    std::vector<double> u_double(u_flat.size());
    for (size_t i = 0; i < l_flat.size(); ++i) {
      l_double[i] = l_flat[i];
    }
    for (size_t i = 0; i < u_flat.size(); ++i) {
      u_double[i] = u_flat[i];
    }
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < k; ++j) {
        double want = j < i ? l_double[i * k + j] : 0.0;
        if (j == i) {
          want = 1.0;
        }
        CHECK_MESSAGE(
            std::abs(l_double[i * k + j] - want) <= 1e-5,
            "lu unit-lower L at (",
            i,
            ",",
            j,
            ")");
      }
    }
    for (int i = 0; i < k; ++i) {
      for (int j = 0; j < i && j < n; ++j) {
        CHECK_MESSAGE(
            std::abs(u_double[i * n + j]) <= 1e-5,
            "lu upper U at (",
            i,
            ",",
            j,
            ")");
      }
    }
    std::vector<double> lu_product = host_matmul(l_double, u_double, m, k, n);
    std::vector<double> expected(m * n, 0.0);
    for (int i = 0; i < m; ++i) {
      int dest = static_cast<int>(pivot_data[i]);
      REQUIRE(dest >= 0);
      REQUIRE(dest < m);
      for (int j = 0; j < n; ++j) {
        expected[dest * n + j] = a_host[i * n + j];
      }
    }
    for (int i = 0; i < m * n; ++i) {
      CHECK_MESSAGE(
          std::abs(lu_product[i] - expected[i]) <= 1e-3,
          "lu reconstruction P*L*U vs A at index ",
          i,
          ": got ",
          lu_product[i],
          " want ",
          expected[i]);
    }
  }
}

TEST_CASE("composed solve computes A x == b through the uint32 ArgSort chain") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int n = 5;
  std::mt19937 gen(29);
  std::vector<double> a_host = host_diag_dominant(gen, n);
  std::vector<double> b_host{3.0, -1.0, 4.0, 1.5, 5.0};
  array a = device_values(stream, a_host, Shape{n, n}, float32);
  array b = device_values(stream, b_host, Shape{n}, float32);
  // solve = lu + argsort(uint32 pivots) + take_along_axis + triangular
  // solves. The uint32 ArgSort link is implemented, so the whole chain
  // must compute, and the result must satisfy A*x == b.
  std::vector<float> x_flat;
  try {
    array x = linalg::solve(a, b, stream);
    CHECK_EQ(x.shape(), Shape{n});
    x_flat = readback_f32(stream, std::move(x));
  } catch (const std::exception& caught) {
    FAIL("composed solve refused: ", std::string(caught.what()));
  }
  std::vector<double> x_double(x_flat.begin(), x_flat.end());
  std::vector<double> ax = host_matmul(a_host, x_double, n, n, 1);
  double tol = 1e-3;
  for (int i = 0; i < n; ++i) {
    CHECK_MESSAGE(
        std::abs(ax[i] - b_host[i]) <= tol,
        "solve residual mismatch at row ",
        i,
        ": A*x = ",
        ax[i],
        " b = ",
        b_host[i]);
  }

  // A second matrix whose LU factorization must swap rows (zero in the
  // leading pivot slot), so the uint32 argsort link sorts a pivot
  // vector that is not already sorted.
  int m2 = 5;
  std::vector<double> a2_host(m2 * m2, 0.0);
  for (int i = 0; i < m2; ++i) {
    a2_host[i * m2 + i] = 4.0;
    if (i > 0) {
      a2_host[i * m2 + i - 1] = 1.0;
    }
  }
  a2_host[0] = 0.0;
  a2_host[1] = 2.0;
  std::vector<double> b2_host{1.0, 2.0, 3.0, 4.0, 5.0};
  array a2 = device_values(stream, a2_host, Shape{m2, m2}, float32);
  array b2 = device_values(stream, b2_host, Shape{m2}, float32);
  std::vector<float> x2_flat;
  try {
    array x2 = linalg::solve(a2, b2, stream);
    CHECK_EQ(x2.shape(), Shape{m2});
    x2_flat = readback_f32(stream, std::move(x2));
  } catch (const std::exception& caught) {
    FAIL("composed solve with pivot swap refused: ",
         std::string(caught.what()));
  }
  std::vector<double> x2_double(x2_flat.begin(), x2_flat.end());
  std::vector<double> a2x2 = host_matmul(a2_host, x2_double, m2, m2, 1);
  for (int i = 0; i < m2; ++i) {
    CHECK_MESSAGE(
        std::abs(a2x2[i] - b2_host[i]) <= tol,
        "swapped-pivot solve residual mismatch at row ",
        i,
        ": A*x = ",
        a2x2[i],
        " b = ",
        b2_host[i]);
  }
}


TEST_CASE("composed pinv satisfies the Moore-Penrose conditions") {
  if (!compute_available()) {
    return;
  }
  auto stream = gpu_stream();
  int m = 4;
  int n = 6;
  std::mt19937 gen(31);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> a_host(m * n);
  for (auto& value : a_host) {
    value = dist(gen);
  }
  array a = device_values(stream, a_host, Shape{m, n}, float32);
  std::vector<float> x_flat;
  try {
    array pinv_x = linalg::pinv(a, stream);
    CHECK_EQ(pinv_x.shape(), Shape{n, m});
    x_flat = readback_f32(stream, pinv_x);
  } catch (const std::exception& caught) {
    FAIL("composed pinv refused: ", std::string(caught.what()));
  }
  std::vector<double> x_double(x_flat.begin(), x_flat.end());
  std::vector<double> a_double(a_host.begin(), a_host.end());
  // A (m x n), X (n x m).
  std::vector<double> ax = host_matmul(a_double, x_double, m, n, m);
  std::vector<double> xa = host_matmul(x_double, a_double, n, m, n);
  std::vector<double> axa = host_matmul(ax, a_double, m, m, n);
  std::vector<double> xax = host_matmul(xa, x_double, n, n, m);
  double tol = 2e-2;
  for (int i = 0; i < m * n; ++i) {
    CHECK_MESSAGE(
        std::abs(axa[i] - a_double[i]) <= tol,
        "pinv A*X*A vs A at index ",
        i,
        ": got ",
        axa[i]);
  }
  for (int i = 0; i < n * m; ++i) {
    CHECK_MESSAGE(
        std::abs(xax[i] - x_double[i]) <= tol,
        "pinv X*A*X vs X at index ",
        i,
        ": got ",
        xax[i]);
  }
  // X*A must be symmetric (Moore-Penrose condition three). This wide
  // shape carried two stacked k-boundary defects. First, the wide SVD
  // work copy walked the source with the transposed shape (m, n) over
  // transposed strides, reading past the input buffer, so work rows
  // >= k held recycled-page values that varied run to run - the
  // nondeterminism. Second, the SVD finalize completion walked columns
  // where the wide vt factor stores vectors as rows, reading
  // never-written cells and corrupting exactly the vt rows pinv slices.
  // The copy now walks the work shape and the completion indexes rows
  // for wide inputs; this check pins both.
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      CHECK_MESSAGE(
          std::abs(xa[i * n + j] - xa[j * n + i]) <= 1e-3,
          "pinv X*A symmetry at (",
          i,
          ",",
          j,
          "): got ",
          xa[i * n + j] - xa[j * n + i]);
    }
  }
}



} // namespace
