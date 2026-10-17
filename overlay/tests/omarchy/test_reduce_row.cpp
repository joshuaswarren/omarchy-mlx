// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Workgroup-per-row suffix reduction (shaders/reduce_row.comp, routed by
// Reduce::eval_gpu when MLX_OMARCHY_ROW_REDUCE=1). The gate is numeric:
//   - Sum error against a float64 reference of the stored input values stays
//     inside a float32 accumulation bound, for float32, float16 and bfloat16
//     storage, at the decode widths 8192, 2816 and 5120 and at batched rows;
//   - Max and Min keep the serial route (bit-equal, signed-zero ties included);
//   - repeating one input gives identical bytes, and a row gives the same
//     bytes alone, inside a larger batch and behind a slice offset (the
//     order is fixed by the shader, not by the dispatch);
//   - NaN poisons the sum wherever it sits in the row; infinities follow
//     IEEE;
//   - rows shorter than 256 keep the serial route.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device (set "
                 "MLX_OMARCHY_ALLOW_NON_APPLE=1 on a development machine).\n";
    return false;
  }
  return true;
}

bool float16_available() {
  const auto& capabilities = omarchy::device(0).capabilities();
  return capabilities.shader_float16 &&
      capabilities.storage_buffer_16bit_access;
}

void set_route(bool row) {
  setenv("MLX_OMARCHY_ROW_REDUCE", row ? "1" : "0", 1);
}

// Reduce `x` over `axes` with the chosen route and return the float32 values
// of the output (the output keeps the input dtype on the device).
std::vector<float> run(
    const array& x,
    const std::vector<int>& axes,
    int op,
    bool row,
    const Stream& stream) {
  set_route(row);
  array r = op == 0 ? sum(x, axes, false, stream)
      : op == 1     ? max(x, axes, false, stream)
                    : min(x, axes, false, stream);
  array f = astype(r, float32, stream);
  f.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* p = f.data<float>();
  return std::vector<float>(p, p + f.size());
}

std::vector<float> make_values(size_t count, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  std::uniform_int_distribution<int> scale(-3, 3);
  std::vector<float> v(count);
  for (auto& x : v) {
    x = std::ldexp(unit(rng), scale(rng));
  }
  return v;
}

bool same_bytes(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
      std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct Case {
  std::vector<int> shape;
  int naxes;
};

double storage_step(Dtype dtype) {
  return dtype == bfloat16 ? 0x1p-8 : dtype == float16 ? 0x1p-11 : 0.0;
}

} // namespace

TEST_CASE("row reduce: sum error against a float64 reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const std::vector<Case> cases = {
      {{1, 1, 8192}, 1},
      {{1, 1, 2816}, 1},
      {{1, 1, 5120}, 1},
      {{1, 1, 256}, 1},
      {{1, 1, 4097}, 1},
      {{3, 5, 4096}, 1},
      {{2, 7, 2816}, 1},
      {{4, 32, 128}, 2},
      {{1, 1, 70001}, 1},
  };
  std::vector<Dtype> dtypes = {float32, bfloat16};
  if (float16_available()) {
    dtypes.push_back(float16);
  }
  for (Dtype dtype : dtypes) {
    for (const auto& c : cases) {
      size_t count = 1;
      for (int d : c.shape) {
        count *= d;
      }
      size_t row = 1;
      for (int i = 0; i < c.naxes; ++i) {
        row *= c.shape[c.shape.size() - 1 - i];
      }
      std::vector<int> axes;
      for (int i = 0; i < c.naxes; ++i) {
        axes.push_back(static_cast<int>(c.shape.size()) - c.naxes + i);
      }
      Shape shape(c.shape.begin(), c.shape.end());
      array x = astype(array(make_values(count, 7u + count).data(), shape, float32), dtype, stream);
      array xf = astype(x, float32, stream);
      xf.eval();
      omarchy::get_command_encoder(stream).synchronize();
      const float* in = xf.data<float>();
      std::vector<float> got = run(x, axes, 0, true, stream);
      std::vector<float> old = run(x, axes, 0, false, stream);
      REQUIRE_EQ(got.size(), count / row);
      double worst_row = 0.0, worst_old = 0.0;
      for (size_t r = 0; r < got.size(); ++r) {
        long double ref = 0.0L, mag = 0.0L;
        for (size_t i = 0; i < row; ++i) {
          ref += static_cast<long double>(in[r * row + i]);
          mag += std::fabs(static_cast<long double>(in[r * row + i]));
        }
        double bound = 32.0 * 0x1p-24 * static_cast<double>(mag) +
            storage_step(dtype) * std::fabs(static_cast<double>(ref));
        double e_row = std::fabs(static_cast<double>(got[r]) - static_cast<double>(ref));
        INFO("dtype " << static_cast<int>(dtype.val()) << " row " << row
                      << " r " << r << " ref " << static_cast<double>(ref)
                      << " got " << got[r] << " old " << old[r]);
        CHECK(e_row <= bound);
        double unit = 0x1p-24 * static_cast<double>(mag);
        if (unit > 0.0) {
          worst_row = std::max(worst_row, e_row / unit);
          worst_old = std::max(
              worst_old,
              std::fabs(static_cast<double>(old[r]) - static_cast<double>(ref)) / unit);
        }
      }
      // Error in units of 2^-24 times the sum of magnitudes, worst row. For
      // float16 and bfloat16 outputs the storage rounding dominates both columns.
      std::cout << "ROWREDUCE-ERR dtype=" << static_cast<int>(dtype.val())
                << " row=" << row << " rows=" << got.size()
                << " row_route=" << worst_row << " serial_route=" << worst_old << "\n";
    }
  }
}

TEST_CASE("row reduce: max and min keep the serial route, ties between signed zeros included") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  for (Dtype dtype : {float32, bfloat16}) {
    for (int width : {256, 300, 2816, 8192}) {
      std::vector<float> data = make_values(static_cast<size_t>(6) * width, 11u + width);
      // Every other row is all zeros, negative at every fifth position: all ties. At width 300 the last element (+0) and element 255
      // (-0), the last the tree combines, differ, so a tree route and the serial walk return different zeros.
      for (size_t i = 0; i < static_cast<size_t>(width) * 6; ++i) {
        if ((i / width) % 2 == 1) {
          data[i] = ((i % width) % 5 == 0) ? -0.0f : 0.0f;
        }
      }
      array x = astype(array(data.data(), Shape{2, 3, width}, float32), dtype, stream);
      for (int op : {1, 2}) {
        CHECK(same_bytes(run(x, {2}, op, true, stream), run(x, {2}, op, false, stream)));
      }
    }
  }
}

TEST_CASE("row reduce: bytes repeat and do not depend on the dispatch") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int width = 8192;
  std::vector<float> data = make_values(static_cast<size_t>(5) * 6 * width, 99u);
  array big = array(data.data(), Shape{5, 6, width}, float32);
  std::vector<float> first = run(big, {2}, 0, true, stream);
  for (int i = 0; i < 20; ++i) {
    CHECK(same_bytes(first, run(big, {2}, 0, true, stream)));
  }
  // The row at index 17 alone, as a (1, 1, width) input.
  array one = array(data.data() + static_cast<size_t>(17) * width, Shape{1, 1, width}, float32);
  std::vector<float> alone = run(one, {2}, 0, true, stream);
  CHECK_EQ(std::memcmp(&alone[0], &first[17], sizeof(float)), 0);
  // The same row behind a slice offset of the batch.
  array sliced = slice(big, {2, 0, 0}, {5, 6, width}, stream);
  std::vector<float> from_slice = run(sliced, {2}, 0, true, stream);
  for (size_t r = 0; r < from_slice.size(); ++r) {
    CHECK_EQ(std::memcmp(&from_slice[r], &first[12 + r], sizeof(float)), 0);
  }
  // More rows than one dispatch width of workgroups does not change a row.
  const int rows = 70000;
  std::vector<float> many = make_values(static_cast<size_t>(rows) * 256, 5u);
  array wide = array(many.data(), Shape{rows, 256}, float32);
  std::vector<float> w = run(wide, {1}, 0, true, stream);
  array head = array(many.data() + static_cast<size_t>(rows - 1) * 256, Shape{1, 256}, float32);
  std::vector<float> h = run(head, {1}, 0, true, stream);
  CHECK_EQ(std::memcmp(&w[rows - 1], &h[0], sizeof(float)), 0);
}

TEST_CASE("row reduce: NaN poisons the sum, infinities follow IEEE") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int width = 2816;
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  for (int position : {0, 255, 256, 1000, width - 1}) {
    std::vector<float> data = make_values(width, 3u);
    data[position] = nan;
    array x = array(data.data(), Shape{1, 1, width}, float32);
    std::vector<float> got = run(x, {2}, 0, true, stream);
    CHECK(std::isnan(got[0]));
  }
  std::vector<float> data = make_values(width, 4u);
  data[17] = inf;
  array has_inf = array(data.data(), Shape{1, 1, width}, float32);
  CHECK_EQ(run(has_inf, {2}, 0, true, stream)[0], inf);
  data[900] = -inf;
  array both = array(data.data(), Shape{1, 1, width}, float32);
  CHECK(std::isnan(run(both, {2}, 0, true, stream)[0]));
}

TEST_CASE("row reduce: rows shorter than 256 keep the serial route") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  std::vector<float> data = make_values(static_cast<size_t>(4) * 255, 21u);
  array x = array(data.data(), Shape{4, 255}, float32);
  CHECK(same_bytes(run(x, {1}, 0, true, stream), run(x, {1}, 0, false, stream)));
}
