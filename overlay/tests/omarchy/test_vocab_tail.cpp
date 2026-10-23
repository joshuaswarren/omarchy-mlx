// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Chunked ArgMax/ArgMin and LogSumExp for long rows
// (shaders/argreduce_chunk.comp + argreduce_combine.comp, routed by
// MLX_OMARCHY_ARGREDUCE_CHUNKED=1; shaders/logsumexp_chunk.comp +
// logsumexp_combine.comp, routed by MLX_OMARCHY_LSE_CHUNKED=1). The gates:
//   - ArgMax/ArgMin: the index is EXACTLY the index of the single-workgroup
//     kernel and of a host reference, for ties across chunk edges, first NaN,
//     infinities, all-equal rows and signed zeros, in float32, float16 and
//     bfloat16, from 8192 to 1048576 columns and for one or several rows;
//   - LogSumExp: the error against a float64 reference of the stored inputs
//     is not worse than the single-workgroup kernel's error on the same row
//     (plus a few units of the output precision), and NaN, infinite and
//     all -inf rows return what the single-workgroup kernel returns;
//   - the route engages exactly when its flag is "1" and the row is in
//     [8192, 1048576] columns (two compute dispatches instead of one).

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
#include "mlx/backend/omarchy/trace.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr float kNan = std::numeric_limits<float>::quiet_NaN();

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

std::vector<Dtype> dtypes() {
  std::vector<Dtype> out = {float32, bfloat16};
  if (float16_available()) {
    out.push_back(float16);
  }
  return out;
}

void set_flags(bool chunked) {
  setenv("MLX_OMARCHY_ARGREDUCE_CHUNKED", chunked ? "1" : "0", 1);
  setenv("MLX_OMARCHY_LSE_CHUNKED", chunked ? "1" : "0", 1);
}

uint64_t dispatches() {
  return omarchy::trace::counters().vk_compute_dispatches.load();
}

// Device array of `values` stored in `dtype`, evaluated, plus the stored
// values (after the dtype rounding) as float32.
struct Input {
  array x;
  std::vector<float> stored;
};

Input make_input(
    const std::vector<float>& values,
    int rows,
    int cols,
    Dtype dtype,
    const Stream& stream) {
  array x(values.data(), Shape{rows, cols}, float32);
  array y = astype(x, dtype, stream);
  y.eval();
  array f = astype(y, float32, stream);
  f.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* p = f.data<float>();
  return {y, std::vector<float>(p, p + f.size())};
}

// Index reference of the mlx rule on the stored values.
uint32_t ref_arg(const float* row, int cols, bool is_max) {
  float best = is_max ? -kInf : kInf;
  uint32_t index = 0;
  for (int i = 0; i < cols; ++i) {
    if (std::isnan(row[i])) {
      return static_cast<uint32_t>(i);
    }
    if (is_max ? row[i] > best : row[i] < best) {
      best = row[i];
      index = static_cast<uint32_t>(i);
    }
  }
  return index;
}

struct ArgRun {
  std::vector<uint32_t> index;
  uint64_t launched;
};

ArgRun run_arg(const array& x, bool is_max, bool chunked, const Stream& stream) {
  set_flags(chunked);
  uint64_t before = dispatches();
  array r = is_max ? argmax(x, -1, false, stream) : argmin(x, -1, false, stream);
  r.eval();
  omarchy::get_command_encoder(stream).synchronize();
  uint64_t launched = dispatches() - before;
  const uint32_t* p = r.data<uint32_t>();
  return {std::vector<uint32_t>(p, p + r.size()), launched};
}

// Row pattern `kind`, seed `seed`, width `cols`.
std::vector<float> pattern(int kind, int cols, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
  std::vector<float> v(cols);
  for (auto& x : v) {
    x = unit(rng);
  }
  const int edges[] = {4095, 4096, 4097, 8191, 8192, 12287, 12288, 100000};
  switch (kind) {
    case 0: // random
      break;
    case 1: // all equal
      std::fill(v.begin(), v.end(), 0.5f);
      break;
    case 2: // equal extremes either side of chunk edges and at the last column
      for (int e : edges) {
        if (e < cols) {
          v[e] = (e % 2 == 0) ? 5.0f : -5.0f;
        }
      }
      v[cols - 1] = 5.0f;
      v[cols - 2] = -5.0f;
      break;
    case 3: // NaNs: the first one wins, wherever it sits
      for (int e : {cols - 1, cols / 2 + 7, 8192 < cols ? 8192 : cols - 3}) {
        v[e] = kNan;
      }
      break;
    case 4: // one NaN in the last chunk, ties in the earlier ones
      std::fill(v.begin(), v.end(), 1.0f);
      v[cols - 1] = kNan;
      break;
    case 5: // -inf everywhere but one finite value
      std::fill(v.begin(), v.end(), -kInf);
      v[cols / 2 + 1] = -3.0f;
      break;
    case 6: // +inf everywhere but one finite value
      std::fill(v.begin(), v.end(), kInf);
      v[cols / 3] = 3.0f;
      break;
    case 7: // all -inf
      std::fill(v.begin(), v.end(), -kInf);
      break;
    case 8: // all +inf
      std::fill(v.begin(), v.end(), kInf);
      break;
    case 9: // signed zeros
      for (int i = 0; i < cols; ++i) {
        v[i] = (i % 3 == 0) ? -0.0f : 0.0f;
      }
      break;
    case 10: // +inf and -inf at chunk edges
      v[4096] = kInf;
      v[4095] = kInf;
      v[cols - 1] = -kInf;
      v[8192 < cols ? 8192 : 0] = -kInf;
      break;
    default:
      break;
  }
  return v;
}

constexpr int kPatterns = 11;

} // namespace

TEST_CASE("chunked arg reduce: index equals the single-workgroup kernel") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const std::vector<int> widths = {
      8192, 8193, 12288, 20000, 248320, 400000, 1048576};
  for (Dtype dtype : dtypes()) {
    for (int cols : widths) {
      int rows = cols > 300000 ? 1 : 3;
      for (int kind = 0; kind < kPatterns; ++kind) {
        std::vector<float> values;
        values.reserve(size_t(rows) * cols);
        for (int r = 0; r < rows; ++r) {
          auto row = pattern(kind, cols, 100 + 7 * r + kind);
          values.insert(values.end(), row.begin(), row.end());
        }
        Input in = make_input(values, rows, cols, dtype, stream);
        for (bool is_max : {true, false}) {
          ArgRun serial = run_arg(in.x, is_max, false, stream);
          ArgRun chunked = run_arg(in.x, is_max, true, stream);
          CHECK(serial.launched == 1);
          CHECK(chunked.launched == 2);
          bool equal = chunked.index == serial.index;
          bool matches_reference = true;
          for (int r = 0; r < rows; ++r) {
            matches_reference &= chunked.index[r] ==
                ref_arg(in.stored.data() + size_t(r) * cols, cols, is_max);
          }
          INFO("dtype=" << dtype << " cols=" << cols << " pattern=" << kind
                        << " max=" << is_max);
          CHECK(equal);
          CHECK(matches_reference);
        }
      }
    }
  }
}

TEST_CASE("chunked arg reduce: route gate") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // 8191 and 1048577 columns keep the single-workgroup kernel with the flag
  // on; integer inputs always do; flag off never chunks.
  for (int cols : {8191, 1048577}) {
    Input in =
        make_input(pattern(0, cols, 5), 1, cols, float32, stream);
    ArgRun run = run_arg(in.x, true, true, stream);
    CHECK(run.launched == 1);
    CHECK(run.index[0] == ref_arg(in.stored.data(), cols, true));
  }
  {
    Input in = make_input(pattern(0, 20000, 5), 1, 20000, float32, stream);
    CHECK(run_arg(in.x, true, false, stream).launched == 1);
    array ints = astype(in.x * 1000.0f, int32, stream);
    ints.eval();
    set_flags(true);
    uint64_t before = dispatches();
    array r = argmax(ints, -1, false, stream);
    r.eval();
    omarchy::get_command_encoder(stream).synchronize();
    CHECK(dispatches() - before == 1);
  }
  // Many rows: 4000 rows of two chunks each (the input stays under the
  // device's storage-buffer range). Rows times chunks past the 65535 group
  // limit would keep the single-workgroup kernel, but the buffer range
  // makes that unreachable at 8192 columns, so the guard is not tested here.
  {
    int rows = 4000;
    int cols = 8192;
    std::vector<float> values(size_t(rows) * cols, 1.0f);
    for (int r = 0; r < rows; ++r) {
      values[size_t(r) * cols + (r * 37) % cols] = 2.0f;
    }
    Input in = make_input(values, rows, cols, bfloat16, stream);
    ArgRun run = run_arg(in.x, true, true, stream);
    CHECK(run.launched == 2);
    for (int r = 0; r < rows; ++r) {
      CHECK(run.index[r] == uint32_t((r * 37) % cols));
    }
  }
}

namespace {

// float64 log-sum-exp of the stored row, same special-value rules.
double ref_lse(const float* row, int cols) {
  double m = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < cols; ++i) {
    if (std::isnan(row[i])) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    m = std::max(m, static_cast<double>(row[i]));
  }
  if (std::isinf(m)) {
    return m;
  }
  double s = 0.0;
  for (int i = 0; i < cols; ++i) {
    s += std::exp(static_cast<double>(row[i]) - m);
  }
  return m + std::log(s);
}

std::vector<float> run_lse(const array& x, bool chunked, uint64_t* launched,
                           const Stream& stream) {
  set_flags(chunked);
  uint64_t before = dispatches();
  array r = logsumexp(x, std::vector<int>{-1}, false, stream);
  r.eval();
  omarchy::get_command_encoder(stream).synchronize();
  if (launched != nullptr) {
    *launched = dispatches() - before;
  }
  array f = astype(r, float32, stream);
  f.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* p = f.data<float>();
  return std::vector<float>(p, p + f.size());
}

std::vector<float> lse_pattern(int kind, int cols, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 3.0f);
  std::vector<float> v(cols);
  for (auto& x : v) {
    x = normal(rng);
  }
  switch (kind) {
    case 0: // logits-like
      break;
    case 1: // large offset
      for (auto& x : v) {
        x += 900.0f;
      }
      break;
    case 2: // one dominant spike near a chunk edge
      v[4096] = 40.0f;
      break;
    case 3: // all equal
      std::fill(v.begin(), v.end(), -2.0f);
      break;
    case 4: // masked: whole chunks of -inf and scattered -inf
      for (int i = 0; i < cols; ++i) {
        if ((i / 4096) % 2 == 0 || i % 5 == 0) {
          v[i] = -kInf;
        }
      }
      break;
    case 5: // all -inf
      std::fill(v.begin(), v.end(), -kInf);
      break;
    case 6: // +inf present
      v[cols / 2] = kInf;
      break;
    case 7: // NaN present
      v[cols - 1] = kNan;
      break;
    case 8: // very negative values (sum would underflow without the max)
      for (auto& x : v) {
        x -= 600.0f;
      }
      break;
    default:
      break;
  }
  return v;
}

constexpr int kLsePatterns = 9;

bool same_special(float a, double b) {
  if (std::isnan(b)) {
    return std::isnan(a);
  }
  if (std::isinf(b)) {
    return a == static_cast<float>(b);
  }
  return false;
}

} // namespace

TEST_CASE("chunked logsumexp: error against a float64 reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const std::vector<int> widths = {8192, 8193, 20000, 248320, 400000, 1048576};
  for (Dtype dtype : dtypes()) {
    // Output precision: units of one rounding step of the output dtype.
    double unit = dtype == float32 ? 0x1p-23 : dtype == float16 ? 0x1p-10
                                                              : 0x1p-7;
    for (int cols : widths) {
      int rows = cols > 300000 ? 1 : 3;
      for (int kind = 0; kind < kLsePatterns; ++kind) {
        std::vector<float> values;
        for (int r = 0; r < rows; ++r) {
          auto row = lse_pattern(kind, cols, 300 + 11 * r + kind);
          values.insert(values.end(), row.begin(), row.end());
        }
        Input in = make_input(values, rows, cols, dtype, stream);
        uint64_t serial_launched = 0;
        uint64_t chunked_launched = 0;
        auto serial = run_lse(in.x, false, &serial_launched, stream);
        auto chunked = run_lse(in.x, true, &chunked_launched, stream);
        INFO("dtype=" << dtype << " cols=" << cols << " pattern=" << kind);
        CHECK(serial_launched == 1);
        CHECK(chunked_launched == 2);
        for (int r = 0; r < rows; ++r) {
          double ref = ref_lse(in.stored.data() + size_t(r) * cols, cols);
          if (std::isnan(ref) || std::isinf(ref)) {
            CHECK(same_special(serial[r], ref));
            CHECK(same_special(chunked[r], ref));
            continue;
          }
          double old_err = std::fabs(serial[r] - ref);
          double new_err = std::fabs(chunked[r] - ref);
          double slack = 2.0 * unit * std::max(std::fabs(ref), 1.0) * 0.5;
          CHECK(new_err <= old_err + slack);
          // Both are close to the reference in absolute terms.
          CHECK(new_err <= 4.0 * unit * std::max(std::fabs(ref), 1.0));
        }
      }
    }
  }
}

TEST_CASE("chunked logsumexp: route gate and fallback bytes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  for (int cols : {4096, 8191, 1048577}) {
    Input in = make_input(lse_pattern(0, cols, 9), 1, cols, float32, stream);
    uint64_t launched = 0;
    auto serial = run_lse(in.x, false, nullptr, stream);
    auto flagged = run_lse(in.x, true, &launched, stream);
    CHECK(launched == 1);
    CHECK(std::memcmp(serial.data(), flagged.data(), sizeof(float)) == 0);
  }
  Input in = make_input(lse_pattern(0, 20000, 9), 1, 20000, float32, stream);
  uint64_t launched = 0;
  run_lse(in.x, false, &launched, stream);
  CHECK(launched == 1);
}

TEST_CASE("vocab tail default: the part rule") {
  CHECK(omarchy::vocab_tail_chunked_default("Apple M2 Max (G14C B1)"));
  CHECK_FALSE(omarchy::vocab_tail_chunked_default("Apple M1 Max (G13C C0)"));
  CHECK_FALSE(omarchy::vocab_tail_chunked_default("Apple M1 (G13G B1)"));
  CHECK_FALSE(omarchy::vocab_tail_chunked_default(""));
}

// Launches of one ArgMax or one LogSumExp over the row with the flags as the environment holds them.
static uint64_t tail_launches(const array& x, bool arg, const Stream& stream) {
  uint64_t before = dispatches();
  array r = arg ? argmax(x, -1, false, stream) : logsumexp(x, std::vector<int>{-1}, false, stream);
  r.eval();
  omarchy::get_command_encoder(stream).synchronize();
  return dispatches() - before;
}

TEST_CASE("vocab tail default: unset flags follow the part, a set flag wins") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  Input in = make_input(pattern(0, 20000, 5), 1, 20000, float32, stream);
  bool on = omarchy::vocab_tail_chunked_default(omarchy::device(0).capabilities().device_name);
  for (bool arg : {true, false}) {
    const char* name = arg ? "MLX_OMARCHY_ARGREDUCE_CHUNKED" : "MLX_OMARCHY_LSE_CHUNKED";
    unsetenv("MLX_OMARCHY_ARGREDUCE_CHUNKED");
    unsetenv("MLX_OMARCHY_LSE_CHUNKED");
    CHECK(tail_launches(in.x, arg, stream) == (on ? 2u : 1u));
    setenv(name, "0", 1);
    CHECK(tail_launches(in.x, arg, stream) == 1);
    setenv(name, "1", 1);
    CHECK(tail_launches(in.x, arg, stream) == 2);
  }
  unsetenv("MLX_OMARCHY_ARGREDUCE_CHUNKED");
  unsetenv("MLX_OMARCHY_LSE_CHUNKED");
}
