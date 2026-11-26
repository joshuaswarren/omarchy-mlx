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
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
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

void set_flags(std::optional<bool> chunked) {
  if (!chunked.has_value()) {
    unsetenv("MLX_OMARCHY_ARGREDUCE_CHUNKED");
    unsetenv("MLX_OMARCHY_LSE_CHUNKED");
    return;
  }
  setenv("MLX_OMARCHY_ARGREDUCE_CHUNKED", *chunked ? "1" : "0", 1);
  setenv("MLX_OMARCHY_LSE_CHUNKED", *chunked ? "1" : "0", 1);
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

ArgRun run_arg(const array& x, bool is_max, std::optional<bool> chunked, const Stream& stream) {
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

std::vector<float> run_lse(const array& x, std::optional<bool> chunked,
                           uint64_t* launched, const Stream& stream) {
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
  CHECK(omarchy::vocab_tail_chunked_default("Apple M5 Max (G17C B1)"));
  CHECK(omarchy::vocab_tail_chunked_default("Apple M6 (G18G B1)"));
  CHECK(omarchy::vocab_tail_chunked_default("Apple M3 (G15G B1)"));
  CHECK(omarchy::vocab_tail_chunked_default(""));
  CHECK_FALSE(omarchy::vocab_tail_chunked_default("Apple M1 Max (G13C C0)"));
  CHECK_FALSE(omarchy::vocab_tail_chunked_default("Apple M1 (G13G B1)"));
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
    setenv(name, "true", 1);
    CHECK(tail_launches(in.x, arg, stream) == 1);
  }
  unsetenv("MLX_OMARCHY_ARGREDUCE_CHUNKED");
  unsetenv("MLX_OMARCHY_LSE_CHUNKED");
}

TEST_CASE("vocab tail default: bf16 logits preserve serial parity") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr int cols = 248320;
  bool on = omarchy::vocab_tail_chunked_default(
      omarchy::device(0).capabilities().device_name);
  Input in = make_input(pattern(2, cols, 17), 1, cols, bfloat16, stream);
  for (bool is_max : {true, false}) {
    auto serial = run_arg(in.x, is_max, false, stream);
    auto selected = run_arg(in.x, is_max, std::nullopt, stream);
    CHECK(selected.launched == (on ? 2u : 1u));
    CHECK(selected.index == serial.index);
    CHECK(selected.index[0] == ref_arg(in.stored.data(), cols, is_max));
  }

  Input logits = make_input(lse_pattern(0, cols, 19), 1, cols, bfloat16, stream);
  auto serial = run_lse(logits.x, false, nullptr, stream);
  uint64_t launched = 0;
  auto selected = run_lse(logits.x, std::nullopt, &launched, stream);
  CHECK(launched == (on ? 2u : 1u));
  double ref = ref_lse(logits.stored.data(), cols);
  double old_err = std::fabs(serial[0] - ref);
  double new_err = std::fabs(selected[0] - ref);
  constexpr double unit = 0x1p-7;
  double slack = 2.0 * unit * std::max(std::fabs(ref), 1.0) * 0.5;
  CHECK(new_err <= old_err + slack);
  CHECK(new_err <= 4.0 * unit * std::max(std::fabs(ref), 1.0));
  set_flags(std::nullopt);
}

// ---- fused greedy head over the Q4 word column (DEC-46) ----
// fast::greedy_quantized_argmax answers argmax(logits -
// logsumexp(logits)) for the tied 4-bit/group-64 lm_head. Where
// QuantizedMatmul picks the non-subgroup QmmVecQ4Word column the
// primitive answers in three dispatches (shaders/qmm_vec.comp
// QMM_VEC_GREEDY_WORD); these cases pin
// the token to the composed expression's, including argreduce's
// first-index and first-NaN rules.

namespace {

constexpr int kGreedyK = 2048;
constexpr int kGreedyWords = kGreedyK / 8;     // 256
constexpr int kGreedyGroups = kGreedyK / 64;   // 32
constexpr int kGreedyPlane = kGreedyK * 3 / 32; // 192

struct GreedyInputs {
  array x;
  array w;
  array scales;
  array biases;
  array high;
};

// A synthetic lm_head: `n` rows of packed 4-bit codes over `k`,
// per-64-group bf16 scales/biases, a bf16 activation row, and the
// (unused by the word route) high-bits plane the API requires.
// `duplicates` copies one row's words and parameters over later rows.
GreedyInputs make_greedy_inputs(
    int n,
    uint64_t seed,
    const std::vector<float>& scale_pattern,
    const std::vector<float>& bias_pattern,
    const std::vector<int>& duplicate_sources,
    const std::vector<int>& duplicate_targets,
    bool ones_x,
    const Stream& stream,
    int k = kGreedyK) {
  const int words_per = k / 8;
  const int groups_per = k / 64;
  const int plane_per = k * 3 / 32;
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::vector<uint32_t> words(n * words_per);
  for (auto& v : words) {
    v = rng();
  }
  std::vector<float> scales_v(n * groups_per);
  std::vector<float> biases_v(n * groups_per);
  for (int r = 0; r < n; ++r) {
    for (int g = 0; g < groups_per; ++g) {
      scales_v[r * groups_per + g] =
          scale_pattern[static_cast<size_t>(r + g) % scale_pattern.size()];
      biases_v[r * groups_per + g] =
          bias_pattern[static_cast<size_t>(r + g) % bias_pattern.size()];
    }
  }
  for (size_t d = 0; d < duplicate_sources.size(); ++d) {
    int src = duplicate_sources[d];
    int dst = duplicate_targets[d];
    std::copy(
        words.begin() + src * words_per,
        words.begin() + (src + 1) * words_per,
        words.begin() + dst * words_per);
    std::copy(
        scales_v.begin() + src * groups_per,
        scales_v.begin() + (src + 1) * groups_per,
        scales_v.begin() + dst * groups_per);
    std::copy(
        biases_v.begin() + src * groups_per,
        biases_v.begin() + (src + 1) * groups_per,
        biases_v.begin() + dst * groups_per);
  }
  std::vector<float> xv(k);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : xv) {
    v = ones_x ? 1.0f : dist(rng);
  }
  array xf(xv.data(), Shape{1, k}, float32);
  array wf(words.data(), Shape{n, words_per}, uint32);
  array sf(scales_v.data(), Shape{n, groups_per}, float32);
  array bf(biases_v.data(), Shape{n, groups_per}, float32);
  std::vector<uint32_t> zeros(n * plane_per, 0u);
  array high(zeros.data(), Shape{n, plane_per}, uint32);
  return {astype(xf, bfloat16, stream),
          wf,
          astype(sf, bfloat16, stream),
          astype(bf, bfloat16, stream),
          high};
}

// The primitive's own fallback expression, evaluated through the normal
// ops so the reference exercises the composed vocab tail.
uint32_t composed_argmax(const GreedyInputs& in, const Stream& stream) {
  auto logits = quantized_matmul(
      in.x, in.w, in.scales, in.biases, true, 64, 4, "affine", stream);
  auto logprobs =
      subtract(logits, logsumexp(logits, -1, true, stream), stream);
  auto token = astype(argmax(logprobs, -1, false, stream), uint32, stream);
  token.eval();
  omarchy::get_command_encoder(stream).synchronize();
  return token.item<uint32_t>();
}

struct GreedyRun {
  uint32_t token;
  std::vector<uint32_t> stats;
  uint64_t launched;
};

GreedyRun fused_argmax(const GreedyInputs& in, const Stream& stream) {
  // Materialize the lazy astype inputs first so the dispatch window
  // counts only the head itself (array handles share the node).
  array x_m = in.x, w_m = in.w, s_m = in.scales, b_m = in.biases,
        h_m = in.high;
  x_m.eval();
  w_m.eval();
  s_m.eval();
  b_m.eval();
  h_m.eval();
  uint64_t before = dispatches();
  auto out = fast::greedy_quantized_argmax(
      in.x, in.w, in.scales, in.biases, in.high, 64, 4, stream);
  out.at(0).eval();
  out.at(1).eval();
  omarchy::get_command_encoder(stream).synchronize();
  GreedyRun r;
  r.token = out.at(0).item<uint32_t>();
  const uint32_t* p = out.at(1).data<uint32_t>();
  r.stats.assign(p, p + 3);
  r.launched = dispatches() - before;
  return r;
}

// The word route's device terms (K-independent); the same terms
// primitives.cpp's eval_gpu checks.
bool word_caps() {
  const auto& caps = omarchy::device(0).capabilities();
  const char* env = std::getenv("MLX_OMARCHY_QMM_VEC_Q4_WORD");
  bool word_enabled = env == nullptr || std::strcmp(env, "1") == 0;
  return word_enabled && caps.storage_buffer_16bit_access &&
      caps.shader_int16;
}

bool subgroup_ready() {
  const auto& caps = omarchy::device(0).capabilities();
  return caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) !=
          0u;
}

// True when eval_gpu takes the non-subgroup QMM_VEC_GREEDY_WORD route
// for this K and row count: the staged QMM_VEC_GREEDY route claims
// K = 2048 on subgroup devices; every other K takes the word route.
// Both routes bind w and high_bits whole, so a device at the Vulkan
// minimum maxStorageBufferRange composes on vocab-scale inputs.
bool word_route_taken(int k, int n) {
  const auto& caps = omarchy::device(0).capabilities();
  const uint64_t w_bytes = static_cast<uint64_t>(n) * k / 8u * 4u;
  const uint64_t plane_bytes = static_cast<uint64_t>(n) * k * 3u / 32u * 4u;
  const bool bindings_fit =
      caps.max_storage_buffer_range == 0u ||
      (w_bytes <= caps.max_storage_buffer_range &&
       plane_bytes <= caps.max_storage_buffer_range);
  return word_caps() && bindings_fit &&
      !(subgroup_ready() && k == kGreedyK && bindings_fit);
}

const std::vector<float> kScales{0.5f, 0.75f, 1.0f, 1.25f, 0.875f};
const std::vector<float> kBiases{-0.1f, 0.0f, 0.1f, -0.05f, 0.05f};

} // namespace

TEST_CASE("fused greedy word head: token matches the composed argmax") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // K = 2560 forces the word route on subgroup-capable devices too.
  for (int k : {kGreedyK, 2560}) {
    for (int n : {2048, 8192, 12289}) {
      auto in = make_greedy_inputs(
          n, 0xdec46 + n + k, kScales, kBiases, {}, {}, false, stream, k);
      auto run = fused_argmax(in, stream);
      CHECK(run.token == composed_argmax(in, stream));
      CHECK(run.token < static_cast<uint32_t>(n));
      // stats[0] = rows evaluated exactly, stats[1] = overflow flag on
      // the word route; the staged route reports its own stats.
      if (word_route_taken(k, n)) {
        CHECK(run.stats[0] == static_cast<uint32_t>(n));
        CHECK(run.stats[1] == 0u);
      }
    }
  }
}

TEST_CASE("fused greedy word head: equal max rows pick the first index") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  for (int k : {kGreedyK, 2560}) {
    const int n = 12289;
    auto base = make_greedy_inputs(
        n, 0xdec47 + k, kScales, kBiases, {}, {}, false, stream, k);
    uint32_t arg = composed_argmax(base, stream);
    // Duplicate the argmax row at later positions (crossing a chunk edge
    // when it fits): argreduce's first-index rule keeps the original. A
    // last-position argmax gets an earlier duplicate instead, which then
    // becomes the answer.
    std::vector<int> targets;
    uint32_t expected = arg;
    if (arg == static_cast<uint32_t>(n - 1)) {
      targets = {0};
      expected = 0;
    } else {
      if (arg + 4096 < static_cast<uint32_t>(n)) {
        targets.push_back(static_cast<int>(arg) + 4096);
      }
      if (targets.empty() || targets.back() != n - 1) {
        targets.push_back(n - 1);
      }
    }
    std::vector<int> sources(targets.size(), static_cast<int>(arg));
    auto tied = make_greedy_inputs(
        n, 0xdec47 + k, kScales, kBiases, sources, targets, false, stream,
        k);
    CHECK(fused_argmax(tied, stream).token == expected);
    CHECK(composed_argmax(tied, stream) == expected);
  }
}

TEST_CASE("fused greedy word head: NaN row yields the first normalized index") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  for (int k : {kGreedyK, 2560}) {
    const int n = 8192;
    const int groups = k / 64;
    auto in = make_greedy_inputs(
        n, 0xdec48 + k, kScales, kBiases, {}, {}, false, stream, k);
    // One NaN scale makes row 37's logit NaN; nan_max propagates it to
    // the row maximum, so every normalized element is NaN and argreduce
    // answers index 0.
    std::vector<float> nan_scales(n * groups, 0.5f);
    nan_scales[37 * groups] = kNan;
    array sf(nan_scales.data(), Shape{n, groups}, float32);
    GreedyInputs nan_in{
        in.x, in.w, astype(sf, bfloat16, stream), in.biases, in.high};
    auto run = fused_argmax(nan_in, stream);
    CHECK(run.token == 0u);
    CHECK(composed_argmax(nan_in, stream) == 0u);
  }
}

TEST_CASE("fused greedy word head: +inf rows pick the first") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The staged subgroup route (K = 2048 on subgroup devices) diverges
  // from the composed contract on +inf logits — pre-existing, reported
  // separately — so this test exercises legs where the word route
  // actually fires (K = 2560 always; K = 2048 only without subgroup).
  for (int k : {kGreedyK, 2560}) {
    if (!word_route_taken(k, 8192)) {
      continue;
    }
    const int n = 8192;
    const int groups = k / 64;
    // x all-ones keeps every row's dot and input_sum positive, so an
    // infinite scale and bias make that row's logit exactly +inf; the
    // normalized row is NaN at the +inf logits and -inf elsewhere, and
    // argreduce's first-NaN rule answers the first +inf row.
    auto in = make_greedy_inputs(
        n, 0xdec49 + k, {0.5f}, {0.1f}, {}, {}, true, stream, k);
    std::vector<float> inf_scales(n * groups, 0.5f);
    std::vector<float> inf_biases(n * groups, 0.1f);
    for (int r : {100, 2300}) {
      for (int g = 0; g < groups; ++g) {
        inf_scales[r * groups + g] = kInf;
        inf_biases[r * groups + g] = kInf;
      }
    }
    array sf(inf_scales.data(), Shape{n, groups}, float32);
    array bf(inf_biases.data(), Shape{n, groups}, float32);
    GreedyInputs inf_in{in.x,
                        in.w,
                        astype(sf, bfloat16, stream),
                        astype(bf, bfloat16, stream),
                        in.high};
    auto run = fused_argmax(inf_in, stream);
    CHECK(run.token == 100u);
    CHECK(composed_argmax(inf_in, stream) == 100u);
  }
}

TEST_CASE("fused greedy word head: all -inf logits pick index 0") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  for (int k : {kGreedyK, 2560}) {
    const int n = 8192;
    // scale 0 and bias -inf: bias_term = input_sum * -inf = -inf, so
    // every logit is -inf; the normalized row is all-NaN and argreduce
    // answers index 0.
    auto in = make_greedy_inputs(
        n, 0xdec50 + k, {0.0f}, {-kInf}, {}, {}, true, stream, k);
    auto run = fused_argmax(in, stream);
    CHECK(run.token == 0u);
    CHECK(composed_argmax(in, stream) == 0u);
  }
}

TEST_CASE("fused greedy word head: three dispatches on the word route") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  auto in = make_greedy_inputs(
      12289, 0xdec51, kScales, kBiases, {}, {}, false, stream);
  auto run = fused_argmax(in, stream);
  if (word_route_taken(kGreedyK, 12289)) {
    CHECK(run.launched == 3u);
    CHECK(
        omarchy::trace::counters().last_dispatched_kernel.load() ==
        static_cast<int64_t>(
            omarchy::ComputeKernel::QmmVecGreedyWordBF16));
  }
  CHECK(run.token == composed_argmax(in, stream));
}

TEST_CASE("fused greedy word head: K = 2560 (Qwen3.5-4B) matches") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // The word column is K-generic over 64-wide groups; K = 2560 is the
  // 4B head the DEC-46 card serves, and it also crosses to the word
  // route on subgroup-capable devices (the staged route is K = 2048).
  for (int n : {8192, 131072}) {
    auto in = make_greedy_inputs(
        n, 0xdec52 + n, kScales, kBiases, {}, {}, false, stream, 2560);
    auto run = fused_argmax(in, stream);
    if (word_route_taken(2560, n)) {
      CHECK(run.launched == 3u);
      CHECK(
          omarchy::trace::counters().last_dispatched_kernel.load() ==
          static_cast<int64_t>(
              omarchy::ComputeKernel::QmmVecGreedyWordBF16));
    }
    CHECK(run.token == composed_argmax(in, stream));
    CHECK(run.token < static_cast<uint32_t>(n));
  }
}
