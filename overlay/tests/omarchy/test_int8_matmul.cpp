// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// fast::int8_matmul correctness: the shader route (shaders/int8_matmul.comp)
// must match an fp64 reference of the exact-int8 scheme within one bfloat16
// rounding, for the linear and fused-SwiGLU epilogues, on GPU streams.
// The composed fallback (CPU streams) computes the same math through
// per-group float32 matmuls and is exercised by the same checks.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/backend/gpu/device_info.h"

using namespace mlx::core;

namespace {

// One bfloat16 rounding of the fp64 reference is the bar: the in-group
// integer dot is exact in int32 and the fp32 group sums match the
// reference's own fp32 accumulation order.
bool within_one_bf16_round(const array& got, const std::vector<double>& ref) {
  auto got_host = got; // GPU sync happens on the copy below
  std::vector<float> values(got_host.size());
  float* raw = values.data();
  got_host.eval();
  // Buffer reads through the public API: cast to float32 first, which is
  // exact for bfloat16 values.
  auto wide = astype(got_host, float32);
  wide.eval();
  std::memcpy(raw, wide.data<float>(), values.size() * sizeof(float));
  double worst = 0.0;
  double scale = 1.0;
  for (size_t i = 0; i < values.size(); i++) {
    worst = std::max(worst, std::abs(static_cast<double>(values[i]) - ref[i]));
    scale = std::max(scale, std::abs(ref[i]));
  }
  // bfloat16 keeps 8 mantissa bits: one rounding is 2^-8 relative.
  return worst <= scale * 0.004; // 2^-8
}

std::vector<double> reference_linear(
    const std::vector<int8_t>& x,
    const std::vector<int8_t>& w,
    const std::vector<float>& xs,
    const std::vector<float>& ws,
    const std::vector<float>& bias,
    int rows,
    int n,
    int k,
    int group,
    bool swiglu) {
  int n_out = swiglu ? n * 2 : n;
  std::vector<double> acc(rows * n_out, 0.0);
  for (int m = 0; m < rows; m++) {
    for (int r = 0; r < n_out; r++) {
      for (int g = 0; g < k / group; g++) {
        long long dot = 0;
        for (int i = 0; i < group; i++) {
          dot += static_cast<long long>(x[m * k + g * group + i]) *
              static_cast<long long>(w[r * k + g * group + i]);
        }
        acc[m * n_out + r] +=
            static_cast<double>(dot) * static_cast<double>(xs[m * (k / group) + g]);
      }
      acc[m * n_out + r] =
          acc[m * n_out + r] * static_cast<double>(ws[r]) + bias[r];
    }
  }
  if (!swiglu) {
    return acc;
  }
  std::vector<double> out(rows * n);
  for (int m = 0; m < rows; m++) {
    for (int c = 0; c < n; c++) {
      double gate = acc[m * n * 2 + c];
      double value = acc[m * n * 2 + n + c];
      out[m * n + c] = gate / (1.0 + std::exp(-gate)) * value;
    }
  }
  return out;
}

} // namespace

TEST_CASE("int8_matmul writes every output row past the single-dispatch clamp") {
  // One dispatch spawns at most kMaxComputeGroupCountX * kComputeThreadsPerGroup
  // = 65535 * 256 = 16,776,960 threads. rows*n = 3200*8192 = 26,214,400 exceeds
  // that, so an unchunked single dispatch must leave the tail rows unwritten.
  // floor(16776960 / 8192) = 2047: the first bad element is row 2047 col 7936,
  // and rows 2048..3199 are entirely garbage without the chunked dispatch.
  if (!mlx::core::gpu::is_available()) {
    MESSAGE("no GPU device; skipping");
    return;
  }
  constexpr int rows = 3200;
  constexpr int k = 256;
  constexpr int group = 256;
  constexpr int n = 8192;
  constexpr uint32_t kMaxThreadsPerDispatch = 65535u * 256u;
  static_assert(
      static_cast<uint64_t>(rows) * n > kMaxThreadsPerDispatch,
      "test must exceed the one-dispatch thread budget");
  constexpr int first_bad_row = kMaxThreadsPerDispatch / n; // 2047
  constexpr int first_bad_col = kMaxThreadsPerDispatch % n; // 7936

  std::mt19937 rng(11);
  std::uniform_int_distribution<int> values(-127, 127);
  std::uniform_real_distribution<float> scales(0.001f, 0.011f);

  std::vector<int8_t> x(rows * k), w(n * k);
  for (auto& v : x) {
    v = static_cast<int8_t>(values(rng));
  }
  for (auto& v : w) {
    v = static_cast<int8_t>(values(rng));
  }
  std::vector<float> xs(rows * (k / group)), ws(n), bias(n);
  for (auto& v : xs) {
    v = scales(rng);
  }
  for (auto& v : ws) {
    v = scales(rng);
  }
  for (auto& v : bias) {
    v = scales(rng);
  }

  auto mx_x = array(x.data(), {rows, k}, int8);
  auto mx_xs = array(xs.data(), {rows, k / group}, float32);
  auto mx_w = array(w.data(), {n, k}, int8);
  auto mx_ws = array(ws.data(), {n}, float32);
  auto mx_bias = array(bias.data(), {n}, float32);
  auto out = fast::int8_matmul(
      mx_x, mx_xs, mx_w, mx_ws, mx_bias, group, false,
      new_stream(Device::gpu));

  // Exact integer reference on a sample of rows that includes the first bad
  // row (2047), the first fully unwritten row (2048) and the last row (3199).
  const std::vector<int> sampled = {0, 2046, 2047, 2048, 3199};
  auto wide = astype(out, float32);
  wide.eval();
  std::vector<float> got(wide.data<float>(), wide.data<float>() + wide.size());

  int bad_row = -1;
  int bad_col = -1;
  double worst = 0.0;
  for (int m : sampled) {
    for (int col = 0; col < n; col++) {
      long long dot = 0;
      for (int i = 0; i < k; i++) {
        dot += static_cast<long long>(x[m * k + i]) *
            static_cast<long long>(w[col * k + i]);
      }
      double ref = static_cast<double>(static_cast<float>(dot)) *
          static_cast<double>(xs[m]) * static_cast<double>(ws[col]) +
          static_cast<double>(bias[col]);
      double scale = std::max(1.0, std::abs(ref));
      double err = std::abs(static_cast<double>(got[m * n + col]) - ref);
      if (err > scale * 0.004 && bad_row < 0) {
        bad_row = m;
        bad_col = col;
      }
      worst = std::max(worst, err / scale);
    }
  }
  if (bad_row >= 0) {
    MESSAGE("first bad element: row ", bad_row, " col ", bad_col,
            " (expected first bad row ", first_bad_row, " col ", first_bad_col,
            "); worst rel err ", worst);
  }
  CHECK(bad_row < 0);
}

TEST_CASE("int8_matmul matches the fp64 int8 reference") {
  if (!mlx::core::gpu::is_available()) {
    MESSAGE("no GPU device; skipping");
    return;
  }
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> values(-127, 127);
  std::uniform_real_distribution<float> scales(0.001f, 0.011f);
  const int rows = 8;
  const int k = 256;
  const int group = 128;
  const int n = 64;

  std::vector<int8_t> x(rows * k), w(n * k);
  for (auto& v : x) {
    v = static_cast<int8_t>(values(rng));
  }
  for (auto& v : w) {
    v = static_cast<int8_t>(values(rng));
  }
  std::vector<float> xs(rows * (k / group)), ws(n), bias(n);
  for (auto& v : xs) {
    v = scales(rng);
  }
  for (auto& v : ws) {
    v = scales(rng);
  }
  for (auto& v : bias) {
    v = scales(rng);
  }

  auto mx_x = array(x.data(), {rows, k}, int8);
  auto mx_xs = array(xs.data(), {rows, k / group}, float32);
  auto mx_w = array(w.data(), {n, k}, int8);
  auto mx_ws = array(ws.data(), {n}, float32);
  auto mx_bias = array(bias.data(), {n}, float32);
  auto out = fast::int8_matmul(
      mx_x, mx_xs, mx_w, mx_ws, mx_bias, group, false, new_stream(Device::gpu));
  auto ref = reference_linear(x, w, xs, ws, bias, rows, n, k, group, false);
  CHECK(within_one_bf16_round(out, ref));

  // Fused SwiGLU: weight packs [gate; value] rows of 2n.
  std::vector<int8_t> w2(2 * n * k);
  for (auto& v : w2) {
    v = static_cast<int8_t>(values(rng));
  }
  std::vector<float> ws2(2 * n), bias2(2 * n);
  for (auto& v : ws2) {
    v = scales(rng);
  }
  for (auto& v : bias2) {
    v = scales(rng);
  }
  auto mx_w2 = array(w2.data(), {2 * n, k}, int8);
  auto mx_ws2 = array(ws2.data(), {2 * n}, float32);
  auto mx_bias2 = array(bias2.data(), {2 * n}, float32);
  auto out2 = fast::int8_matmul(
      mx_x, mx_xs, mx_w2, mx_ws2, mx_bias2, group, true, new_stream(Device::gpu));
  auto ref2 =
      reference_linear(x, w2, xs, ws2, bias2, rows, n, k, group, true);
  CHECK(within_one_bf16_round(out2, ref2));
}
