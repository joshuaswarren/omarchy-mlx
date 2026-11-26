// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Regression tests for the silent wrong-value defects W9/W10/W11 from the
// 2026-09-02 upstream-suite sweep (receipts/2026-09-02-upstream-suite-
// coverage.md). Every case runs at the shape that actually failed upstream;
// a case that only used smaller shapes is what let these ship.
//
//   W10/W11: fast::scaled_dot_product_attention GQA (n_q_heads a multiple
//     of n_kv_heads with n_kv_heads == 1) returned stride-scrambled values
//     for every head dim; the result was 5-D (kv, repeat) while the output
//     array was 4-D and copy_shared_buffer installed 5-D strides on it.
//     Head dims 72 and 96 here reproduce the upstream head_dim_72/96
//     failures; the D=64 MHA case is the control that already passed.
//   W9: the LayerNorm VJP weight gradient returned NaN / wrong values once
//     rows * 256-column tiles pushed the old single-workgroup kernel past
//     llvmpipe's serial-loop barrier budget. 8x100x1024 NaNs; 8x100x8192 is
//     the upstream test_layer_norm_grad shape. The old kernel was correct
//     at 8x100x32, so the 1024/8192 cases are the regression guards.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
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

bool compute_available() {
  if (!gpu::is_available()) {
    skip(
        "no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1 on"
        " a development machine).");
    return false;
  }
  return true;
}

std::vector<float> flat(const array& value, Stream stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = copy.data<float>();
  return std::vector<float>(data, data + copy.size());
}

// Deterministic pseudo-random values in [0, 1).
std::vector<float> unit_pattern(size_t count, uint32_t seed) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        static_cast<float>(static_cast<double>(state % 10000u) / 10000.0));
  }
  return values;
}

// Host attention reference in double, upstream fallback semantics:
// scale, GQA unflatten, softmax(-1) over additive or bool masks.
std::vector<double> host_attention(
    const std::vector<double>& q,
    const std::vector<double>& k,
    const std::vector<double>& v,
    int batch,
    int q_heads,
    int kv_heads,
    int q_len,
    int kv_len,
    int head_dim,
    double scale,
    const std::vector<double>& additive_mask /* per (b, h, ql, kl) or empty */,
    bool causal) {
  int repeats = q_heads / kv_heads;
  std::vector<double> out;
  out.reserve(batch * q_heads * q_len * head_dim);
  for (int b = 0; b < batch; ++b) {
    for (int kv = 0; kv < kv_heads; ++kv) {
      for (int rep = 0; rep < repeats; ++rep) {
        for (int qi = 0; qi < q_len; ++qi) {
          std::vector<double> scores(kv_len);
          double max_score = -INFINITY;
          for (int ki = 0; ki < kv_len; ++ki) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) {
              dot += q[((((b * q_heads + kv * repeats + rep) * q_len) + qi) *
                        head_dim) +
                       d] *
                  k[((((b * kv_heads + kv) * kv_len) + ki) * head_dim) + d];
            }
            double score = dot * scale;
            if (causal && ki > kv_len - q_len + qi) {
              score = -INFINITY;
            }
            if (!additive_mask.empty()) {
              score += additive_mask[(((b * q_heads + kv * repeats + rep) *
                                           q_len) +
                                      qi) *
                                          kv_len +
                                      ki];
            }
            scores[ki] = score;
            max_score = std::max(max_score, score);
          }
          double total = 0.0;
          for (int ki = 0; ki < kv_len; ++ki) {
            scores[ki] = std::exp(scores[ki] - max_score);
            total += scores[ki];
          }
          for (int d = 0; d < head_dim; ++d) {
            double acc = 0.0;
            for (int ki = 0; ki < kv_len; ++ki) {
              acc += scores[ki] * v[((((b * kv_heads + kv) * kv_len) + ki) *
                                     head_dim) +
                                    d];
            }
            out.push_back(acc / total);
          }
        }
      }
    }
  }
  return out;
}

void require_close(
    const std::vector<float>& got,
    const std::vector<double>& want,
    double tolerance,
    const std::string& what) {
  REQUIRE_EQ(got.size(), want.size());
  size_t bad = 0;
  for (size_t index = 0; index < want.size(); ++index) {
    if (std::isnan(static_cast<double>(got[index]))) {
      ++bad;
      continue;
    }
    double diff = std::abs(static_cast<double>(got[index]) - want[index]);
    if (diff > tolerance) {
      ++bad;
    }
  }
  CHECK_MESSAGE(
      bad == 0,
      what,
      ": ",
      bad,
      " of ",
      want.size(),
      " elements outside tolerance; first bad value ",
      got[0]);
}

} // namespace

TEST_CASE("sdpa GQA matches host math at the upstream failing shapes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Upstream test_sdpa_head_dim_72/96 shapes: B=1, qH=8, kH=2, qL=64,
  // kL=128. The D=64 MHA control mirrors test_sdpa, which passed before
  // the fix; every GQA case here scrambled on the old code.
  struct Case {
    int head_dim;
    int q_heads;
    int kv_heads;
    bool causal;
  };
  std::vector<Case> cases = {
      {72, 8, 2, false},
      {72, 8, 2, true},
      {96, 8, 2, false},
      {64, 8, 2, false},
      {64, 24, 24, false}, // MHA control
  };
  const int batch = 1;
  const int q_len = 64;
  const int kv_len = 128;
  for (const auto& test : cases) {
    size_t q_count = batch * test.q_heads * q_len * test.head_dim;
    size_t kv_count = batch * test.kv_heads * kv_len * test.head_dim;
    auto q_data = unit_pattern(q_count, 101u + test.head_dim);
    auto k_data = unit_pattern(kv_count, 211u + test.head_dim);
    auto v_data = unit_pattern(kv_count, 307u + test.head_dim);
    array q = array(
        q_data.begin(),
        Shape{batch, test.q_heads, q_len, test.head_dim},
        float32);
    array k = array(
        k_data.begin(),
        Shape{batch, test.kv_heads, kv_len, test.head_dim},
        float32);
    array v = array(
        v_data.begin(),
        Shape{batch, test.kv_heads, kv_len, test.head_dim},
        float32);
    double scale = 1.0 / std::sqrt(static_cast<double>(test.head_dim));
    std::vector<double> q_host(q_data.begin(), q_data.end());
    std::vector<double> k_host(k_data.begin(), k_data.end());
    std::vector<double> v_host(v_data.begin(), v_data.end());
    array out = fast::scaled_dot_product_attention(
        q,
        k,
        v,
        static_cast<float>(scale),
        test.causal ? std::string("causal") : std::string(""),
        std::nullopt,
        std::nullopt,
        false,
        stream);
    auto got = flat(out, stream);
    auto want = host_attention(
        q_host,
        k_host,
        v_host,
        batch,
        test.q_heads,
        test.kv_heads,
        q_len,
        kv_len,
        test.head_dim,
        scale,
        {},
        test.causal);
    require_close(
        got,
        want,
        1e-4,
        "sdpa head_dim=" + std::to_string(test.head_dim) + " qH=" +
            std::to_string(test.q_heads) + " kH=" +
            std::to_string(test.kv_heads) +
            (test.causal ? " causal" : " no-mask"));
  }
}

TEST_CASE("LayerNorm VJP dw matches host math at the upstream shapes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  struct Grid {
    int rows;
    int cols;
  };
  const float eps = 1e-5f;
  std::vector<Grid> shapes = {{8, 32}, {8, 1024}, {8, 8192}};
  uint32_t seed = 53u;
  for (const auto& grid : shapes) {
    size_t count = static_cast<size_t>(grid.rows) * grid.cols;
    auto x_data = unit_pattern(count, seed);
    auto w_data = unit_pattern(grid.cols, seed + 1u);
    auto b_data = unit_pattern(grid.cols, seed + 2u);
    auto y_data = unit_pattern(count, seed + 3u);
    seed += 7u;
    array x = array(x_data.begin(), Shape{grid.rows, grid.cols}, float32);
    array w = array(w_data.begin(), Shape{grid.cols}, float32);
    array b = array(b_data.begin(), Shape{grid.cols}, float32);
    array y = array(y_data.begin(), Shape{grid.rows, grid.cols}, float32);

    auto fun = [&](const std::vector<array>& inputs) {
      return std::vector<array>{
          fast::layer_norm(inputs[0], inputs[1], inputs[2], eps, stream)};
    };
    auto [outputs, grads] = vjp(fun, std::vector<array>{x, w, b}, {y});
    auto got_dw = flat(grads[1], stream);

    // Host math in double.
    std::vector<double> dw_host(grid.cols, 0.0);
    for (int row = 0; row < grid.rows; ++row) {
      double sum = 0.0;
      double sum_sq = 0.0;
      for (int col = 0; col < grid.cols; ++col) {
        double value = x_data[row * grid.cols + col];
        sum += value;
        sum_sq += value * value;
      }
      double mu = sum / grid.cols;
      double variance = sum_sq / grid.cols - mu * mu;
      double norm = 1.0 / std::sqrt(variance + eps);
      double wg_sum = 0.0;
      double wgxc_sum = 0.0;
      for (int col = 0; col < grid.cols; ++col) {
        double value = x_data[row * grid.cols + col];
        double grad = y_data[row * grid.cols + col];
        double wg = w_data[col] * grad;
        wg_sum += wg;
        wgxc_sum += wg * (value - mu);
      }
      double mean_wg = wg_sum / grid.cols;
      double mean_wgxc = wgxc_sum / grid.cols;
      for (int col = 0; col < grid.cols; ++col) {
        double value = x_data[row * grid.cols + col];
        double grad = y_data[row * grid.cols + col];
        double xc = value - mu;
        dw_host[col] += grad * xc * norm;
      }
    }
    require_close(
        got_dw,
        dw_host,
        5e-3,
        "layer_norm vjp dw " + std::to_string(grid.rows) + "x" +
            std::to_string(grid.cols));
  }
}
