// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Wave 9 fast-op tests: fused RMSNorm/LayerNorm forward and VJP kernels, the
// cross-entropy VJP, FP8 (E4M3) conversion, and the composed-path anchors.
// Every value test names its reference:
//   - "finite differences": central differences through the real device
//     kernel, with the perturbed math evaluated in double on the host,
//   - "composed formula": the upstream mlx/fast.cpp fallback algebra rebuilt
//     from core mlx ops in the test (the graph the composed fallback ran
//     before these primitives went native),
//   - "host math": the closed-form math computed in double in this file,
//   - "upstream bit algorithm": the ConvertFP8 CPU bit twiddling, scalarized
//     in this file.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <tuple>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/cpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
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

std::string caught_message(const std::function<void()>& body) {
  try {
    body();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

void require_close(
    const std::vector<float>& got,
    const std::vector<double>& want,
    double tolerance,
    const std::string& what) {
  REQUIRE_EQ(got.size(), want.size());
  for (size_t index = 0; index < want.size(); ++index) {
    double diff = std::abs(static_cast<double>(got[index]) - want[index]);
    CHECK_MESSAGE(
        diff <= tolerance,
        what,
        " element ",
        index,
        ": got ",
        got[index],
        " want ",
        want[index]);
  }
}

std::vector<float> flat(const array& value, Stream stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = copy.data<float>();
  return std::vector<float>(data, data + copy.size());
}

// Deterministic pseudo-random values in [-1, 1).
std::vector<float> pattern(size_t count, uint32_t seed) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        static_cast<float>(static_cast<double>(state % 20000u) / 10000.0) -
        1.0f);
  }
  return values;
}

std::vector<double> widen(const std::vector<float>& values) {
  return std::vector<double>(values.begin(), values.end());
}

// ---- host math references (double) ----

struct RowStats {
  double norm;
  double mu;
};

RowStats host_row_stats(
    const std::vector<double>& x,
    size_t row,
    size_t cols,
    double eps,
    bool layer) {
  double sum = 0.0;
  double sum_sq = 0.0;
  for (size_t col = 0; col < cols; ++col) {
    double value = x[row * cols + col];
    sum += value;
    sum_sq += value * value;
  }
  double mu = sum / cols;
  RowStats stats;
  stats.mu = mu;
  double mean_square = sum_sq / cols;
  stats.norm =
      1.0 / std::sqrt((layer ? (mean_square - mu * mu) : mean_square) + eps);
  return stats;
}

std::vector<double> host_norm(
    const std::vector<double>& x,
    const std::vector<double>& w,
    const std::vector<double>& b,
    size_t rows,
    size_t cols,
    double eps,
    bool layer) {
  std::vector<double> out(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    auto stats = host_row_stats(x, row, cols, eps, layer);
    for (size_t col = 0; col < cols; ++col) {
      double xc =
          layer ? (x[row * cols + col] - stats.mu) : x[row * cols + col];
      out[row * cols + col] =
          xc * stats.norm * w[col] + (layer ? b[col] : 0.0);
    }
  }
  return out;
}

// The upstream mlx/fast.cpp VJP fallback algebra, host math.
std::vector<double> host_norm_vjp_dx(
    const std::vector<double>& x,
    const std::vector<double>& w,
    const std::vector<double>& g,
    size_t rows,
    size_t cols,
    double eps,
    bool layer) {
  std::vector<double> dx(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    auto stats = host_row_stats(x, row, cols, eps, layer);
    double n3 = stats.norm * stats.norm * stats.norm;
    double sum_wg = 0.0;
    double sum_wgxc = 0.0;
    for (size_t col = 0; col < cols; ++col) {
      size_t index = row * cols + col;
      double wg = w[col] * g[index];
      double xc = layer ? (x[index] - stats.mu) : x[index];
      sum_wg += wg;
      sum_wgxc += wg * xc;
    }
    for (size_t col = 0; col < cols; ++col) {
      size_t index = row * cols + col;
      double wg = w[col] * g[index];
      double xc = layer ? (x[index] - stats.mu) : x[index];
      if (layer) {
        dx[index] =
            (wg - sum_wg / cols) * stats.norm - xc * (sum_wgxc / cols) * n3;
      } else {
        dx[index] = wg * stats.norm - x[index] * (sum_wgxc / cols) * n3;
      }
    }
  }
  return dx;
}

std::vector<double> host_norm_vjp_dw(
    const std::vector<double>& x,
    const std::vector<double>& g,
    size_t rows,
    size_t cols,
    double eps,
    bool layer) {
  std::vector<double> dw(cols, 0.0);
  for (size_t row = 0; row < rows; ++row) {
    auto stats = host_row_stats(x, row, cols, eps, layer);
    for (size_t col = 0; col < cols; ++col) {
      size_t index = row * cols + col;
      double xc = layer ? (x[index] - stats.mu) : x[index];
      dw[col] += g[index] * xc * stats.norm;
    }
  }
  return dw;
}

std::vector<double> host_norm_vjp_db(
    const std::vector<double>& g, size_t rows, size_t cols) {
  std::vector<double> db(cols, 0.0);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      db[col] += g[row * cols + col];
    }
  }
  return db;
}

// Upstream CrossEntropy::vjp fallback algebra, host math.
std::vector<double> host_cross_entropy_vjp(
    const std::vector<double>& x,
    const std::vector<int>& y,
    const std::vector<double>& g,
    size_t rows,
    size_t cols) {
  std::vector<double> gx(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    double max_value = x[row * cols];
    for (size_t col = 1; col < cols; ++col) {
      max_value = std::max(max_value, x[row * cols + col]);
    }
    double sum_exp = 0.0;
    for (size_t col = 0; col < cols; ++col) {
      sum_exp += std::exp(x[row * cols + col] - max_value);
    }
    double lse = max_value + std::log(sum_exp);
    for (size_t col = 0; col < cols; ++col) {
      double p = std::exp(x[row * cols + col] - lse);
      double onehot = (static_cast<int>(col) == y[row]) ? 1.0 : 0.0;
      gx[row * cols + col] = g[row] * (p - onehot);
    }
  }
  return gx;
}

// ---- finite differences through the real device kernel ----

double central_difference(
    const std::vector<double>& base,
    size_t element,
    double h,
    const std::function<double(const std::vector<double>&)>& objective) {
  std::vector<double> plus = base;
  std::vector<double> minus = base;
  plus[element] += h;
  minus[element] -= h;
  return (objective(plus) - objective(minus)) / (2.0 * h);
}

// ---- FP8 host reference: the upstream bit algorithm, scalarized ----

uint8_t host_to_fp8(double value) {
  float f = static_cast<float>(value);
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  uint32_t fp8_max = 543u << 21u;
  uint32_t denorm_mask = 141u << 23u;
  uint32_t sign = bits & 0x80000000u;
  bits ^= sign;
  float low_input;
  std::memcpy(&low_input, &bits, sizeof(low_input));
  float denorm_bias;
  std::memcpy(&denorm_bias, &denorm_mask, sizeof(denorm_bias));
  float low_sum = low_input + denorm_bias;
  uint32_t f_bits_low;
  std::memcpy(&f_bits_low, &low_sum, sizeof(f_bits_low));
  uint32_t result_low = (f_bits_low - denorm_mask) & 0xFFu;
  uint32_t mant_odd = (bits >> 20u) & 1u;
  uint32_t f_bits_high = bits + (((7u - 127u) << 23u) + 0x7FFFFu);
  f_bits_high += mant_odd;
  uint32_t result_high = (f_bits_high >> 20u) & 0xFFu;
  uint32_t result = (bits < (121u << 23u)) ? result_low : result_high;
  result = (bits >= fp8_max) ? 0x7Eu : result;
  return static_cast<uint8_t>(result | (sign >> 24u));
}

} // namespace

// ---- forward kernels against host math ----

TEST_CASE("native RMSNorm matches host math on f32 rows") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 3;
  const size_t cols = 8;
  const float eps = 1e-5f;
  auto x_data = pattern(rows * cols, 7);
  auto w_data = pattern(cols, 11);

  array x = array(x_data.begin(), Shape{int(rows), int(cols)}, float32);
  array w = array(w_data.begin(), Shape{int(cols)}, float32);
  auto got = flat(fast::rms_norm(x, w, eps, stream), stream);

  std::vector<double> x_host = widen(x_data);
  std::vector<double> w_host = widen(w_data);
  std::vector<double> w_broad(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      w_broad[row * cols + col] = w_host[col];
    }
  }
  std::vector<double> zeros(rows * cols, 0.0);
  auto want = host_norm(x_host, w_broad, zeros, rows, cols, eps, false);
  require_close(got, want, 1e-5, "rms_norm weighted");

  // Weightless form: upstream passes a scalar 1.0 weight.
  auto got_bare = flat(fast::rms_norm(x, std::nullopt, eps, stream), stream);
  std::vector<double> ones(rows * cols, 1.0);
  auto want_bare = host_norm(x_host, ones, zeros, rows, cols, eps, false);
  require_close(got_bare, want_bare, 1e-5, "rms_norm weightless");
}

TEST_CASE("native LayerNorm matches host math on f32 rows") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 4;
  const size_t cols = 6;
  const float eps = 1e-5f;
  auto x_data = pattern(rows * cols, 13);
  auto w_data = pattern(cols, 17);
  auto b_data = pattern(cols, 19);

  array x = array(x_data.begin(), Shape{int(rows), int(cols)}, float32);
  array w = array(w_data.begin(), Shape{int(cols)}, float32);
  array b = array(b_data.begin(), Shape{int(cols)}, float32);
  auto got = flat(fast::layer_norm(x, w, b, eps, stream), stream);

  std::vector<double> x_host = widen(x_data);
  std::vector<double> w_host = widen(w_data);
  std::vector<double> b_host = widen(b_data);
  std::vector<double> w_broad(rows * cols);
  std::vector<double> b_broad(rows * cols);
  for (size_t row = 0; row < rows; ++row) {
    for (size_t col = 0; col < cols; ++col) {
      w_broad[row * cols + col] = w_host[col];
      b_broad[row * cols + col] = b_host[col];
    }
  }
  auto want = host_norm(x_host, w_broad, b_broad, rows, cols, eps, true);
  require_close(got, want, 1e-5, "layer_norm weighted");

  // Weightless and biasless (upstream passes scalar 1 and scalar 0).
  auto got_bare =
      flat(fast::layer_norm(x, std::nullopt, std::nullopt, eps, stream), stream);
  std::vector<double> ones(rows * cols, 1.0);
  std::vector<double> zeros(rows * cols, 0.0);
  auto want_bare = host_norm(x_host, ones, zeros, rows, cols, eps, true);
  require_close(got_bare, want_bare, 1e-5, "layer_norm bare");
}

TEST_CASE("fused norm forward on f16 and bf16 storage") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 2;
  const size_t cols = 8;
  const float eps = 1e-5f;
  auto x_data = pattern(rows * cols, 23);
  std::vector<double> x_host = widen(x_data);
  std::vector<double> ones(rows * cols, 1.0);
  std::vector<double> zeros(rows * cols, 0.0);
  auto want = host_norm(x_host, ones, zeros, rows, cols, eps, false);

  array x16 = astype(
      array(x_data.begin(), Shape{int(rows), int(cols)}, float32),
      float16,
      stream);
  auto got16 = flat(fast::rms_norm(x16, std::nullopt, eps, stream), stream);
  require_close(got16, want, 2e-2, "rms_norm f16");

  array xbf = astype(
      array(x_data.begin(), Shape{int(rows), int(cols)}, float32),
      bfloat16,
      stream);
  auto gotbf = flat(fast::rms_norm(xbf, std::nullopt, eps, stream), stream);
  require_close(gotbf, want, 5e-2, "rms_norm bf16");
}

TEST_CASE("RMSNormVJP matches finite differences and the composed formula") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 2;
  const size_t cols = 4;
  const float eps = 1e-5f;
  auto x_data = pattern(rows * cols, 29);
  auto w_data = pattern(cols, 31);
  auto c_data = pattern(rows * cols, 37);

  array x = array(x_data.begin(), Shape{int(rows), int(cols)}, float32);
  array w = array(w_data.begin(), Shape{int(cols)}, float32);
  array cot = array(c_data.begin(), Shape{int(rows), int(cols)}, float32);

  auto fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        fast::rms_norm(inputs[0], inputs[1], eps, stream)};
  };
  auto [outputs, grads] = vjp(fun, std::vector<array>{x, w}, {cot});
  auto got_dx = flat(grads[0], stream);
  auto got_dw = flat(grads[1], stream);

  std::vector<double> x_host = widen(x_data);
  std::vector<double> w_host = widen(w_data);
  std::vector<double> c_host = widen(c_data);
  auto objective_x = [&](const std::vector<double>& point) {
    auto out = host_norm(point, w_host, std::vector<double>(w_host.size(), 0.0),
                         rows, cols, eps, false);
    double dot = 0.0;
    for (size_t index = 0; index < out.size(); ++index) {
      dot += out[index] * c_host[index];
    }
    return dot;
  };
  auto objective_w = [&](const std::vector<double>& point) {
    auto out = host_norm(x_host, point, std::vector<double>(point.size(), 0.0),
                         rows, cols, eps, false);
    double dot = 0.0;
    for (size_t index = 0; index < out.size(); ++index) {
      dot += out[index] * c_host[index];
    }
    return dot;
  };
  double h = 1e-2;
  std::vector<double> fd_dx(x_host.size());
  for (size_t index = 0; index < x_host.size(); ++index) {
    fd_dx[index] = central_difference(x_host, index, h, objective_x);
  }
  std::vector<double> fd_dw(w_host.size());
  for (size_t index = 0; index < w_host.size(); ++index) {
    fd_dw[index] = central_difference(w_host, index, h, objective_w);
  }
  // Reference 1: finite differences through the real kernel.
  require_close(got_dx, fd_dx, 2e-2, "rms vjp dx finite difference");
  require_close(got_dw, fd_dw, 2e-2, "rms vjp dw finite difference");

  // Reference 2: the composed fallback algebra rebuilt from core ops,
  // exactly as mlx/fast.cpp RMSNorm::vjp writes it.
  auto n = rsqrt(
      add(mean(square(x, stream), -1, true, stream),
          array(eps, float32),
          stream),
      stream);
  auto n3 = power(n, array(3.0f, float32), stream);
  auto gw = multiply(cot, w, stream);
  auto t = mean(multiply(gw, x, stream), -1, true, stream);
  auto composed_dx = subtract(
      multiply(gw, n, stream),
      multiply(multiply(x, t, stream), n3, stream),
      stream);
  auto composed_dw = sum(
      multiply(cot, multiply(x, n, stream), stream), 0, false, stream);
  require_close(
      got_dx,
      widen(flat(composed_dx, stream)),
      1e-5,
      "rms vjp dx composed formula");
  require_close(
      got_dw,
      widen(flat(composed_dw, stream)),
      1e-5,
      "rms vjp dw composed formula");

  // Reference 3: the closed-form host algebra.
  require_close(
      got_dx,
      host_norm_vjp_dx(x_host, w_host, c_host, rows, cols, eps, false),
      1e-5,
      "rms vjp dx host math");
  require_close(
      got_dw,
      host_norm_vjp_dw(x_host, c_host, rows, cols, eps, false),
      1e-5,
      "rms vjp dw host math");
}

TEST_CASE("LayerNormVJP matches finite differences and the composed formula") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 2;
  const size_t cols = 4;
  const float eps = 1e-5f;
  auto x_data = pattern(rows * cols, 41);
  auto w_data = pattern(cols, 43);
  auto b_data = pattern(cols, 47);
  auto c_data = pattern(rows * cols, 53);

  array x = array(x_data.begin(), Shape{int(rows), int(cols)}, float32);
  array w = array(w_data.begin(), Shape{int(cols)}, float32);
  array b = array(b_data.begin(), Shape{int(cols)}, float32);
  array cot = array(c_data.begin(), Shape{int(rows), int(cols)}, float32);

  auto fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        fast::layer_norm(inputs[0], inputs[1], inputs[2], eps, stream)};
  };
  auto [outputs, grads] = vjp(fun, std::vector<array>{x, w, b}, {cot});
  auto got_dx = flat(grads[0], stream);
  auto got_dw = flat(grads[1], stream);
  auto got_db = flat(grads[2], stream);

  std::vector<double> x_host = widen(x_data);
  std::vector<double> w_host = widen(w_data);
  std::vector<double> b_host = widen(b_data);
  std::vector<double> c_host = widen(c_data);
  auto dot_with_c = [&](const std::vector<double>& out) {
    double dot = 0.0;
    for (size_t index = 0; index < out.size(); ++index) {
      dot += out[index] * c_host[index];
    }
    return dot;
  };
  auto objective_x = [&](const std::vector<double>& point) {
    return dot_with_c(host_norm(point, w_host, b_host, rows, cols, eps, true));
  };
  auto objective_w = [&](const std::vector<double>& point) {
    return dot_with_c(host_norm(x_host, point, b_host, rows, cols, eps, true));
  };
  auto objective_b = [&](const std::vector<double>& point) {
    return dot_with_c(host_norm(x_host, w_host, point, rows, cols, eps, true));
  };
  double h = 1e-2;
  std::vector<double> fd_dx(x_host.size());
  for (size_t index = 0; index < x_host.size(); ++index) {
    fd_dx[index] = central_difference(x_host, index, h, objective_x);
  }
  std::vector<double> fd_dw(w_host.size());
  for (size_t index = 0; index < w_host.size(); ++index) {
    fd_dw[index] = central_difference(w_host, index, h, objective_w);
  }
  std::vector<double> fd_db(b_host.size());
  for (size_t index = 0; index < b_host.size(); ++index) {
    fd_db[index] = central_difference(b_host, index, h, objective_b);
  }
  require_close(got_dx, fd_dx, 2e-2, "layer vjp dx finite difference");
  require_close(got_dw, fd_dw, 2e-2, "layer vjp dw finite difference");
  require_close(got_db, fd_db, 2e-2, "layer vjp db finite difference");

  // Composed fallback algebra, exactly as mlx/fast.cpp LayerNorm::vjp
  // writes it.
  auto norm_count = number_of_elements(x, {-1}, true, float32, stream);
  auto sumx = sum(x, -1, true, stream);
  auto sumx2 = sum(square(x, stream), -1, true, stream);
  auto mu = multiply(sumx, norm_count, stream);
  auto mu2 = multiply(sumx2, norm_count, stream);
  auto var = subtract(mu2, square(mu, stream), stream);
  auto n = rsqrt(add(var, array(eps, float32), stream), stream);
  auto n3 = power(n, array(3.0f, float32), stream);
  auto x_c = subtract(x, mu, stream);
  auto wg = multiply(w, cot, stream);
  auto sumwg = multiply(sum(wg, -1, true, stream), norm_count, stream);
  auto sumwgxc = multiply(
      sum(multiply(wg, x_c, stream), -1, true, stream), norm_count, stream);
  auto t1 = multiply(multiply(x_c, sumwgxc, stream), n3, stream);
  auto t2 = multiply(subtract(wg, sumwg, stream), n, stream);
  auto composed_dx = subtract(t2, t1, stream);
  auto composed_dw = sum(
      multiply(cot, multiply(x_c, n, stream), stream), 0, false, stream);
  auto composed_db = sum(cot, 0, false, stream);
  require_close(
      got_dx,
      widen(flat(composed_dx, stream)),
      1e-5,
      "layer vjp dx composed formula");
  require_close(
      got_dw,
      widen(flat(composed_dw, stream)),
      1e-5,
      "layer vjp dw composed formula");
  require_close(
      got_db,
      widen(flat(composed_db, stream)),
      1e-5,
      "layer vjp db composed formula");

  require_close(
      got_dx,
      host_norm_vjp_dx(x_host, w_host, c_host, rows, cols, eps, true),
      1e-5,
      "layer vjp dx host math");
  require_close(
      got_dw,
      host_norm_vjp_dw(x_host, c_host, rows, cols, eps, true),
      1e-5,
      "layer vjp dw host math");
  require_close(
      got_db,
      host_norm_vjp_db(c_host, rows, cols),
      1e-5,
      "layer vjp db host math");
}

TEST_CASE("cross_entropy forward matches host math on the composed path") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 3;
  const size_t cols = 7;
  auto logits_data = pattern(rows * cols, 67);
  std::vector<int> targets{0, 3, 6};

  array logits =
      array(logits_data.begin(), Shape{int(rows), int(cols)}, float32);
  array targets_arr = array(targets.begin(), Shape{int(rows)}, int32);
  auto got = flat(fast::cross_entropy(logits, targets_arr, stream), stream);

  // Reference: host math lse - score.
  std::vector<double> want(rows);
  std::vector<double> logit_host = widen(logits_data);
  for (size_t row = 0; row < rows; ++row) {
    double max_value = logit_host[row * cols];
    for (size_t col = 1; col < cols; ++col) {
      max_value = std::max(max_value, logit_host[row * cols + col]);
    }
    double sum_exp = 0.0;
    for (size_t col = 0; col < cols; ++col) {
      sum_exp += std::exp(logit_host[row * cols + col] - max_value);
    }
    want[row] = max_value + std::log(sum_exp) -
        logit_host[row * cols + size_t(targets[row])];
  }
  require_close(got, want, 1e-5, "cross_entropy composed forward");
}

TEST_CASE("CrossEntropyVJP matches finite differences and the composed formula") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const size_t rows = 2;
  const size_t cols = 5;
  auto logits_data = pattern(rows * cols, 71);
  std::vector<int> targets{1, 4};
  auto cot_data = pattern(rows, 73);
  array logits =
      array(logits_data.begin(), Shape{int(rows), int(cols)}, float32);
  array targets_arr = array(targets.begin(), Shape{int(rows)}, int32);
  array cot = array(cot_data.begin(), Shape{int(rows)}, float32);
  // Targets are constants in training: capture them so the VJP runs with
  // respect to the logits only, with an arbitrary cotangent.
  auto fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        fast::cross_entropy(inputs[0], targets_arr, stream)};
  };
  auto [outputs, grads] = vjp(fun, std::vector<array>{logits}, {cot});
  auto got = flat(grads[0], stream);

  std::vector<double> logit_host = widen(logits_data);
  std::vector<double> cot_host = widen(cot_data);
  auto objective = [&](const std::vector<double>& point) {
    // Objective: dot(loss, cotangent), loss_row = lse_row - x_row[y_row].
    double total = 0.0;
    for (size_t row = 0; row < rows; ++row) {
      double max_value = point[row * cols];
      for (size_t col = 1; col < cols; ++col) {
        max_value = std::max(max_value, point[row * cols + col]);
      }
      double sum_exp = 0.0;
      for (size_t col = 0; col < cols; ++col) {
        sum_exp += std::exp(point[row * cols + col] - max_value);
      }
      total += cot_host[row] *
          (max_value + std::log(sum_exp) -
           point[row * cols + size_t(targets[row])]);
    }
    return total;
  };
  double h = 1e-2;
  std::vector<double> fd(logits_data.size());
  for (size_t index = 0; index < logits_data.size(); ++index) {
    fd[index] = central_difference(logit_host, index, h, objective);
  }
  require_close(got, fd, 2e-2, "cross entropy vjp finite difference");

  // Composed formula from upstream CrossEntropy::vjp: g * (p - onehot).
  auto score = squeeze(
      take_along_axis(
          logits, expand_dims(targets_arr, -1, stream), -1, stream),
      -1,
      stream);
  auto lse = add(fast::cross_entropy(logits, targets_arr, stream),
                 score,
                 stream);
  auto p = exp(subtract(logits, expand_dims(lse, -1, stream), stream), stream);
  Shape class_shape{1, int(cols)};
  auto onehot = astype(
      equal(
          expand_dims(targets_arr, -1, stream),
          reshape(arange(0, int(cols), int32, stream), class_shape, stream),
          stream),
      float32,
      stream);
  auto composed = multiply(
      expand_dims(cot, -1, stream), subtract(p, onehot, stream), stream);
  require_close(
      got,
      widen(flat(composed, stream)),
      1e-5,
      "cross entropy vjp composed formula");

  require_close(
      got,
      host_cross_entropy_vjp(logit_host, targets, cot_host, rows, cols),
      1e-5,
      "cross entropy vjp host math");
}

TEST_CASE("CrossEntropyVJP accepts strided inputs and a broadcast cotangent") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array logits = transpose(
      array({1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}, Shape{3, 2}), stream);
  array targets = slice(array({0, 99, 2, 99}, int32), {0}, {4}, {2}, stream);
  array cot = broadcast_to(array(0.5f), Shape{2}, stream);
  auto fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{fast::cross_entropy(inputs[0], targets, stream)};
  };
  auto [outputs, grads] = vjp(fun, std::vector<array>{logits}, {cot});
  require_close(
      flat(grads[0], stream),
      host_cross_entropy_vjp({1, 2, 3, 4, 5, 6}, {0, 2}, {0.5, 0.5}, 2, 3),
      1e-5,
      "strided cross entropy vjp host math");
}

TEST_CASE("native-order decode SDPA handles the strided Qwen KV cache") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  constexpr int heads = 14;
  constexpr int kv_heads = 2;
  constexpr int keys = 263;
  constexpr int capacity = 320;
  constexpr int width = 64;
  std::vector<float> q_values(heads * width, 0.0f);
  std::vector<float> k_values(kv_heads * capacity * width, 0.0f);
  std::vector<float> v_values(kv_heads * capacity * width, 0.5f);
  array q = astype(
      array(q_values.begin(), Shape{1, heads, 1, width}, float32),
      float16,
      stream);
  array k_cache = astype(
      array(k_values.begin(), Shape{1, kv_heads, capacity, width}, float32),
      float16,
      stream);
  array v_cache = astype(
      array(v_values.begin(), Shape{1, kv_heads, capacity, width}, float32),
      float16,
      stream);
  array k = slice(
      k_cache, {0, 0, 0, 0}, {1, kv_heads, keys, width}, stream);
  array v = slice(
      v_cache, {0, 0, 0, 0}, {1, kv_heads, keys, width}, stream);
  eval({q, k, v});
  omarchy::get_command_encoder(stream).synchronize();

  setenv("MLX_OMARCHY_SDPA_DECODE_NATIVE", "1", 1);
  uint64_t before = omarchy::trace::counters().vk_compute_dispatches.load();
  array output = fast::scaled_dot_product_attention(
      q, k, v, 1.0f / std::sqrt(float(width)), "", {}, std::nullopt, false,
      stream);
  output.eval();
  omarchy::get_command_encoder(stream).synchronize();
  uint64_t dispatches =
      omarchy::trace::counters().vk_compute_dispatches.load() - before;
  unsetenv("MLX_OMARCHY_SDPA_DECODE_NATIVE");

  const auto& caps = omarchy::device(0).capabilities();
  constexpr VkSubgroupFeatureFlags required =
      VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
  if (caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32u &&
      (caps.subgroup_operations & required) == required) {
    CHECK_EQ(dispatches, 1);
  }
  for (float value : flat(output, stream)) {
    CHECK(std::abs(value - 0.5f) < 5e-4f);
  }
}

TEST_CASE("scaled_dot_product_attention backward matches finite differences") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Composed anchor: fast::ScaledDotProductAttentionVJP keeps use_fallback
  // true, so autograd runs the composed fallback graph over the same
  // Elementwise, Matmul, and Softmax kernels the forward uses. Reference:
  // finite differences of the attention output, host math.
  const int head_dim = 4;
  const float scale = 1.0f / std::sqrt(float(head_dim));
  auto q_data = pattern(2 * 1 * 2 * head_dim, 79);
  auto k_data = pattern(2 * 1 * 2 * head_dim, 83);
  auto v_data = pattern(2 * 1 * 2 * head_dim, 89);
  auto cot_data = pattern(2 * 1 * 2 * head_dim, 97);

  Shape shape{2, 1, 2, head_dim};
  array q = array(q_data.begin(), shape, float32);
  array k = array(k_data.begin(), shape, float32);
  array v = array(v_data.begin(), shape, float32);
  array cot = array(cot_data.begin(), shape, float32);

  auto fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{
        fast::scaled_dot_product_attention(
            inputs[0], inputs[1], inputs[2], scale, "", {}, std::nullopt, false, stream)};
  };
  auto [outputs, grads] = vjp(fun, std::vector<array>{q, k, v}, {cot});
  auto got_dq = flat(grads[0], stream);

  std::vector<double> q_host = widen(q_data);
  std::vector<double> k_host = widen(k_data);
  std::vector<double> v_host = widen(v_data);
  std::vector<double> cot_host = widen(cot_data);
  auto attention = [&](const std::vector<double>& qq) {
    std::vector<double> out(2 * 1 * 2 * head_dim);
    for (int batch = 0; batch < 2; ++batch) {
      for (int q_pos = 0; q_pos < 2; ++q_pos) {
        double max_score = -1e30;
        std::vector<double> scores(2);
        for (int k_pos = 0; k_pos < 2; ++k_pos) {
          double dot = 0.0;
          for (int dim = 0; dim < head_dim; ++dim) {
            dot += qq[((batch * 2 + q_pos) * head_dim) + dim] *
                k_host[((batch * 2 + k_pos) * head_dim) + dim];
          }
          scores[k_pos] = dot * scale;
          max_score = std::max(max_score, scores[k_pos]);
        }
        double sum_exp = 0.0;
        for (auto& score : scores) {
          score = std::exp(score - max_score);
          sum_exp += score;
        }
        for (int dim = 0; dim < head_dim; ++dim) {
          double acc = 0.0;
          for (int k_pos = 0; k_pos < 2; ++k_pos) {
            acc += scores[k_pos] / sum_exp *
                v_host[((batch * 2 + k_pos) * head_dim) + dim];
          }
          out[(batch * 2 + q_pos) * head_dim + dim] = acc;
        }
      }
    }
    return out;
  };
  auto objective_q = [&](const std::vector<double>& point) {
    auto out = attention(point);
    double dot = 0.0;
    for (size_t index = 0; index < out.size(); ++index) {
      dot += out[index] * cot_host[index];
    }
    return dot;
  };
  double h = 1e-2;
  std::vector<double> fd_dq(q_host.size());
  for (size_t index = 0; index < q_host.size(); ++index) {
    fd_dq[index] = central_difference(q_host, index, h, objective_q);
  }
  require_close(got_dq, fd_dq, 2e-2, "sdpa dq finite difference");
}


// ---- SDPA host reference (double) ----

// Host attention in double precision. q: [B,H,qL,D]; k,v: [B,KV,kL,D],
// both compact row-major. causal applies the upstream offset kL-qL
// (negative offsets fully mask the leading rows); sinks carries one
// logit per query head and rides the softmax denominator the way
// upstream's concatenate-softmax-slice composition does.
std::vector<double> host_sdpa(
    const std::vector<float>& q_data,
    const std::vector<float>& k_data,
    const std::vector<float>& v_data,
    int B,
    int H,
    int KV,
    int qL,
    int kL,
    int D,
    float scale,
    bool causal,
    const std::vector<float>& sinks = {},
    const std::vector<float>& mask = {},
    int mask_batches = 1,
    int mask_heads = 1) {
  const int rep = H / KV;
  const int offset = kL - qL;
  std::vector<double> out(B * H * qL * D, 0.0);
  for (int b = 0; b < B; ++b) {
    for (int kv = 0; kv < KV; ++kv) {
      for (int r = 0; r < rep; ++r) {
        int head = kv * rep + r;
        for (int qi = 0; qi < qL; ++qi) {
          double max_score = -1e30;
          std::vector<double> scores(kL);
          for (int ki = 0; ki < kL; ++ki) {
            double dot = 0.0;
            for (int d = 0; d < D; ++d) {
              dot += q_data[((b * H + head) * qL + qi) * D + d] *
                  k_data[((b * KV + kv) * kL + ki) * D + d];
            }
            scores[ki] = dot * scale;
            if (causal && offset + qi < ki) {
              scores[ki] = -1e30;
            }
            if (!mask.empty()) {
              int mb = mask_batches == 1 ? 0 : b;
              int mh = mask_heads == 1 ? 0 : head;
              scores[ki] +=
                  mask[((mb * mask_heads + mh) * qL + qi) * kL + ki];
            }
            max_score = std::max(max_score, scores[ki]);
          }
          double sink_exp = 0.0;
          if (!sinks.empty()) {
            max_score = std::max(max_score, static_cast<double>(sinks[head]));
            sink_exp = std::exp(
                static_cast<double>(sinks[head]) - max_score);
          }
          double sum = 0.0;
          for (auto& score : scores) {
            score = std::exp(score - max_score);
            sum += score;
          }
          sum += sink_exp;
          for (int d = 0; d < D; ++d) {
            double acc = 0.0;
            for (int ki = 0; ki < kL; ++ki) {
              acc += scores[ki] / sum *
                  v_data[((b * KV + kv) * kL + ki) * D + d];
            }
            out[((b * H + head) * qL + qi) * D + d] = acc;
          }
        }
      }
    }
  }
  return out;
}

TEST_CASE("scaled_dot_product_attention primitive broadcasts additive masks") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int B = 2, H = 4, KV = 2, qL = 3, kL = 5, D = 8;
  const int repeats = H / KV;
  const float scale = 1.0f / std::sqrt(float(D));
  auto q_data = pattern(B * H * qL * D, 233);
  auto k_data = pattern(B * KV * kL * D, 239);
  auto v_data = pattern(B * KV * kL * D, 241);
  auto mask_data = pattern(B * qL * kL, 251);
  array q(q_data.begin(), Shape{B, H, qL, D}, float32);
  array k(k_data.begin(), Shape{B, KV, kL, D}, float32);
  array v(v_data.begin(), Shape{B, KV, kL, D}, float32);
  array mask(mask_data.begin(), Shape{B, 1, qL, kL}, float32);
  auto fallback = [=](std::vector<array> inputs) {
    auto fq = reshape(
        multiply(inputs[0], array(scale, float32), stream),
        Shape{B, KV, repeats, qL, D},
        stream);
    auto fk = expand_dims(inputs[1], 2, stream);
    auto fv = expand_dims(inputs[2], 2, stream);
    auto fm = expand_dims(inputs[3], -3, stream);
    auto scores = add(
        matmul(fq, swapaxes(fk, -1, -2, stream), stream), fm, stream);
    scores = softmax(scores, std::vector<int>{-1}, true, stream);
    auto result = matmul(scores, fv, stream);
    return std::vector<array>{
        reshape(result, Shape{B, H, qL, D}, stream)};
  };
  auto primitive = std::make_shared<fast::ScaledDotProductAttention>(
      stream, fallback, scale, false, false, false, false);
  array out(
      Shape{B, H, qL, D},
      float32,
      primitive,
      std::vector<array>{q, k, v, mask});
  require_close(
      flat(out, stream),
      host_sdpa(
          q_data,
          k_data,
          v_data,
          B,
          H,
          KV,
          qL,
          kL,
          D,
          scale,
          false,
          {},
          mask_data,
          B,
          1),
      1e-5,
      "sdpa primitive additive mask batch/head broadcast gqa");
}

TEST_CASE("scaled_dot_product_attention broadcasts additive masks through GQA") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int B = 2, H = 4, KV = 1, qL = 3, kL = 5, D = 8;
  const float scale = 1.0f / std::sqrt(float(D));
  auto q_data = pattern(B * H * qL * D, 211);
  auto k_data = pattern(B * KV * kL * D, 223);
  auto v_data = pattern(B * KV * kL * D, 227);
  auto mask_data = pattern(B * qL * kL, 229);
  for (size_t index = 0; index < mask_data.size(); ++index) {
    mask_data[index] *= (index % 3 == 0) ? 8.0f : 0.25f;
  }
  struct Bf16FastGuard {
    Bf16FastGuard() {
      if (const char* value = std::getenv("MLX_OMARCHY_SDPA_BF16_FAST")) {
        original = value;
      }
      unsetenv("MLX_OMARCHY_SDPA_BF16_FAST");
    }
    ~Bf16FastGuard() {
      if (original.empty()) {
        unsetenv("MLX_OMARCHY_SDPA_BF16_FAST");
      } else {
        setenv("MLX_OMARCHY_SDPA_BF16_FAST", original.c_str(), 1);
      }
    }
    std::string original;
  } bf16_guard;
  auto check = [&](Dtype dtype, double tolerance, const char* label) {
    array q = astype(
        array(q_data.begin(), Shape{B, H, qL, D}, float32), dtype, stream);
    array k = astype(
        array(k_data.begin(), Shape{B, KV, kL, D}, float32), dtype, stream);
    array v = astype(
        array(v_data.begin(), Shape{B, KV, kL, D}, float32), dtype, stream);
    array mask = astype(
        swapaxes(
            array(mask_data.begin(), Shape{1, B, qL, kL}, float32),
            0,
            1,
            stream),
        dtype,
        stream);
    auto q_ref = flat(q, stream);
    auto k_ref = flat(k, stream);
    auto v_ref = flat(v, stream);
    auto mask_ref = flat(mask, stream);
    auto out = fast::scaled_dot_product_attention(
        q, k, v, scale, "", mask, std::nullopt, false, stream);
    require_close(
        flat(out, stream),
        host_sdpa(
            q_ref,
            k_ref,
            v_ref,
            B,
            H,
            KV,
            qL,
            kL,
            D,
            scale,
            false,
            {},
            mask_ref,
            B,
            1),
        tolerance,
        label);
  };
  check(float32, 1e-5, "sdpa additive mask broadcast gqa f32");
  check(float16, 2e-2, "sdpa additive mask broadcast gqa f16");
  check(bfloat16, 5e-2, "sdpa additive mask broadcast gqa bf16 wide");
  setenv("MLX_OMARCHY_SDPA_BF16_FAST", "1", 1);
  check(bfloat16, 5e-2, "sdpa additive mask broadcast gqa bf16 fast");
}

TEST_CASE("scaled_dot_product_attention bf16 fast scores scale through MatmulBF16Coopmat") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // qL >= 32 with k = D = 8 puts the alpha-scaled scores matmul inside
  // the MatmulBF16Coopmat gate (dispatch_matmul relaxes the coopmat
  // alpha == 1 requirement for bf16). The kernel must scale its f32
  // accumulator by alpha at the drain: a missing scale feeds the
  // softmax unscaled scores and fails this comparison by orders of
  // magnitude, not by a rounding margin.
  struct Bf16FastGuard {
    Bf16FastGuard() {
      if (const char* value = std::getenv("MLX_OMARCHY_SDPA_BF16_FAST")) {
        original = value;
      }
      setenv("MLX_OMARCHY_SDPA_BF16_FAST", "1", 1);
    }
    ~Bf16FastGuard() {
      if (original.empty()) {
        unsetenv("MLX_OMARCHY_SDPA_BF16_FAST");
      } else {
        setenv("MLX_OMARCHY_SDPA_BF16_FAST", original.c_str(), 1);
      }
    }
    std::string original;
  } bf16_guard;
  const int B = 1, H = 2, KV = 1, qL = 32, kL = 40, D = 8;
  const float scale = 0.25f;
  auto q_data = pattern(B * H * qL * D, 257);
  auto k_data = pattern(B * KV * kL * D, 263);
  auto v_data = pattern(B * KV * kL * D, 269);
  array q = astype(
      array(q_data.begin(), Shape{B, H, qL, D}, float32), bfloat16, stream);
  array k = astype(
      array(k_data.begin(), Shape{B, KV, kL, D}, float32), bfloat16, stream);
  array v = astype(
      array(v_data.begin(), Shape{B, KV, kL, D}, float32), bfloat16, stream);
  auto q_ref = flat(q, stream);
  auto k_ref = flat(k, stream);
  auto v_ref = flat(v, stream);
  auto out = fast::scaled_dot_product_attention(
      q, k, v, scale, "", {}, std::nullopt, false, stream);
  require_close(
      flat(out, stream),
      host_sdpa(q_ref, k_ref, v_ref, B, H, KV, qL, kL, D, scale, false),
      5e-2,
      "sdpa bf16 fast coopmat scores alpha");
}

TEST_CASE("scaled_dot_product_attention causal offset keeps kL below qL") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int B = 2, H = 4, KV = 2, qL = 5, kL = 3, D = 8;
  const float scale = 1.0f / std::sqrt(float(D));
  auto q_data = pattern(B * H * qL * D, 101);
  auto k_data = pattern(B * KV * kL * D, 103);
  auto v_data = pattern(B * KV * kL * D, 107);
  array q = array(q_data.begin(), Shape{B, H, qL, D}, float32);
  array k = array(k_data.begin(), Shape{B, KV, kL, D}, float32);
  array v = array(v_data.begin(), Shape{B, KV, kL, D}, float32);
  auto out = fast::scaled_dot_product_attention(
      q, k, v, scale, "causal", {}, std::nullopt, false, stream);
  auto got = flat(out, stream);
  auto want = host_sdpa(
      q_data, k_data, v_data, B, H, KV, qL, kL, D, scale, true);
  require_close(got, want, 1e-5, "sdpa causal offset including fully masked rows");
}

TEST_CASE("scaled_dot_product_attention folds sinks into the denominator") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  {
    const int B = 2, H = 4, KV = 4, qL = 3, kL = 7, D = 8;
    const float scale = 1.0f / std::sqrt(float(D));
    auto q_data = pattern(B * H * qL * D, 113);
    auto k_data = pattern(B * KV * kL * D, 127);
    auto v_data = pattern(B * KV * kL * D, 131);
    auto sink_data = pattern(H, 137);
    array q = array(q_data.begin(), Shape{B, H, qL, D}, float32);
    array k = array(k_data.begin(), Shape{B, KV, kL, D}, float32);
    array v = array(v_data.begin(), Shape{B, KV, kL, D}, float32);
    array sinks = array(sink_data.begin(), Shape{H}, float32);
    auto out = fast::scaled_dot_product_attention(
        q, k, v, scale, "", {}, sinks, false, stream);
    require_close(
        flat(out, stream),
        host_sdpa(
            q_data,
            k_data,
            v_data,
            B,
            H,
            KV,
            qL,
            kL,
            D,
            scale,
            false,
            sink_data),
        1e-5,
        "sdpa sinks f32");
  }
  {
    const int B = 1, H = 4, KV = 2, qL = 1, kL = 96, D = 16;
    const float scale = 1.0f / std::sqrt(float(D));
    auto q_data = pattern(B * H * qL * D, 139);
    auto k_data = pattern(B * KV * kL * D, 149);
    auto v_data = pattern(B * KV * kL * D, 151);
    auto sink_data = pattern(H, 157);
    array q = astype(
        array(q_data.begin(), Shape{B, H, qL, D}, float32), float16, stream);
    array k = astype(
        array(k_data.begin(), Shape{B, KV, kL, D}, float32), float16, stream);
    array v = astype(
        array(v_data.begin(), Shape{B, KV, kL, D}, float32), float16, stream);
    array sinks = astype(
        array(sink_data.begin(), Shape{H}, float32), float16, stream);
    auto out = fast::scaled_dot_product_attention(
        q, k, v, scale, "", {}, sinks, false, stream);
    require_close(
        flat(out, stream),
        host_sdpa(
            q_data,
            k_data,
            v_data,
            B,
            H,
            KV,
            qL,
            kL,
            D,
            scale,
            false,
            sink_data),
        2e-2,
        "sdpa sinks f16 gqa");
  }
}

TEST_CASE("scaled_dot_product_attention causal matches host on head dims 72 and 96") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  for (auto [D, dtype, tolerance] :
       std::vector<std::tuple<int, Dtype, double>>{
           {96, float32, 1e-5},
           {96, float16, 2e-2},
           {72, float16, 2e-2}}) {
    const int B = 1, H = 4, KV = 2, qL = 6, kL = 11;
    const float scale = 1.0f / std::sqrt(float(D));
    auto q_data = pattern(B * H * qL * D, 163 + D);
    auto k_data = pattern(B * KV * kL * D, 167 + D);
    auto v_data = pattern(B * KV * kL * D, 173 + D);
    array q = astype(
        array(q_data.begin(), Shape{B, H, qL, D}, float32), dtype, stream);
    array k = astype(
        array(k_data.begin(), Shape{B, KV, kL, D}, float32), dtype, stream);
    array v = astype(
        array(v_data.begin(), Shape{B, KV, kL, D}, float32), dtype, stream);
    auto out = fast::scaled_dot_product_attention(
        q, k, v, scale, "causal", {}, std::nullopt, false, stream);
    require_close(
        flat(out, stream),
        host_sdpa(q_data, k_data, v_data, B, H, KV, qL, kL, D, scale, true),
        tolerance,
        "sdpa causal head dim");
  }
}

TEST_CASE("scaled_dot_product_attention causal matches on sliced cache K/V") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int B = 1, H = 4, KV = 2, qL = 3, kL = 11, D = 8;
  const int cache_len = 16;
  const float scale = 1.0f / std::sqrt(float(D));
  auto q_data = pattern(B * H * qL * D, 179);
  auto k_wide = pattern(B * KV * cache_len * D, 181);
  auto v_wide = pattern(B * KV * cache_len * D, 191);
  array q = array(q_data.begin(), Shape{B, H, qL, D}, float32);
  array kw = array(k_wide.begin(), Shape{B, KV, cache_len, D}, float32);
  array vw = array(v_wide.begin(), Shape{B, KV, cache_len, D}, float32);
  array k = astype(
      slice(kw, {0, 0, 0, 0}, {B, KV, kL, D}, {1, 1, 1, 1}, stream),
      float16,
      stream);
  array v = astype(
      slice(vw, {0, 0, 0, 0}, {B, KV, kL, D}, {1, 1, 1, 1}, stream),
      float16,
      stream);
  array q16 = astype(q, float16, stream);
  auto out = fast::scaled_dot_product_attention(
      q16, k, v, scale, "causal", {}, std::nullopt, false, stream);
  auto compact = [&](const std::vector<float>& wide) {
    std::vector<float> rows;
    rows.reserve(size_t(B * KV * kL * D));
    for (int b = 0; b < B; ++b) {
      for (int kv = 0; kv < KV; ++kv) {
        for (int ki = 0; ki < kL; ++ki) {
          for (int d = 0; d < D; ++d) {
            rows.push_back(
                wide[((b * KV + kv) * cache_len + ki) * D + d]);
          }
        }
      }
    }
    return rows;
  };
  require_close(
      flat(out, stream),
      host_sdpa(
          q_data,
          compact(k_wide),
          compact(v_wide),
          B,
          H,
          KV,
          qL,
          kL,
          D,
          scale,
          true),
      2e-2,
      "sdpa sliced cache causal");
}

// Upstream test_sdpa_full_head_dim_256's shape class: head_dim 256,
// GQA, causal, keys at and past 2048. The composed path was already
// correct here (the upstream failure was the test's own mx.where
// reference, see test_select_ops.cpp); this pins it against the double
// host reference. qL 512 with the causal offset keeps the llvmpipe
// matmuls and the host reference under a minute.
TEST_CASE("scaled_dot_product_attention causal head dim 256 at kL 2048 and 4096") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int B = 1, H = 4, KV = 2, qL = 512, D = 256;
  const float scale = 1.0f / std::sqrt(float(D));
  for (int kL : {2048, 4096}) {
    auto q_data = pattern(size_t(B) * H * qL * D, 211 + kL);
    auto k_data = pattern(size_t(B) * KV * kL * D, 223 + kL);
    auto v_data = pattern(size_t(B) * KV * kL * D, 227 + kL);
    array q = array(q_data.begin(), Shape{B, H, qL, D}, float32);
    array k = array(k_data.begin(), Shape{B, KV, kL, D}, float32);
    array v = array(v_data.begin(), Shape{B, KV, kL, D}, float32);
    auto out = fast::scaled_dot_product_attention(
        q, k, v, scale, "causal", {}, std::nullopt, false, stream);
    require_close(
        flat(out, stream),
        host_sdpa(q_data, k_data, v_data, B, H, KV, qL, kL, D, scale, true),
        1e-4,
        "sdpa causal head dim 256 kL " + std::to_string(kL));
  }
}

TEST_CASE("scaled_dot_product_attention respects strided sink storage") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array q = zeros({2, 4, 2, 8}, float32, stream);
  array k = zeros({2, 2, 3, 8}, float32, stream);
  array v = ones({2, 2, 3, 8}, float32, stream);
  array storage = array({99.0f, 0.0f, 99.0f, 1.0f, 99.0f, 2.0f, 99.0f, 3.0f});
  array sinks = slice(storage, {1}, {8}, {2}, stream);
  auto out = fast::scaled_dot_product_attention(
      q, k, v, 1.0f, "", {}, sinks, false, stream);
  std::vector<double> want;
  for (int batch = 0; batch < 2; ++batch) {
    for (int head = 0; head < 4; ++head) {
      want.insert(want.end(), 16, 3.0 / (3.0 + std::exp(double(head))));
    }
  }
  require_close(flat(out, stream), want, 1e-5, "sdpa strided sinks");
}

TEST_CASE("scaled_dot_product_attention floors a fully masked bool mask") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const int B = 2, H = 4, KV = 2, qL = 8, kL = 8, D = 128;
  const float scale = 1.0f / std::sqrt(float(D));
  auto q_data = pattern(B * H * qL * D, 193);
  auto k_data = pattern(B * KV * kL * D, 197);
  auto v_data = pattern(B * KV * kL * D, 199);
  array q = array(q_data.begin(), Shape{B, H, qL, D}, float32);
  array k = array(k_data.begin(), Shape{B, KV, kL, D}, float32);
  array v = array(v_data.begin(), Shape{B, KV, kL, D}, float32);
  array mask = swapaxes(zeros({1, B, qL, kL}, bool_, stream), 0, 1, stream);
  auto out = fast::scaled_dot_product_attention(
      q, k, v, scale, "", mask, std::nullopt, false, stream);
  auto got = flat(out, stream);
  for (size_t index = 0; index < got.size(); ++index) {
    CHECK_MESSAGE(
        std::isfinite(got[index]),
        "fully masked output ",
        index,
        " must stay finite");
  }
  std::vector<double> want;
  want.reserve(got.size());
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int qi = 0; qi < qL; ++qi) {
        for (int d = 0; d < D; ++d) {
          double acc = 0.0;
          for (int ki = 0; ki < kL; ++ki) {
            int kv = h / (H / KV);
            acc += v_data[((b * KV + kv) * kL + ki) * D + d];
          }
          want.push_back(acc / kL);
        }
      }
    }
  }
  require_close(got, want, 1e-5, "sdpa fully masked broadcast gqa mean");
}

TEST_CASE("fp8 conversion matches the upstream bit algorithm") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Known E4M3 patterns: zero, denorm floor 2^-9, one, min normal 2^-6,
  // max 448, signs, non-trivial mantissas.
  std::vector<double> probe{
      0.0,
      1.0,
      -1.0,
      448.0,
      -448.0,
      0.5,
      2.0,
      3.5,
      0.001953125,
      7.0,
      0.015625,
      264.0};
  std::vector<float> probe_f(probe.begin(), probe.end());
  array x = array(probe_f.begin(), Shape{int(probe.size())}, float32);

  array encoded = to_fp8(x, stream);
  encoded.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const uint8_t* bytes = encoded.data<uint8_t>();
  for (size_t index = 0; index < probe.size(); ++index) {
    CHECK_EQ(int(bytes[index]), int(host_to_fp8(probe[index])));
  }
  // Pinpoints: one, the saturation ceiling, its negation, denorm floor.
  CHECK_EQ(int(host_to_fp8(1.0)), 0x38);
  CHECK_EQ(int(host_to_fp8(448.0)), 0x7E);
  CHECK_EQ(int(host_to_fp8(-448.0)), 0xFE);
  CHECK_EQ(int(host_to_fp8(0.001953125)), 0x01);

  array decoded = from_fp8(encoded, float32, stream);
  auto round_trip = flat(decoded, stream);
  for (size_t index = 0; index < probe.size(); ++index) {
    double quantum = std::max(1.0, std::abs(probe[index])) * 0.0625;
    CHECK(std::abs(double(round_trip[index]) - probe[index]) <= quantum + 1e-6);
  }

  // f16 and bf16 storages carry the same E4M3 payload.
  array x16 = astype(x, float16, stream);
  array encoded16 = to_fp8(x16, stream);
  encoded16.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const uint8_t* bytes16 = encoded16.data<uint8_t>();
  for (size_t index = 0; index < probe.size(); ++index) {
    CHECK_EQ(int(bytes16[index]), int(host_to_fp8(probe[index])));
  }
  array xbf = astype(x, bfloat16, stream);
  array encodedbf = to_fp8(xbf, stream);
  encodedbf.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const uint8_t* bytesbf = encodedbf.data<uint8_t>();
  for (size_t index = 0; index < probe.size(); ++index) {
    CHECK_EQ(int(bytesbf[index]), int(host_to_fp8(probe[index])));
  }
  // Narrow decodes must reproduce the f32 decode payload exactly: every
  // E4M3 value is exact in f16 and bf16.
  auto decoded16 = flat(from_fp8(encoded16, float16, stream), stream);
  require_close(decoded16, widen(round_trip), 1e-6, "fp8 decode f16");
  auto decodedbf = flat(from_fp8(encodedbf, bfloat16, stream), stream);
  require_close(decodedbf, widen(round_trip), 1e-6, "fp8 decode bf16");
}

TEST_CASE("norm kernels normalize non-contiguous inputs exactly") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array base = arange(0, 16, float32, stream);
  array transposed =
      transpose(reshape(base, Shape{4, 4}, stream), {1, 0}, stream);
  array weight = ones({4}, float32, stream);

  // Reference over the logical transposed values t[r][c] = 4*c + r.
  double x[4][4];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      x[r][c] = 4.0 * c + r;
    }
  }
  std::vector<double> rms_expected;
  std::vector<double> layer_expected;
  rms_expected.reserve(16);
  layer_expected.reserve(16);
  for (int r = 0; r < 4; ++r) {
    double sum = 0.0;
    double ms = 0.0;
    for (int c = 0; c < 4; ++c) {
      sum += x[r][c];
      ms += x[r][c] * x[r][c];
    }
    double mean = sum / 4.0;
    ms /= 4.0;
    double rms_inv = 1.0 / std::sqrt(ms + 1e-5);
    double layer_inv = 1.0 / std::sqrt((ms - mean * mean) + 1e-5);
    for (int c = 0; c < 4; ++c) {
      rms_expected.push_back(x[r][c] * rms_inv);
      layer_expected.push_back((x[r][c] - mean) * layer_inv + 1.0);
    }
  }
  require_close(
      flat(fast::rms_norm(transposed, weight, 1e-5f, stream), stream),
      rms_expected,
      1e-5,
      "rms_norm transposed");
  require_close(
      flat(fast::layer_norm(transposed, weight, weight, 1e-5f, stream), stream),
      layer_expected,
      1e-5,
      "layer_norm transposed");

  // A weight whose last axis does not match the row length is refused by
  // the upstream op validation before the backend gate can run.
  auto weight_message = caught_message([&] {
    fast::rms_norm(
        reshape(base, Shape{4, 4}, stream),
        ones({3}, float32, stream),
        1e-5f,
        stream)
        .eval();
  });
  CHECK(weight_message.find("[rms_norm]") != std::string::npos);
  CHECK(weight_message.find("same size") != std::string::npos);

  // The backend keeps its own parameter gate for direct primitive
  // construction, which bypasses upstream validation.
  array mismatched = array(
      Shape{4, 4},
      float32,
      std::make_shared<fast::RMSNorm>(
          stream,
          [](const std::vector<array>& inputs) {
            return std::vector<array>{inputs[0]};
          },
          1e-5f),
      {reshape(base, Shape{4, 4}, stream), ones({3}, float32, stream)});
  auto gate_message = caught_message([&] { mismatched.eval(); });
  CHECK(gate_message.find("parameter shape") != std::string::npos);
  CHECK(gate_message.find("[omarchy]") != std::string::npos);
}


// ---- fused RoPE ----

// The composed reference: the mlx/fast.cpp rope() fallback algebra rebuilt
// from core ops, the graph the eager fallback dispatched before this
// primitive went native. float16 and bfloat16 must match it bit for bit;
// float32 rides the contraction tolerance documented at f32_tolerance.
array composed_rope(
    const array& x_in,
    int dims,
    bool traditional,
    float base,
    float scale,
    const array& offset,
    const std::optional<array>& freqs,
    bool forward,
    Stream s) {
  array x = x_in;
  auto shape = x.shape();
  if (x.ndim() == 3) {
    x = expand_dims(x, 1, s);
  } else if (x.ndim() > 4) {
    x = flatten(x, 1, 1 + (x.ndim() - 4), s);
  }
  auto B = x.shape(0);
  auto N = x.shape(1);
  auto T = x.shape(2);
  auto t = x.dtype();
  auto half_dims = dims / 2;
  auto off = offset;
  if (off.size() > 1) {
    off = expand_dims(off, std::vector<int>{-1, -2}, s);
  }
  auto positions = multiply(
      add(arange(x.shape(2), float32, s), off, s), array(scale, float32), s);
  auto inv_freqs =
      freqs ? reciprocal(*freqs, s)
            : exp(
                  multiply(
                      arange(0, -half_dims, -1, float32, s),
                      array(std::log(base) / half_dims, float32),
                      s),
                  s);
  auto theta = multiply(expand_dims(positions, -1, s), inv_freqs, s);
  auto coss = astype(cos(theta, s), t, s);
  auto sins = astype(sin(theta, s), t, s);
  auto apply_rope = [&](const array& x1, const array& x2) {
    std::vector<array> outs;
    if (forward) {
      outs.push_back(
          subtract(multiply(x1, coss, s), multiply(x2, sins, s), s));
      outs.push_back(add(multiply(x1, sins, s), multiply(x2, coss, s), s));
    } else {
      outs.push_back(add(multiply(x2, sins, s), multiply(x1, coss, s), s));
      outs.push_back(
          subtract(multiply(x2, coss, s), multiply(x1, sins, s), s));
    }
    return outs;
  };
  if (traditional) {
    auto x1 = slice(x, {0, 0, 0, 0}, {B, N, T, dims}, {1, 1, 1, 2}, s);
    auto x2 = slice(x, {0, 0, 0, 1}, {B, N, T, dims}, {1, 1, 1, 2}, s);
    auto outs = apply_rope(x1, x2);
    for (auto& o : outs) {
      o = expand_dims(o, -1, s);
    }
    auto out = reshape(concatenate(outs, -1, s), {B, N, T, dims}, s);
    if (dims < x.shape(-1)) {
      out =
          concatenate({out, slice(x, {0, 0, 0, dims}, x.shape(), s)}, -1, s);
    }
    return reshape(out, shape, s);
  } else {
    auto out_s = x.shape();
    out_s.back() = half_dims;
    auto x1 = slice(x, {0, 0, 0, 0}, out_s, s);
    out_s.back() = dims;
    auto x2 = slice(x, {0, 0, 0, half_dims}, out_s, s);
    auto outs = apply_rope(x1, x2);
    if (dims < x.shape(-1)) {
      outs.push_back(slice(x, {0, 0, 0, dims}, x.shape(), s));
    }
    return reshape(concatenate(outs, -1, s), shape, s);
  }
}

// The f32 image of a float16/bfloat16 value is exact and injective, so
// comparing f32 images is bit comparison of the stored values.
void require_bit_equal(
    const array& got,
    const array& want,
    Stream stream,
    const std::string& what) {
  REQUIRE_EQ(got.shape(), want.shape());
  REQUIRE_EQ(got.dtype(), want.dtype());
  auto got_v = flat(got, stream);
  auto want_v = flat(want, stream);
  for (size_t index = 0; index < want_v.size(); ++index) {
    CHECK_MESSAGE(
        got_v[index] == want_v[index],
        what,
        " element ",
        index,
        ": got ",
        got_v[index],
        " want ",
        want_v[index]);
  }
}

// float32 tolerance for the fused kernel against the composed fallback:
// the driver's compiler may contract x*y - z*w into a fused multiply-add,
// and the composed path can never do that because its intermediates cross
// kernel boundaries. Cancellation amplifies the round-off, so the check is
// absolute, not in ulps: observed spread stays at or below ~1e-8, pinned
// at 1e-6. float16 and bfloat16 keep the bit-exact contract because their
// storage-precision roundings break the expression tree before the add.
// True on the Apple GPU backend (Honeykrisp). There the fused and the
// composed paths compile their builtins in separate shaders, and the
// backend does not promise the two compilations agree in the last ulp,
// so bit-exactness against the composed path is not an enforceable
// contract. The float64 reference assertions carry the accuracy
// contract on this device; the structural bit-exact contract runs on
// the development box, where llvmpipe's codegen is deterministic and
// both paths agree bit for bit.
static bool rope_on_apple_gpu() {
  static const bool apple = [] {
    if (!gpu::is_available()) {
      return false;
    }
    for (const auto& [key, value] : gpu::device_info()) {
      if (key != "device_name") {
        continue;
      }
      const auto* name = std::get_if<std::string>(&value);
      return name != nullptr &&
          name->find("Apple") != std::string::npos;
    }
    return false;
  }();
  return apple;
}
constexpr double f32_tolerance = 1e-6;

// Honeykrisp (Apple M-series) does not guarantee bit-identical trig
// codegen across separately compiled shaders. The fused kernel and the
// composed kernels compute the same theta from the same GLSL builtins,
// but the AGX backend compiles each shader independently and does not
// promise the two compilations agree in the last bit - and the
// driver's sin/cos range reduction has a documented accuracy envelope
// (known-defects: 4.8e-3 absolute error at theta 123457). Observed on
// the M1 (jwm1 rope-fastops.log, main 6a57c84, 2026-09-04): f32 deltas
// <= 6.2e-4 and f16 deltas <= 2 ulp, every failing element inside
// that envelope; no garbage-corruption class values. llvmpipe codegen
// is deterministic and bit-exact, so the strict contract holds where
// it is provable and the documented envelope governs where it is not.
void require_rope_close(
    const array& got,
    const array& want,
    Stream stream,
    const std::string& what,
    double theta_bound = 1e3) {
  REQUIRE_EQ(got.shape(), want.shape());
  bool apple = rope_on_apple_gpu();
  if (got.dtype() == float32) {
    // The Apple tier steps through the MEASURED sin/cos error curve
    // from the v0.3.1 scalar sweep (receipts/2026-09-02-m1-red-suites-
    // root-cause.md, first committed 959c7a0, one day before this
    // kernel existed): 4.2e-6 at 100, 2.8e-5 at 1e3, 3.6e-4 at 5e3,
    // 4.5e-4 at 12345, 1.2e-3 at 2e4, 4.8e-3 at 123457. Each band is
    // bounded by the sweep's measurement at its upper edge - the worst
    // measured error inside the band, since the error grows with
    // theta - so a hundred-fold accuracy regression near theta 1e3
    // cannot hide behind the 1e5-band worst point. theta_bound is the
    // analytic maximum |theta| of the configuration, the same bound
    // rope_trig_gate computes. Above 1e5 the gate refuses outright.
    // Dev box (llvmpipe): codegen is deterministic per Mesa version,
    // so its divergence envelope is a property, not a sample - the
    // bounds below are the measured spread of the offset sweep (fma
    // contraction plus range-reduction drift, both growing with
    // theta); re-measure on Mesa bumps. Deterministic-envelope bounds
    // are regression gates, unlike the Apple tier which must absorb
    // nondeterministic trig codegen.
    double tolerance = f32_tolerance;
    if (apple) {
      if (theta_bound <= 100.0) {
        tolerance = 1e-5;
      } else if (theta_bound <= 1e3) {
        tolerance = 2.8e-5;
      } else if (theta_bound <= 5e3) {
        tolerance = 3.6e-4;
      } else if (theta_bound <= 12345.0) {
        tolerance = 4.5e-4;
      } else if (theta_bound <= 2e4) {
        tolerance = 1.2e-3;
      } else {
        tolerance = 4.8e-3;
      }
    } else if (theta_bound > 100.0) {
      if (theta_bound <= 1e3) {
        tolerance = 1e-5;
      } else if (theta_bound <= 5e3) {
        tolerance = 5e-5;
      } else if (theta_bound <= 12345.0) {
        tolerance = 2e-4;
      } else if (theta_bound <= 2e4) {
        tolerance = 6e-4;
      } else {
        tolerance = 2e-3;
      }
    }
    require_close(flat(got, stream), widen(flat(want, stream)),
                  tolerance, what);
  } else if (got.dtype() == bfloat16) {
    // bfloat16 rides a proven CastF32BF16 dispatch on this driver,
    // which lands within one ulp of the composed per-op rounding. An
    // absolute 1e-2 bound keeps well under the bf16 grid spacing for
    // the test inputs.
    require_close(flat(got, stream), widen(flat(want, stream)),
                  1e-2, what);
  } else if (got.dtype() == float16) {
    // DERIVED two-approximations bound (Main/RopeAccuracyVerdict,
    // 2026-09-04), not an observed spread: each path's theta differs
    // from the true angle by up to theta * 2^-23 (one codegen ulp in
    // the angle construction), the trig slope is at most 1, and each
    // path then rounds to the f16 grid once (0.5 ulp each, 1 ulp
    // combined = 1.5 grid ulps at magnitude 2 = 1.465e-3, the
    // verifier's constant). 2B therefore covers any two orderings of
    // the same arithmetic; it is NOT fitted to observed deltas. The
    // four M1 reds at theta 28460.5 (3.4-3.8e-3) sit under 2B =
    // 9.7e-3 with 2.56x margin. Mechanism note for the record: these
    // elements say NOTHING about the kernel - they are the same
    // theta * 2^-23 divergence as the f32 band failures (28460 *
    // 2^-23 = 3.39e-3, exactly the observed magnitude), rendered as
    // multi-lane f16 jumps because the f16 grid at |r| <= 2 is
    // 9.77e-4. Same cause, coarser grid.
    constexpr double kTwoPowMinus23 = 1.1920929e-7;
    constexpr double kF16GridUlpsAtTwo = 1.465e-3;
    double derived_2b =
        2.0 * (theta_bound * kTwoPowMinus23 + kF16GridUlpsAtTwo);
    bool bit_exact_ok = !apple && theta_bound <= 12345.0;
    if (bit_exact_ok) {
      // Dev box, low theta: codegen is deterministic and the measured
      // spread across the offset sweep is zero - keep the strict
      // contract where it provably holds. This comparator is
      // superseded on Apple by RopeAccuracyVerdict's per-element
      // vs-truth scoring (a path above its OWN B is the real defect
      // gate); until those assertions land, 2B governs here.
      require_bit_equal(got, want, stream, what);
    } else {
      require_close(flat(got, stream), widen(flat(want, stream)),
                    derived_2b, what);
    }
  } else {
    require_bit_equal(got, want, stream, what);
  }
}


// ---- host float64 RoPE reference ----
//
// Ground truth for the fused/composed RoPE value tests: the RoPE algebra
// evaluated in double from the exact stored inputs. The composed fallback
// is NOT the reference - it is a second float32 approximation running on
// the same driver. Its exp/reciprocal/sin/cos compile inside
// elementwise.comp while the fused kernel's compile inside fast_rope.comp,
// and the Apple GPU backend does not promise that two separately compiled
// shaders agree in the last ulp of a transcendental. One inv_freq ulp at
// magnitude 3162 is 2.4e-4; times a position of 9 that is 2.2e-3 of
// absolute theta - the same order as float32's own quantization of a
// 28460 radian theta. On llvmpipe the two compilations agree bit for
// bit, which is why the bit-exact contract still holds on the
// development box and nowhere else.
std::vector<double> host_rope_reference(
    const std::vector<float>& x_bits,
    const Shape& shape,
    int dims,
    bool traditional,
    const std::vector<float>& freqs_bits,
    float base,
    double scale,
    int offset_value) {
  const int T = shape[shape.size() - 2];
  const int D = shape[shape.size() - 1];
  const int half = dims / 2;
  std::vector<double> inv_freq(half);
  if (!freqs_bits.empty()) {
    // The freqs array is caller data: its stored float32 bits are exact
    // inputs, so the reference reciprocates them in double.
    for (int i = 0; i < half; ++i) {
      inv_freq[i] = 1.0 / static_cast<double>(freqs_bits[i]);
    }
  } else {
    const double beta = std::log(static_cast<double>(base)) / half;
    for (int i = 0; i < half; ++i) {
      inv_freq[i] = std::exp(-static_cast<double>(i) * beta);
    }
  }
  size_t count = 1;
  for (auto dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  std::vector<double> out(count, 0.0);
  const size_t rows = count / static_cast<size_t>(D);
  for (size_t row = 0; row < rows; ++row) {
    const int t = static_cast<int>(row % static_cast<size_t>(T));
    const double position =
        (static_cast<double>(t) + static_cast<double>(offset_value)) * scale;
    for (int d = 0; d < dims; ++d) {
      const bool first_lane = traditional ? ((d & 1) == 0) : (d < half);
      const int i = traditional ? (d >> 1) : (first_lane ? d : d - half);
      const int x1_d = traditional ? (d & ~1) : i;
      const int x2_d = traditional ? (x1_d + 1) : (i + half);
      const double x1 = x_bits[row * D + x1_d];
      const double x2 = x_bits[row * D + x2_d];
      const double theta = position * inv_freq[i];
      const double c = std::cos(theta);
      const double s = std::sin(theta);
      out[row * D + d] =
          first_lane ? (x1 * c - x2 * s) : (x1 * s + x2 * c);
    }
    // The passthrough tail is a verbatim copy of the input in both
    // implementations, so the reference carries it through too.
    for (int d = dims; d < D; ++d) {
      out[row * D + d] = x_bits[row * D + d];
    }
  }
  return out;
}

// Tolerance of a path against the float64 reference, derived from the
// same bound rope_trig_gate uses for |theta|: two half-ulp roundings
// (inv_freq, then theta) give a trig-argument error of |theta| * 2^-23,
// sin and cos are 1-Lipschitz, and the elementwise products, sums, and
// any driver contraction add at most three roundings of operands <= 2.
// This bound dominates the documented Honeykrisp builtin envelope
// (5e-3 absolute across 1e3..1e5) at every theta the gate allows.
constexpr double kRefRounding = 1.1920929e-7; // 2^-23

double rope_reference_tolerance_f32(double theta_bound) {
  return theta_bound * kRefRounding + 1e-6;
}

// float16 stores the output on a ~1e-3 grid at magnitude 1-2; a sub-ulp
// theta shift (the codegen variance above) can move an element one
// extra grid lane, so the bound is 1.5 ulps of 2 plus the argument term.
double rope_reference_tolerance_f16(double theta_bound) {
  return theta_bound * kRefRounding + 9.765625e-4 * 1.5;
}

// bfloat16 stores on a ~7.8e-3 grid at magnitude 2-4 (ulp 2^-7): the
// same theta * 2^-23 argument-construction term as the f16 bound, with
// the coarser storage grid.
double rope_reference_tolerance_bf16(double theta_bound) {
  return theta_bound * kRefRounding + 7.8125e-3 * 1.5;
}

void require_matches_reference(
    const array& got,
    const std::vector<double>& reference,
    double tolerance,
    Stream stream,
    const std::string& what) {
  auto got_v = flat(got, stream);
  REQUIRE_EQ(got_v.size(), reference.size());
  for (size_t index = 0; index < reference.size(); ++index) {
    double diff = std::abs(
        static_cast<double>(got_v[index]) - reference[index]);
    CHECK_MESSAGE(
        diff <= tolerance,
        what,
        " element ",
        index,
        ": got ",
        got_v[index],
        " reference ",
        reference[index]);
  }
}


// Worst-case |theta| for a config: the same product rope_trig_gate
// bounds, (max offset + T - 1) * |scale| * max|inv_freq|. Exact for the
// non-negative offsets every caller sends.
double rope_theta_bound(
    int offset_value,
    int T,
    float scale,
    bool with_freqs,
    const std::vector<float>& freqs_bits,
    float base,
    int half) {
  double inv_freq_max;
  if (with_freqs) {
    float min_abs = std::abs(freqs_bits[0]);
    for (auto f : freqs_bits) {
      min_abs = std::min(min_abs, std::abs(f));
    }
    inv_freq_max = 1.0 / static_cast<double>(min_abs);
  } else {
    inv_freq_max = std::exp(
        -static_cast<double>(half - 1) *
        (std::log(static_cast<double>(base)) / half));
  }
  return (static_cast<double>(offset_value) + T - 1) *
      std::abs(static_cast<double>(scale)) * inv_freq_max;
}

// The passthrough tail (last-axis index >= dims) is a verbatim copy in
// both paths - no arithmetic runs on it - so it must be bit-identical
// to the input on every device. No accuracy tier covers it: a single
// differing tail element is a real defect, not tolerance noise.
void require_passthrough_exact(
    const array& got,
    const array& x,
    int dims,
    Stream stream,
    const std::string& what) {
  auto got_v = flat(got, stream);
  auto x_v = flat(x, stream);
  REQUIRE_EQ(got_v.size(), x_v.size());
  const int D = x.shape().back();
  for (size_t index = 0; index < got_v.size(); ++index) {
    if (static_cast<int>(index % static_cast<size_t>(D)) >= dims) {
      CHECK_MESSAGE(
          got_v[index] == x_v[index],
          what,
          " passthrough element ",
          index,
          ": got ",
          got_v[index],
          " input ",
          x_v[index]);
    }
  }
}

array rope_input(const Shape& shape, uint32_t seed) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  auto values = pattern(count, seed);
  return array(values.begin(), shape, float32);
}

TEST_CASE("fused rope matches the composed fallback bit for bit") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array offset = array(3, int32);
  array freqs = exp(
      multiply(
          arange(0, -8, -1, float32, stream),
          array(std::log(10000.0f) / 8, float32),
          stream),
      stream);
  auto freqs_bits = flat(freqs, stream);
  // The freqs variant reciprocates its smallest entry, so theta runs to
  // (3 + 7 - 1) * 10000^(7/8) ~ 28460 rad: inside the gate's trusted
  // envelope but far past the band where the two shader compilations
  // stop agreeing in the last ulp. The reference bound scales with it.
  const double theta_bound =
      rope_theta_bound(3, 7, 1.0f, true, freqs_bits, 10000.0f, 8);

  for (auto dtype : {float32, float16, bfloat16}) {
    for (bool traditional : {false, true}) {
      for (bool with_freqs : {false, true}) {
        Shape shape{2, 3, 7, 16};
        array x = astype(rope_input(shape, 101), dtype, stream);
        std::string what = std::string("rope variant ") +
            std::to_string((traditional ? 1 : 0) + (with_freqs ? 2 : 0)) +
            (dtype == float32 ? " f32" : (dtype == float16 ? " f16" : " bf16"));
        auto got = fast::rope(
            x,
            16,
            traditional,
            with_freqs ? std::nullopt : std::optional<float>(10000.0f),
            1.0f,
            offset,
            with_freqs ? std::optional<array>(freqs) : std::nullopt,
            stream);
        auto want = composed_rope(
            x,
            16,
            traditional,
            10000.0f,
            1.0f,
            offset,
            with_freqs ? std::optional<array>(freqs) : std::nullopt,
            true,
            stream);
        // Analytic theta bound: positions run [3, 3+T-1]; the largest
        // inv-frequency is the reciprocal of the smallest freqs entry
        // (with_freqs) or exactly 1 (base >= 1).
        double theta_bound = 9.0;
        if (with_freqs) {
          double min_freq = 1.0 / std::exp(7.0 * (std::log(10000.0) / 8));
          theta_bound = 9.0 / min_freq;
        }
        // Structural leg: on the development box both paths agree bit
        // for bit (f32 within the contraction bound). On Apple the
        // float64 reference assertions below carry the contract.
        if (!rope_on_apple_gpu()) {
          require_rope_close(got, want, stream, what, theta_bound);
        }
        // Accuracy leg, every device: both paths against the float64
        // reference. bf16 keeps its existing composed-path contract
        // (the proven CastF32BF16 dispatch) instead of this bound.
        if (dtype != bfloat16) {
          double tolerance = (dtype == float32)
              ? rope_reference_tolerance_f32(theta_bound)
              : rope_reference_tolerance_f16(theta_bound);
          auto reference = host_rope_reference(
              flat(x, stream),
              shape,
              16,
              traditional,
              with_freqs ? freqs_bits : std::vector<float>{},
              10000.0f,
              1.0,
              3);
          require_matches_reference(
              got, reference, tolerance, stream, what + " fused vs f64");
          require_matches_reference(
              want, reference, tolerance, stream, what + " composed vs f64");
        }
      }
    }
  }
}

TEST_CASE("fused rope decode shapes match the composed fallback") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Qwen2.5-0.5B decode: T == 1, one growing scalar offset, 14 query
  // heads at dims 128 and 2 KV heads at dims 64, f16 storage.
  array offset = array(17, int32);
  Shape q_shape{1, 14, 1, 128};
  array q = astype(rope_input(q_shape, 103), float16, stream);
  auto got_q =
      fast::rope(q, 128, false, 500000.0f, 1.0f, offset, std::nullopt, stream);
  auto want_q = composed_rope(
      q, 128, false, 500000.0f, 1.0f, offset, std::nullopt, true, stream);
  // Structural leg (dev box): both paths agree bit for bit. On Apple
  // the float64 reference assertions carry the contract instead: theta
  // stays <= 17 here, so the bound is the f16 output grid plus a small
  // argument term.
  const double theta_bound =
      rope_theta_bound(17, 1, 1.0f, false, {}, 500000.0f, 64);
  const double tolerance = rope_reference_tolerance_f16(theta_bound);
  if (!rope_on_apple_gpu()) {
    require_bit_equal(got_q, want_q, stream, "rope decode q f16");
  }
  {
    auto reference = host_rope_reference(
        flat(q, stream), q_shape, 128, false, {}, 500000.0f, 1.0, 17);
    require_matches_reference(
        got_q, reference, tolerance, stream, "rope decode q fused vs f64");
    require_matches_reference(
        want_q, reference, tolerance, stream, "rope decode q composed vs f64");
  }

  Shape k_shape{1, 2, 1, 64};
  array k = astype(rope_input(k_shape, 107), float16, stream);
  auto got_k =
      fast::rope(k, 64, false, 500000.0f, 1.0f, offset, std::nullopt, stream);
  auto want_k = composed_rope(
      k, 64, false, 500000.0f, 1.0f, offset, std::nullopt, true, stream);
  if (!rope_on_apple_gpu()) {
    require_bit_equal(got_k, want_k, stream, "rope decode k f16");
  }
  {
    auto reference = host_rope_reference(
        flat(k, stream), k_shape, 64, false, {}, 500000.0f, 1.0, 17);
    require_matches_reference(
        got_k, reference, tolerance, stream, "rope decode k fused vs f64");
    require_matches_reference(
        want_k, reference, tolerance, stream, "rope decode k composed vs f64");
  }
}

TEST_CASE("fused rope partial dims keep the passthrough exact") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array offset = array(2, int32);
  Shape shape{2, 3, 5, 32};
  // theta <= (2 + 5 - 1) * 1.0 = 6: no range-reduction band effects.
  const double theta_bound =
      rope_theta_bound(2, 5, 1.0f, false, {}, 10000.0f, 8);
  for (auto dtype : {float32, float16}) {
    array x = astype(rope_input(shape, 109), dtype, stream);
    auto got =
        fast::rope(x, 16, true, 10000.0f, 1.0f, offset, std::nullopt, stream);
    auto want = composed_rope(
        x, 16, true, 10000.0f, 1.0f, offset, std::nullopt, true, stream);
    // PASSTHROUGH, every device, absolute: tail elements (last-axis
    // index >= 16) are a verbatim copy in both paths - no arithmetic
    // runs on them - so they must be bit-identical to the input. A
    // single differing tail element is a real defect, not tolerance
    // noise, and no accuracy tier covers it.
    require_passthrough_exact(got, x, 16, stream, "rope partial dims");
    require_passthrough_exact(want, x, 16, stream, "rope partial dims");
    // Structural leg on the dev box; reference assertions everywhere.
    if (!rope_on_apple_gpu()) {
      require_rope_close(got, want, stream, "rope partial dims");
    }
    double tolerance = (dtype == float32)
        ? rope_reference_tolerance_f32(theta_bound)
        : rope_reference_tolerance_f16(theta_bound);
    auto reference = host_rope_reference(
        flat(x, stream), shape, 16, true, {}, 10000.0f, 1.0, 2);
    require_matches_reference(
        got, reference, tolerance, stream, "rope partial dims fused vs f64");
    require_matches_reference(
        want, reference, tolerance, stream, "rope partial dims composed vs f64");
  }
}

TEST_CASE("fused rope boundary shapes match the composed fallback") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array offset = array(0, int32);
  // dims == 2: one frequency per row.
  Shape tiny{2, 1, 4, 2};
  array x_tiny = rope_input(tiny, 113);
  require_rope_close(
      fast::rope(
          x_tiny, 2, false, 10000.0f, 1.0f, offset, std::nullopt, stream),
      composed_rope(
          x_tiny, 2, false, 10000.0f, 1.0f, offset, std::nullopt, true,
          stream),
      stream,
      "rope dims 2");
  // dims == D boundary on a 4D shape with scale and a non-default base.
  Shape full{2, 2, 6, 8};
  array x_full = rope_input(full, 127);
  require_rope_close(
      fast::rope(x_full, 8, true, 100.0f, 2.5f, offset, std::nullopt, stream),
      composed_rope(
          x_full, 8, true, 100.0f, 2.5f, offset, std::nullopt, true, stream),
      stream,
      "rope scale 2.5 base 100");
  // 5D input: the middle dims fold into N.
  Shape five{2, 3, 2, 5, 8};
  auto five_values = pattern(2 * 3 * 2 * 5 * 8, 131);
  array x_five = array(five_values.begin(), five, float32);
  require_rope_close(
      fast::rope(
          x_five, 8, false, 10000.0f, 1.0f, offset, std::nullopt, stream),
      composed_rope(
          x_five, 8, false, 10000.0f, 1.0f, offset, std::nullopt, true,
          stream),
      stream,
      "rope 5D");
}

TEST_CASE("fused rope bf16 direct bind matches wrapped and composed paths") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array offset = array(5, int32);
  array freqs = exp(
      multiply(
          arange(0, -8, -1, float32, stream),
          array(std::log(10000.0f) / 8, float32),
          stream),
      stream);
  auto freqs_bits = flat(freqs, stream);

  // The bf16 path binds the bf16 input straight into the packed-word
  // load leg and stores packed bf16 words (no f32 interior).
  for (bool traditional : {false, true}) {
    for (bool with_freqs : {false, true}) {
      Shape shape{2, 3, 5, 16};
      array x = astype(rope_input(shape, 211), bfloat16, stream);
      std::string what = std::string("rope bf16 direct variant ") +
          std::to_string((traditional ? 1 : 0) + (with_freqs ? 2 : 0));
      auto direct = fast::rope(
          x,
          16,
          traditional,
          with_freqs ? std::nullopt : std::optional<float>(10000.0f),
          1.0f,
          offset,
          with_freqs ? std::optional<array>(freqs) : std::nullopt,
          stream);
      auto direct_v = flat(direct, stream);
      auto composed = composed_rope(
          x,
          16,
          traditional,
          10000.0f,
          1.0f,
          offset,
          with_freqs ? std::optional<array>(freqs) : std::nullopt,
          true,
          stream);
      auto composed_v = flat(composed, stream);

      // Theta bound for the cross-path and f64 legs (the same product
      // rope_trig_gate bounds).
      const double theta_bound =
          rope_theta_bound(5, 5, 1.0f, with_freqs, freqs_bits, 10000.0f, 8);
      if (!rope_on_apple_gpu()) {
        // The packed-word load is the exact f32 image of the stored
        // bf16 input, and every intermediate passes rope_round exactly
        // like the eager composition's per-op bf16 stores; llvmpipe
        // codegen is deterministic, so the direct leg meets the
        // composed bf16 chain bit for bit - the contract the wrapped
        // path only held within one ulp (require_rope_close's 1e-2
        // bf16 tier).
        require_bit_equal(
            composed, direct, stream, what + " direct vs composed");
      } else {
        // Apple compiles fused and composed trig in separate shaders
        // with no last-ulp agreement (require_rope_close's f32/f16
        // notes). The cross-path contract is the same derived 2B form
        // the f16 case uses, with the bf16 grid: two theta * 2^-23
        // angle constructions and two storage roundings.
        const double cross_path =
            2.0 * (theta_bound * kRefRounding + 7.8125e-3);
        require_close(
            flat(composed, stream),
            widen(flat(direct, stream)),
            cross_path,
            what + " direct vs composed");
      }

      // Direct vs f64 reference.
      auto reference = host_rope_reference(
          flat(x, stream),
          shape,
          16,
          traditional,
          with_freqs ? freqs_bits : std::vector<float>{},
          10000.0f,
          1.0,
          5);
      require_matches_reference(
          direct,
          reference,
          rope_reference_tolerance_bf16(theta_bound),
          stream,
          what + " direct vs f64");
    }
  }
}

TEST_CASE("fused rope strided and transposed inputs match the composed fallback") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array offset = array(1, int32);
  // 3D strided: every other row of a wider buffer (dispatch_ndim == 3).
  array wide3 = rope_input(Shape{3, 10, 16}, 137);
  array strided3 = slice(wide3, {0, 0, 0}, {3, 10, 16}, {1, 2, 1}, stream);
  // MLX packs stepped views with the natural stride here (the
  // dispatched path is decided by non-trivial middle-dim strides; the
  // fused branch handles whatever strides arrive).
  REQUIRE(strided3.shape() == Shape{3, 5, 16});
  REQUIRE(strided3.strides()[1] != strided3.strides()[0]);
  require_rope_close(
      fast::rope(
          strided3, 16, true, 10000.0f, 1.0f, offset, std::nullopt, stream),
      composed_rope(
          strided3, 16, true, 10000.0f, 1.0f, offset, std::nullopt, true,
          stream),
      stream,
      "rope 3D strided");
  // 4D head/sequence transposed cache layout.
  array bnt = rope_input(Shape{2, 5, 3, 8}, 139);
  array btn = transpose(bnt, {0, 2, 1, 3}, stream);
  // Just verify the kernel handles the transposed view; the exact
  // stride layout varies between MLX versions and is not the property
  // the fused kernel relies on (the kernel reads the strides it
  // receives through the push-constant mapping).
  REQUIRE(btn.shape() == Shape{2, 3, 5, 8});
  require_rope_close(
      fast::rope(btn, 8, false, 10000.0f, 1.0f, offset, std::nullopt, stream),
      composed_rope(
          btn, 8, false, 10000.0f, 1.0f, offset, std::nullopt, true, stream),
      stream,
      "rope head-seq transpose");
  // 5D strided: the general-copy path through the temporary.
  array wide5 = rope_input(Shape{2, 6, 2, 7, 8}, 149);
  array strided5 = slice(
      wide5, {0, 0, 0, 0, 0}, {2, 6, 2, 7, 8}, {1, 3, 1, 1, 1}, stream);
  require_rope_close(
      fast::rope(
          strided5, 8, false, 10000.0f, 1.0f, offset, std::nullopt, stream),
      composed_rope(
          strided5, 8, false, 10000.0f, 1.0f, offset, std::nullopt, true,
          stream),
      stream,
      "rope 5D strided");
}

TEST_CASE("fused rope bf16 noncontiguous layouts match the f64 reference") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  array offset = array(5, int32);
  // The wrapped bf16 path up-casts the input into a dense f32 image and
  // the stride classification must describe that image. The pre-fix
  // code classified the ORIGINAL view instead, so sliced and transposed
  // bf16 inputs read the dense image with stale strides and diverged
  // from the whole-contiguous case. Contract: the same derived
  // bf16-vs-f64 bound the contiguous case holds, on every device.

  // 3D strided slice: strides classified from a sliced view.
  array wide3 = astype(rope_input(Shape{3, 10, 16}, 257), bfloat16, stream);
  array strided3 = slice(wide3, {0, 0, 0}, {3, 10, 16}, {1, 2, 1}, stream);
  REQUIRE(strided3.shape() == Shape{3, 5, 16});
  const double theta3 =
      rope_theta_bound(5, 5, 1.0f, false, {}, 10000.0f, 8);
  auto got3 = fast::rope(
      strided3, 16, false, 10000.0f, 1.0f, offset, std::nullopt, stream);
  require_matches_reference(
      got3,
      host_rope_reference(
          flat(contiguous(strided3, false, stream), stream),
          strided3.shape(),
          16,
          false,
          {},
          10000.0f,
          1.0,
          5),
      rope_reference_tolerance_bf16(theta3),
      stream,
      "rope bf16 3D strided vs f64");

  // 4D head/sequence transposed cache layout: the pre-fix path kept the
  // transpose flag while binding the promoted dense image.
  array bnt = astype(rope_input(Shape{2, 5, 3, 8}, 263), bfloat16, stream);
  array btn = transpose(bnt, {0, 2, 1, 3}, stream);
  REQUIRE(btn.shape() == Shape{2, 3, 5, 8});
  const double theta4 =
      rope_theta_bound(5, 5, 1.0f, false, {}, 10000.0f, 4);
  auto got4 =
      fast::rope(btn, 8, false, 10000.0f, 1.0f, offset, std::nullopt, stream);
  require_matches_reference(
      got4,
      host_rope_reference(
          flat(contiguous(btn, false, stream), stream),
          btn.shape(),
          8,
          false,
          {},
          10000.0f,
          1.0,
          5),
      rope_reference_tolerance_bf16(theta4),
      stream,
      "rope bf16 head-seq transpose vs f64");

  // 5D strided: the promotion itself must survive a general layout, and
  // its owning temporary must stay alive through the dispatch.
  array wide5 =
      astype(rope_input(Shape{2, 6, 2, 7, 8}, 269), bfloat16, stream);
  array strided5 = slice(
      wide5, {0, 0, 0, 0, 0}, {2, 6, 2, 7, 8}, {1, 3, 1, 1, 1}, stream);
  REQUIRE(strided5.shape() == Shape{2, 2, 2, 7, 8});
  const double theta5 =
      rope_theta_bound(5, 7, 1.0f, false, {}, 10000.0f, 4);
  auto got5 = fast::rope(
      strided5, 8, false, 10000.0f, 1.0f, offset, std::nullopt, stream);
  require_matches_reference(
      got5,
      host_rope_reference(
          flat(contiguous(strided5, false, stream), stream),
          strided5.shape(),
          8,
          false,
          {},
          10000.0f,
          1.0,
          5),
      rope_reference_tolerance_bf16(theta5),
      stream,
      "rope bf16 5D strided vs f64");
}

TEST_CASE("fused rope bf16 chunked input matches the whole-input slice") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Whole-contiguous bf16 input matched the reference while the same
  // rows arriving as a sliced chunk diverged. The contract is exact:
  // the chunk rides the same promotion, kernel, and narrow over
  // identical values, so its output must be bit-identical to the
  // corresponding slice of the whole-input output. Each chunk offset is
  // the whole offset plus the slice start so positions line up.
  array offset = array(1, int32);
  // 3D sequence chunk: the defect shape class.
  array whole3 =
      astype(rope_input(Shape{2, 10, 16}, 271), bfloat16, stream);
  auto roped3 = fast::rope(
      whole3, 16, false, 10000.0f, 1.0f, offset, std::nullopt, stream);
  array chunk3 = slice(whole3, {0, 3, 0}, {2, 9, 16}, stream);
  REQUIRE(chunk3.shape() == Shape{2, 6, 16});
  array chunk_offset3 = array(4, int32);
  auto chunk3_out = fast::rope(
      chunk3, 16, false, 10000.0f, 1.0f, chunk_offset3, std::nullopt, stream);
  require_bit_equal(
      chunk3_out,
      contiguous(
          slice(roped3, {0, 3, 0}, {2, 9, 16}, stream), false, stream),
      stream,
      "rope bf16 3D chunk vs whole");

  // 4D sequence chunk.
  array whole4 =
      astype(rope_input(Shape{2, 4, 10, 16}, 277), bfloat16, stream);
  auto roped4 = fast::rope(
      whole4, 16, false, 10000.0f, 1.0f, offset, std::nullopt, stream);
  array chunk4 = slice(whole4, {0, 0, 2, 0}, {2, 4, 8, 16}, stream);
  REQUIRE(chunk4.shape() == Shape{2, 4, 6, 16});
  array chunk_offset4 = array(3, int32);
  auto chunk4_out = fast::rope(
      chunk4, 16, false, 10000.0f, 1.0f, chunk_offset4, std::nullopt, stream);
  require_bit_equal(
      chunk4_out,
      contiguous(
          slice(roped4, {0, 0, 2, 0}, {2, 4, 8, 16}, stream), false, stream),
      stream,
      "rope bf16 4D chunk vs whole");
}

TEST_CASE("fused rope vector and int64 offsets match the composed fallback") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  Shape shape{2, 2, 5, 16};
  array x = rope_input(shape, 151);
  // Per-batch offsets: FENCED to the composition (the fused kernel's
  // vector-offset leg failed equivalence on 2026-09-03), so this
  // asserts the fence routes correctly - the result must equal the
  // composed fallback exactly, element for element.
  std::vector<int32_t> batch_offsets{3, 9};
  array offset_vec = array(batch_offsets.begin(), Shape{2}, int32);
  require_bit_equal(
      fast::rope(x, 16, true, 10000.0f, 1.0f, offset_vec, std::nullopt, stream),
      composed_rope(
          x, 16, true, 10000.0f, 1.0f, offset_vec, std::nullopt, true,
          stream),
      stream,
      "rope vector offset (fenced)");
  // int64 offsets exercise the same kernel path through the
  // upstream wrapper's int32 cast. Cast-and-eval on the host produces the
  // same int32 offset the fused path sees, so this asserts only that the
  // wrapper round-trip is lossless, not that the kernel can read int64
  // directly. The omarchy backend does not currently carry an
  // int64-to-int32 device copy; mlx_lm passes int32 offsets in practice.
  // TODO(per-batch-offset-debug): the per-batch offset leg fails
  // 160 of 320 element checks (observed at float32, dims 16); the
  // fused value differs from the composed value by up to 0.5 absolute.
  // The scalar offset leg agrees bit-exactly on the same test fixture,
  // so the issue tracks the host broadcast of a multi-element offset
  // array into the per-batch reading. Deferred to a follow-up that
  // compares outputs of each per-batch lane in isolation.
  std::vector<int32_t> from_host_int32{4};
  array offset32 = array(from_host_int32.begin(), Shape{1}, int32);
  // The scalar-offset leg stays fused; f32 rides the contraction
  // tolerance (bit-equal is not achievable under the driver's fma).
  require_rope_close(
      fast::rope(x, 16, false, 10000.0f, 1.0f, offset32, std::nullopt, stream),
      composed_rope(
          x, 16, false, 10000.0f, 1.0f, offset32, std::nullopt, true,
          stream),
      stream,
      "rope int32 offset");
}

// The inverse/VJP path is FENCED to the composition (it failed
// equivalence on 2026-09-03), so this asserts the fence routes
// correctly: the gradient must equal the composed inverse rope of the
// cotangent exactly. When the fused inverse passes equivalence, this
// test is what proves the unfence.
TEST_CASE("fused rope vjp gradient rides the fence to the composition") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  Shape shape{2, 2, 5, 16};
  array x = rope_input(shape, 157);
  array cot = rope_input(shape, 163);
  array offset = array(2, int32);
  auto fun = [&](const std::vector<array>& inputs) {
    return std::vector<array>{fast::rope(
        inputs[0], 16, true, 10000.0f, 1.0f, offset, std::nullopt, stream)};
  };
  auto [outputs, grads] = vjp(fun, std::vector<array>{x}, {cot});
  require_bit_equal(
      grads[0],
      composed_rope(
          cot, 16, true, 10000.0f, 1.0f, offset, std::nullopt, false, stream),
      stream,
      "rope vjp (fenced)");
}

// Offset sweep: the tolerance bands are theta-dependent, so the
// battery must exercise the theta range the bands describe - a
// short-context battery never tests the upper bands. Positions scale
// the angle directly, so each offset lands in a different band of the
// measured sweep curve (shape (1,1,4,16), dims 16, base 1e4, scale 1:
// theta_max = offset + 3). Coordinated with RopeAccuracyVerdict's
// float64 position sweep: the f64 run establishes which path is
// closer to truth; this run establishes that the permanent battery
// covers every band. Offset 99990 tests the last band inside the
// gate; offset 100000 tests that the GATE FIRES (refusal, not
// accuracy - above theta 1e5 the built-in is untrusted by contract).
TEST_CASE("fused rope offset sweep validates every tolerance band") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  Shape shape{1, 1, 4, 16};
  struct SweepPoint {
    int offset;
    double theta_bound;
    const char* band;
  };
  const SweepPoint sweep[] = {
      {0, 3.0, "contraction (theta <= 100)"},
      {512, 515.0, "sweep point 1e3 (theta 515)"},
      {2048, 2051.0, "sweep point 5e3 (theta 2051)"},
      {8192, 8195.0, "sweep point 12345 (theta 8195)"},
      {99990, 99993.0, "sweep point 1e5 (theta 99993, last inside gate)"},
  };
  for (const auto& point : sweep) {
    array offset = array(point.offset, int32);
    for (auto dtype : {float32, float16}) {
      std::string what = std::string("rope offset sweep ") +
          std::to_string(point.offset) + " (" + point.band + ")" +
          (dtype == float32 ? " f32" : " f16");
      array x = astype(rope_input(shape, 173), dtype, stream);
      auto got = fast::rope(
          x, 16, false, 10000.0f, 1.0f, offset, std::nullopt, stream);
      auto want = composed_rope(
          x, 16, false, 10000.0f, 1.0f, offset, std::nullopt, true, stream);
      require_rope_close(got, want, stream, what, point.theta_bound);
    }
  }

  // The gate boundary itself: one step past theta 1e5 the fused path
  // refuses by name - the top of the sweep confirms the gate fires,
  // not that the kernel is accurate there.
  Shape dshape{1, 1, 4, 16};
  array x = astype(rope_input(dshape, 179), float32, stream);
  array past_gate = array(100000, int32);
  auto message = caught_message([&] {
    fast::rope(x, 16, false, 10000.0f, 1.0f, past_gate, std::nullopt, stream)
        .eval();
  });
  CHECK(message.find("[omarchy] RoPE") != std::string::npos);
  CHECK(message.find("exceeds the built-in accuracy limit") !=
        std::string::npos);
}

TEST_CASE("fused rope refuses beyond the trig argument limit by name") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Both legs of the gate: a 32k-class position passes and matches the
  // composed fallback, while a position past 1e5 refuses with the named
  // error instead of silently degrading.
  Shape shape{1, 1, 4, 16};
  array x = astype(rope_input(shape, 167), float16, stream);
  array near_limit = array(32000, int32);
  auto got =
      fast::rope(x, 16, false, 10000.0f, 1.0f, near_limit, std::nullopt, stream);
  auto want = composed_rope(
      x, 16, false, 10000.0f, 1.0f, near_limit, std::nullopt, true, stream);
  // At theta ~ 32 (32k position with the smallest inv_freq), the
  // f16 sin/cos rounding differs by 1 ulp in the LAST arithmetic
  // op relative to the composed per-op rounding chain. The gate's
  // contract is that the result is correct under the trusted envelope,
  // so a coarse absolute bound suffices.
  REQUIRE_EQ(got.shape(), want.shape());
  REQUIRE_EQ(got.dtype(), want.dtype());
  require_close(flat(got, stream), widen(flat(want, stream)), 1e-2,
                "rope 32k-class position");

  array over_limit = array(200000, int32);
  auto message = caught_message([&] {
    fast::rope(x, 16, false, 10000.0f, 1.0f, over_limit, std::nullopt, stream)
        .eval();
  });
  CHECK(message.find("[omarchy] RoPE") != std::string::npos);
  CHECK(message.find("exceeds the built-in accuracy limit") != std::string::npos);
  // The freqs leg carries the same gate: tiny freqs blow the bound up.
  array tiny_freqs = full({8}, 1e-8f, float32, stream);
  auto freqs_message = caught_message([&] {
    fast::rope(x, 16, false, std::nullopt, 1.0f, near_limit, tiny_freqs, stream)
        .eval();
  });
  CHECK(
      freqs_message.find("exceeds the built-in accuracy limit") !=
      std::string::npos);
}

namespace {

uint64_t alpha1_fnv1a64(const void* data, size_t bytes) {
  const uint8_t* p = (const uint8_t*)data;
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

// The bf16-alpha-fix D2 instrument's exact input generator; the pinned
// digest below is only meaningful against these bytes.
std::vector<float> alpha1_pattern(size_t count, uint32_t seed) {
  std::vector<float> data(count);
  uint32_t state = seed;
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    data[i] = ((state >> 8) & 0xFFFF) / 16384.0f - 2.0f;
  }
  return data;
}

struct Bf16FastGuard {
  Bf16FastGuard() {
    if (const char* value = std::getenv("MLX_OMARCHY_SDPA_BF16_FAST")) {
      original = value;
    }
    setenv("MLX_OMARCHY_SDPA_BF16_FAST", "1", 1);
  }
  ~Bf16FastGuard() {
    if (original.empty()) {
      unsetenv("MLX_OMARCHY_SDPA_BF16_FAST");
    } else {
      setenv("MLX_OMARCHY_SDPA_BF16_FAST", original.c_str(), 1);
    }
  }
  std::string original;
};

}  // namespace

TEST_CASE("sdpa bf16 fast scale==1.0 stays bit-identical to the pre-fix alpha==1 route") {
  if (!compute_available()) {
    return;
  }
  const auto& caps = omarchy::device(0).capabilities();
  const bool coopmat_device =
      caps.cooperative_matrix_f32_8 && caps.subgroup_size == 32;
  if (!coopmat_device || std::getenv("MLX_OMARCHY_NO_COOPMAT") != nullptr) {
    skip(
        "scale==1.0 bit pin exercises the MatmulBF16Coopmat scores"
        " route (subgroup 32 + cooperative_matrix_f32_8); other devices"
        " run the untouched non-coopmat kernel.");
    return;
  }
  Bf16FastGuard bf16_guard;
  Stream stream = gpu_stream();
  // The alpha==1 scores dispatch: scale == 1.0 passes alpha = 1.0f to
  // MatmulBF16Coopmat, which both the pre-fix gate (alpha == 1 required)
  // and the fixed gate (any alpha) admit, so both kernel versions served
  // this exact traffic before and after the fix. The digest was captured
  // in the bf16-alpha-fix D2 window, where the pre-fix kernel (relaxed
  // gate, pre-fix shader) and the fixed kernel produced byte-identical
  // dumps for this workload. A one-bit move fails the test - the drain
  // multiply by 1.0f must stay exact.
  const int B = 1, H = 2, KV = 1, qL = 64, kL = 128, D = 8;
  const float scale = 1.0f;
  auto q_data = alpha1_pattern(B * H * qL * D, 401);
  auto k_data = alpha1_pattern(B * KV * kL * D, 409);
  auto v_data = alpha1_pattern(B * KV * kL * D, 419);
  array q = astype(
      array(q_data.begin(), Shape{B, H, qL, D}, float32), bfloat16, stream);
  array k = astype(
      array(k_data.begin(), Shape{B, KV, kL, D}, float32), bfloat16, stream);
  array v = astype(
      array(v_data.begin(), Shape{B, KV, kL, D}, float32), bfloat16, stream);
  auto out = fast::scaled_dot_product_attention(
      q, k, v, scale, "", {}, std::nullopt, false, stream);
  out.eval();
  synchronize(stream);
  const uint16_t* bits = out.data<uint16_t>();
  uint64_t digest = alpha1_fnv1a64(bits, out.size() * 2);
  if (digest != 0xf1f70dbd2f2be747ull) {
    std::cout << "  [alpha1-pin] sdpa scale1 got=0x" << std::hex << digest
              << " want=0xf1f70dbd2f2be747" << std::dec << "\n";
  }
  CHECK_EQ(digest, 0xf1f70dbd2f2be747ull);
}

TEST_CASE("sdpa and gated delta gradients hold the zero CPU dispatch contract") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // ---- SDPA: grad through the fast primitive vs grad through the
  // composed reference graph on the same device (upstream #4563 wires
  // ScaledDotProductAttention::vjp; the Omarchy backend keeps
  // ScaledDotProductAttentionVJP::use_fallback true, so the backward is
  // the composed fallback, never a missing eval_gpu).
  const int B = 2, T = 4, H = 2, D = 4;
  const float scale = 1.0f / std::sqrt(float(D));
  auto q_data = pattern(B * T * H * D, 211);
  auto k_data = pattern(B * T * H * D, 223);
  auto v_data = pattern(B * T * H * D, 227);
  array q = array(q_data.begin(), Shape{B, T, H, D}, float32);
  array k = array(k_data.begin(), Shape{B, T, H, D}, float32);
  array v = array(v_data.begin(), Shape{B, T, H, D}, float32);

  auto sdpa_fun = [&](const std::vector<array>& inputs) {
    return fast::scaled_dot_product_attention(
        inputs[0], inputs[1], inputs[2], scale, "", {}, std::nullopt, false,
        stream);
  };
  auto composed_fun = [&](const std::vector<array>& inputs) {
    // softmax(Q K^T * scale) V in the [B, T, H, D] storage MLX uses.
    auto qt = transpose(inputs[0], 0, 2, 1, 3, stream);
    auto kt = transpose(inputs[1], 0, 2, 1, 3, stream);
    auto vt = transpose(inputs[2], 0, 2, 1, 3, stream);
    auto scores = multiply(
        matmul(qt, transpose(kt, 0, 1, 3, 2, stream), stream),
        array(scale, float32),
        stream);
    auto probs = softmax(scores, -1, stream);
    return transpose(
        matmul(probs, vt, stream), 0, 2, 1, 3, stream);
  };
  auto fast_grads = grad(sdpa_fun, std::vector<int>{0, 1, 2})({q, k, v});
  auto composed_grads = grad(composed_fun, std::vector<int>{0, 1, 2})({q, k, v});
  for (int arg = 0; arg < 3; ++arg) {
    require_close(
        flat(fast_grads[arg], stream),
        widen(flat(composed_grads[arg], stream)),
        5e-4,
        "sdpa grad arg " + std::to_string(arg));
  }

  // ---- GDN, composed shape (T > 1 keeps use_fallback true): grad through
  // the primitive vs the composed recursion on the same device.
  const int GB = 1, GT = 2, GH = 2, GD = 4;
  auto gq_data = pattern(GB * GT * GH * GD, 307);
  auto gk_data = pattern(GB * GT * GH * GD, 311);
  auto gv_data = pattern(GB * GT * GH * GD, 313);
  auto gg_data = pattern(GB * GT * GH, 317);
  auto gb_data = pattern(GB * GT * GH, 331);
  array gq = array(gq_data.begin(), Shape{GB, GT, GH, GD}, float32);
  array gk = array(gk_data.begin(), Shape{GB, GT, GH, GD}, float32);
  array gv = array(gv_data.begin(), Shape{GB, GT, GH, GD}, float32);
  array gg = array(gg_data.begin(), Shape{GB, GT, GH}, float32);
  array gb = array(gb_data.begin(), Shape{GB, GT, GH}, float32);
  array gh0 = zeros({GB, GH, GD, GD}, float32, stream);

  auto gdn_fun = [&](const std::vector<array>& inputs) {
    auto pair = fast::gated_delta_update(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        std::nullopt,
        stream);
    return pair[0];
  };
  auto gdn_ref = [&](const std::vector<array>& inputs) {
    // The composed arithmetic the backend's fallback composes: per-token
    // state scan with decay g and write strength beta.
    auto state = inputs[5];
    std::vector<array> outputs;
    for (int t = 0; t < GT; ++t) {
      auto get_t = [&](const array& arr) {
        auto sliced = slice(
            arr,
            Shape{0, t, 0},
            Shape{arr.shape(0), t + 1, arr.shape(2)},
            stream);
        return squeeze(sliced, 1, stream);
      };
      auto q_t = get_t(inputs[0]);
      auto k_t = get_t(inputs[1]);
      auto v_t = get_t(inputs[2]);
      auto g_t = get_t(inputs[3]);
      auto beta_t = get_t(inputs[4]);
      auto decay = expand_dims(g_t, {-1, -2}, stream);
      auto state_next = multiply(state, decay, stream);
      auto kv = sum(
          multiply(state_next, expand_dims(k_t, -2, stream), stream),
          -1,
          false,
          stream);
      auto delta = multiply(
          subtract(v_t, kv, stream), expand_dims(beta_t, -1, stream), stream);
      state_next = add(
          state_next,
          multiply(
              expand_dims(delta, -1, stream),
              expand_dims(k_t, -2, stream),
              stream),
          stream);
      state = state_next;
      outputs.push_back(
          sum(multiply(state, expand_dims(q_t, -2, stream), stream),
              -1,
              false,
              stream));
    }
    return stack(outputs, 1, stream);
  };
  auto gdn_fast_grads =
      grad(gdn_fun, std::vector<int>{0, 1, 2, 3, 4})({gq, gk, gv, gg, gb, gh0});
  auto gdn_ref_grads =
      grad(gdn_ref, std::vector<int>{0, 1, 2, 3, 4})({gq, gk, gv, gg, gb, gh0});
  for (size_t arg = 0; arg < gdn_fast_grads.size(); ++arg) {
    require_close(
        flat(gdn_fast_grads[arg], stream),
        widen(flat(gdn_ref_grads[arg], stream)),
        5e-4,
        "gdn grad arg " + std::to_string(arg));
  }

  // ---- GDN, fused decode shape (T == 1, bf16, square 128 heads): the
  // forward takes GatedDeltaDecodeBF16 and the backward must take the
  // GatedDeltaUpdateVJP composed fallback (upstream #4565), never a
  // missing eval_gpu.
  const int FB = 1, FH = 2, FD = 128;
  auto fq_data = pattern(FB * FH * FD, 347);
  auto fk_data = pattern(FB * FH * FD, 349);
  auto fv_data = pattern(FB * FH * FD, 353);
  array fq = astype(
      array(fq_data.begin(), Shape{FB, 1, FH, FD}, float32),
      bfloat16,
      stream);
  array fk = astype(
      array(fk_data.begin(), Shape{FB, 1, FH, FD}, float32),
      bfloat16,
      stream);
  array fv = astype(
      array(fv_data.begin(), Shape{FB, 1, FH, FD}, float32),
      bfloat16,
      stream);
  array fg = astype(
      array(pattern(FB * FH, 359).begin(), Shape{FB, 1, FH}, float32),
      bfloat16,
      stream);
  array fbeta = astype(
      array(pattern(FB * FH, 367).begin(), Shape{FB, 1, FH}, float32),
      bfloat16,
      stream);
  array fh0 = zeros({FB, FH, FD, FD}, float32, stream);
  auto gdn_fused_fun = [&](const std::vector<array>& inputs) {
    auto pair = fast::gated_delta_update(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        inputs[4],
        inputs[5],
        std::nullopt,
        stream);
    return pair[0];
  };
  auto cot_out = ones({FB, 1, FH, FD}, float32, stream);
  auto [fused_out, fused_vjps] =
      vjp(gdn_fused_fun,
          std::vector<array>{fq, fk, fv, fg, fbeta, fh0},
          std::vector<array>{cot_out});
  auto fused_dq = flat(fused_vjps[0], stream);
  auto fused_ref_grads =
      grad(gdn_ref, std::vector<int>{0})({fq, fk, fv, fg, fbeta, fh0});
  require_close(fused_dq, widen(flat(fused_ref_grads[0], stream)), 2e-2, "gdn fused dq");

  // ---- Zero CPU dispatch: every tensor op above ran on the Omarchy GPU
  // evaluator (dispatch counters advanced); an explicit CPU stream moves
  // no GPU work, and the GPU counters never moved for CPU-stream work.
  uint64_t gpu_prims = omarchy::trace::counters().gpu_primitive_dispatches.load();
  uint64_t vk_dispatches = omarchy::trace::counters().vk_compute_dispatches.load();
  CHECK_MESSAGE(
      gpu_prims > 0,
      "zero GPU primitive dispatches: gradients never reached the GPU");
  CHECK_MESSAGE(
      vk_dispatches > 0,
      "zero Vulkan compute dispatches: gradients never reached the GPU");
  CHECK_EQ(default_device(), Device::gpu);
  if (cpu::is_available()) {
    uint64_t gpu_prims_before =
        omarchy::trace::counters().gpu_primitive_dispatches.load();
    uint64_t vk_before =
        omarchy::trace::counters().vk_compute_dispatches.load();
    Stream cpu_stream = new_stream(Device::cpu);
    auto probe = add(
        array({1.0f, 2.0f}, Shape{2}, float32),
        array({3.0f, 4.0f}, Shape{2}, float32),
        cpu_stream);
    probe.eval();
    synchronize(cpu_stream);
    // The CPU stream ran its own tensors; the grad work above stayed on
    // the GPU. Zero GPU primitives may move to a CPU stream.
    CHECK_EQ(
        omarchy::trace::counters().gpu_primitive_dispatches.load(),
        gpu_prims_before);
    CHECK_EQ(
        omarchy::trace::counters().vk_compute_dispatches.load(), vk_before);
  }
}
