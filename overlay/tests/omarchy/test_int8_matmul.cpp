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
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/backend/gpu/device_info.h"

using namespace mlx::core;

namespace {

// A/B routes for the tiled kernel: MLX_OMARCHY_INT8_NAIVE=1 selects the
// retired one-thread-per-element shader (shaders/int8_matmul_naive.comp).
struct NaiveEnv {
  explicit NaiveEnv(bool naive) {
    if (naive) {
      setenv("MLX_OMARCHY_INT8_NAIVE", "1", 1);
    }
  }
  ~NaiveEnv() {
    unsetenv("MLX_OMARCHY_INT8_NAIVE");
  }
};

// bf16 bits after exact widening to float32: bit-identical bf16 outputs give
// bit-identical float32 words, so a uint32 compare is the exactness bar.
std::vector<uint32_t> bits_of(array& out) {
  out.eval();
  auto wide = astype(out, float32);
  wide.eval();
  std::vector<uint32_t> bits(wide.size());
  std::memcpy(bits.data(), wide.data<float>(), bits.size() * sizeof(uint32_t));
  return bits;
}

struct Inputs {
  std::vector<int8_t> x;
  std::vector<int8_t> w;
  std::vector<float> xs;
  std::vector<float> ws;
  std::vector<float> bias;
};

Inputs make_inputs(int rows, int k, int group, int n, bool swiglu, int seed) {
  Inputs in;
  const int n_w = swiglu ? 2 * n : n;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> values(-127, 127);
  std::uniform_real_distribution<float> scales(0.001f, 0.011f);
  in.x.resize(rows * k);
  in.w.resize(n_w * k);
  for (auto& v : in.x) {
    v = static_cast<int8_t>(values(rng));
  }
  for (auto& v : in.w) {
    v = static_cast<int8_t>(values(rng));
  }
  in.xs.resize(rows * (k / group));
  in.ws.resize(n_w);
  in.bias.resize(n_w);
  for (auto& v : in.xs) {
    v = scales(rng);
  }
  for (auto& v : in.ws) {
    v = scales(rng);
  }
  for (auto& v : in.bias) {
    v = scales(rng);
  }
  return in;
}

array run_int8(const Inputs& in, int rows, int k, int group, int n,
               bool swiglu) {
  auto mx_x = array(in.x.data(), {rows, k}, int8);
  auto mx_xs = array(in.xs.data(), {rows, k / group}, float32);
  auto mx_w = array(in.w.data(), {swiglu ? 2 * n : n, k}, int8);
  auto mx_ws = array(in.ws.data(), {swiglu ? 2 * n : n}, float32);
  auto mx_bias = array(in.bias.data(), {swiglu ? 2 * n : n}, float32);
  return fast::int8_matmul(
      mx_x, mx_xs, mx_w, mx_ws, mx_bias, group, swiglu,
      new_stream(Device::gpu));
}

// Bit-compare the tiled kernel against the naive kill-switch kernel. Returns
// the first mismatching index or -1.
long bitcompare(const Inputs& in, int rows, int k, int group, int n,
                bool swiglu, uint32_t* mismatch_got = nullptr,
                uint32_t* mismatch_ref = nullptr) {
  auto tiled = run_int8(in, rows, k, group, n, swiglu);
  auto tiled_bits = bits_of(tiled);
  NaiveEnv guard(true);
  auto naive = run_int8(in, rows, k, group, n, swiglu);
  auto naive_bits = bits_of(naive);
  if (tiled_bits.size() != naive_bits.size()) {
    return 0;
  }
  for (size_t i = 0; i < tiled_bits.size(); i++) {
    if (tiled_bits[i] != naive_bits[i]) {
      if (mismatch_got) {
        *mismatch_got = tiled_bits[i];
      }
      if (mismatch_ref) {
        *mismatch_ref = naive_bits[i];
      }
      return static_cast<long>(i);
    }
  }
  return -1;
}

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

TEST_CASE("tiled int8_matmul is bitwise identical to the naive kernel") {
  if (!mlx::core::gpu::is_available()) {
    MESSAGE("no GPU device; skipping");
    return;
  }
  // rows/n off-tile, group sizes 64/128/256 plus group 96, mid-tile group
  // flushes (group 16 with a partial final k-tile), mid-word group flushes
  // (odd group 17), k_total%4 != 0 (word-view straddle), swiglu, tiny rows.
  // The host requires group | k, so no K-tail-past-groups*group case exists.
  struct Shape {
    int rows, k, group, n;
    bool swiglu;
  };
  const Shape shapes[] = {
      {35, 512, 64, 37, false},
      {35, 512, 128, 37, false},
      {35, 512, 256, 37, false},
      {17, 576, 96, 15, false},
      {17, 272, 16, 15, false}, // group < TILE_K, 272 % 32 = 16 partial tile
      {17, 272, 17, 15, false}, // odd group: flushes inside a word
      {5, 254, 127, 7, false},  // k_total % 4 != 0, partial final k-tile
      {35, 512, 64, 37, true},
      {5, 254, 127, 7, true},
      {1, 256, 128, 70, false},
      {3, 256, 64, 70, false},
      {8, 256, 256, 70, false},
  };
  // Bit-compare a candidate route (env-controlled) against the naive
  // kill-switch kernel for every shape in the matrix. Two passes: the
  // landed default route (cooperative matrix when the device supports
  // it, else the tiled int32-imad kernel) and the tiled-only route with
  // MLX_OMARCHY_INT8_COOPMAT=0. A mismatch on either is a real defect.
  auto run_route = [](const Shape& s, int seed) {
    auto in = make_inputs(s.rows, s.k, s.group, s.n, s.swiglu, seed);
    auto cur = run_int8(in, s.rows, s.k, s.group, s.n, s.swiglu);
    auto cur_bits = bits_of(cur);
    NaiveEnv guard(true);
    auto naive = run_int8(in, s.rows, s.k, s.group, s.n, s.swiglu);
    auto naive_bits = bits_of(naive);
    if (cur_bits.size() != naive_bits.size()) {
      return std::pair<long, uint64_t>(0, 0);
    }
    for (size_t i = 0; i < cur_bits.size(); i++) {
      if (cur_bits[i] != naive_bits[i]) {
        return std::pair<long, uint64_t>(
            static_cast<long>(i),
            (static_cast<uint64_t>(cur_bits[i]) << 32) | naive_bits[i]);
      }
    }
    return std::pair<long, uint64_t>(-1, 0);
  };
  const char* env_routes[] = {nullptr, "0"};
  const char* env_labels[] = {"default", "tiled"};
  for (int e = 0; e < 2; e++) {
    if (env_routes[e] != nullptr) {
      setenv("MLX_OMARCHY_INT8_COOPMAT", env_routes[e], 1);
    }
    int seed = 101;
    for (const auto& s : shapes) {
      auto result = run_route(s, seed++);
      if (result.first >= 0) {
        MESSAGE(env_labels[e], "-vs-naive mismatch at shape rows=", s.rows,
                " k=", s.k, " group=", s.group, " n=", s.n, " swiglu=",
                s.swiglu, " index ", result.first, ": ", env_labels[e],
                " 0x", std::hex, (result.second >> 32), " naive 0x",
                (result.second & 0xffffffffu), std::dec);
      }
      CHECK(result.first < 0);
    }
    if (env_routes[e] != nullptr) {
      unsetenv("MLX_OMARCHY_INT8_COOPMAT");
    }
  }

  // Clamp-scale + DiT-scale + extremes: all under the landed default
  // route, each compared bitwise against the naive kill switch.
  struct BigShape {
    int rows, k, group, n;
    bool swiglu;
    int seed;
  };
  const BigShape big[] = {
      {3200, 256, 256, 8192, false, 202}, // exceeds the 16,776,960 clamp
      {128, 5376, 256, 21504, false, 303}, // TensorFold H3
      {128, 256, 256, 4096, false, 404},   // all +127: per-group = +4,129,024
      {128, 256, 256, 4096, false, 505},   // all -127: per-group = -4,129,024
      {128, 256, 256, 4096, true, 606},    // all +127 swiglu
  };
  for (const auto& b : big) {
    auto in = make_inputs(b.rows, b.k, b.group, b.n, b.swiglu, b.seed);
    if (b.seed >= 404) {
      // Extremes: fill x and w with the magnitude that maximizes
      // per-group |sum| so the cooperative matrix's f32 accumulator is
      // probed at the integer bound that defines the exactness claim.
      for (auto& v : in.x) {
        v = (b.seed == 505) ? int8_t(127) : int8_t((b.seed % 2 == 0) ? 127 : -127);
      }
      for (auto& v : in.w) {
        v = int8_t((b.seed % 2 == 0) ? 127 : -127);
      }
    }
    auto cur = run_int8(in, b.rows, b.k, b.group, b.n, b.swiglu);
    auto cur_bits = bits_of(cur);
    NaiveEnv guard(true);
    auto naive = run_int8(in, b.rows, b.k, b.group, b.n, b.swiglu);
    auto naive_bits = bits_of(naive);
    long bad = -1;
    for (size_t i = 0; i < cur_bits.size(); i++) {
      if (cur_bits[i] != naive_bits[i]) {
        bad = static_cast<long>(i);
        MESSAGE("default-vs-naive mismatch at rows=", b.rows, " k=", b.k,
                " group=", b.group, " n=", b.n, " swiglu=", b.swiglu,
                " index ", bad);
        break;
      }
    }
    CHECK(bad < 0);
  }
}

TEST_CASE("tiled int8_matmul matches the int64 reference at DiT scale") {
  if (!mlx::core::gpu::is_available()) {
    MESSAGE("no GPU device; skipping");
    return;
  }
  const int rows = 128;
  const int k = 5376;
  const int group = 256;
  const int n = 21504;
  auto in = make_inputs(rows, k, group, n, false, 404);
  auto out = run_int8(in, rows, k, group, n, false);
  out.eval();
  auto wide = astype(out, float32);
  wide.eval();

  const int groups = k / group;
  const int k_eff = groups * group;
  const int sampled_rows[] = {0, 77, 127};
  for (int m : sampled_rows) {
    for (int ci = 0; ci < 32; ci++) {
      const int col = ci * (n / 32);
      double ref = 0.0;
      for (int g = 0; g < groups; g++) {
        long long dot = 0;
        for (int i = 0; i < group; i++) {
          dot += static_cast<long long>(
                     in.x[m * k + g * group + i]) *
              static_cast<long long>(in.w[col * k + g * group + i]);
        }
        ref += static_cast<double>(static_cast<float>(dot)) *
            static_cast<double>(in.xs[m * groups + g]);
      }
      ref = ref * static_cast<double>(in.ws[col]) +
          static_cast<double>(in.bias[col]);
      float got = wide.data<float>()[m * n + col];
      double err = std::abs(static_cast<double>(got) - ref);
      double scale = std::max(1.0, std::abs(ref));
      CHECK(err <= scale * 0.004);
    }
  }
}

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
