// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
//
// Capability-simulation tests (MLX_OMARCHY_CAPS_SIM, capability_sim.h).
//
// One binary, one profile per invocation: main() validates argv[1]
// against the profile registry, exports MLX_OMARCHY_CAPS_SIM, and lets
// doctest run the per-profile cases. The CMake wiring registers one
// ctest entry per profile.
//
// Per profile the cases assert the INTENDED outcome only: a correct
// result through the gate-chosen fallback, or a loud named refusal when
// a simulated axis selects a kernel the runtime hardware cannot
// execute. Nothing here accepts a silently different numeric path.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/capability_sim.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/device.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

// The registered profiles, asserted against capsim::profiles() in main.
constexpr const char* kM1Fork = "m1-honeykrisp-fork";
constexpr const char* kM1Stock = "m1-stock-no-coopmat";
constexpr const char* kSubgroup64 = "subgroup-size-64";
constexpr const char* kSmallSmem = "small-shared-memory";
constexpr const char* kNoCoopmat = "no-cooperative-matrix";

std::string g_profile;

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1"
         " on a development machine).");
    return false;
  }
  return true;
}

const omarchy::CapabilityReport& caps() {
  return omarchy::device(0).capabilities();
}

const omarchy::CapabilityReport& hw() {
  return omarchy::device(0).hardware_capabilities();
}

bool float16_available() {
  return caps().shader_float16 && caps().storage_buffer_16bit_access;
}

bool profile_is(const char* name) {
  return g_profile == name;
}

// What the production gates decide under this profile, read straight
// off the simulated report with the same expressions as
// primitives.cpp. These are expectations, not dispatch observers: the
// per-op cases prove the outcome the table predicts.
struct GateExpectations {
  bool coopmat_claim;
  bool subgroup32_arithmetic; // qmm vec subgroup gate
  bool bf16_gemv_gate;        // subgroup32 + SHUFFLE_RELATIVE
  bool sdpa_native_gate;      // coopmat+32+wg>=1024+smem>=21504+ops
};

GateExpectations expectations() {
  const auto& c = caps();
  constexpr VkSubgroupFeatureFlags kDecodeOps =
      VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
      VK_SUBGROUP_FEATURE_SHUFFLE_BIT;
  GateExpectations e;
  e.coopmat_claim = c.cooperative_matrix_f32_8 && c.subgroup_size == 32u;
  e.subgroup32_arithmetic =
      c.subgroup_size == 32u &&
      (c.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
  e.bf16_gemv_gate =
      c.subgroup_size == 32u &&
      (c.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0;
  e.sdpa_native_gate = e.coopmat_claim &&
      c.max_compute_work_group_invocations >= 1024u &&
      c.max_compute_work_group_size[0] >= 1024u &&
      c.max_compute_shared_memory_size >= (32u * 32u + 2u * 128u + 128u * 32u) *
              sizeof(float) &&
      (c.subgroup_operations & kDecodeOps) == kDecodeOps;
  return e;
}

std::string evaluation_error(array value) {
  try {
    value.eval();
  } catch (const std::exception& error) {
    return error.what();
  }
  return {};
}

std::vector<float> readback_f32(const Stream& stream, array value) {
  value = astype(value, float32, stream);
  value.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = value.data<float>();
  return std::vector<float>(data, data + value.size());
}

// Round values through a 16-bit dtype on the device so host references
// see what the kernel sees.
std::vector<float> round_trip(
    const Stream& stream,
    const std::vector<float>& values,
    Dtype dtype) {
  array device(values.begin(), Shape{static_cast<int>(values.size())}, float32);
  return readback_f32(
      stream, astype(astype(device, dtype, stream), float32, stream));
}

float host_bf16_round(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, 4);
  bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
  std::memcpy(&value, &bits, 4);
  return value;
}

struct HostQuantizedWeights {
  std::vector<uint32_t> words;
  std::vector<float> scales;
  std::vector<float> biases;
};

// Host affine quantizer matching the upstream affine_quantize kernel
// (same construction as test_matmul_family.cpp).
HostQuantizedWeights host_affine_quantize(
    const std::vector<float>& matrix,
    int rows,
    int cols,
    int group_size,
    int bits) {
  HostQuantizedWeights result;
  int groups = cols / group_size;
  int pack = 32 / bits;
  int words_per_row = cols / pack;
  float n_bins = static_cast<float>((1 << bits) - 1);
  result.words.assign(static_cast<size_t>(rows) * words_per_row, 0);
  result.scales.resize(static_cast<size_t>(rows) * groups);
  result.biases.resize(static_cast<size_t>(rows) * groups);
  for (int row = 0; row < rows; ++row) {
    for (int group = 0; group < groups; ++group) {
      float w_max = -std::numeric_limits<float>::infinity();
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
        float q = std::clamp(std::round((value - bias) / scale), 0.0f, n_bins);
        uint32_t code = static_cast<uint32_t>(q);
        int col = group * group_size + i;
        result.words[row * words_per_row + col / pack] |=
            code << ((col % pack) * bits);
      }
    }
  }
  return result;
}

// Double-precision host dot of one x row against dequantized weight
// rows (transpose layout: w rows are [n][k]).
std::vector<float> host_quantized_matmul(
    const HostQuantizedWeights& w,
    const std::vector<float>& x,
    int m,
    int n,
    int k,
    int group_size,
    int bits) {
  int pack = 32 / bits;
  int words_per_row = k / pack;
  int groups = k / group_size;
  uint32_t mask = (1u << bits) - 1u;
  std::vector<float> out(static_cast<size_t>(m) * n);
  for (int row = 0; row < m; ++row) {
    for (int column = 0; column < n; ++column) {
      double acc = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        uint32_t code =
            (w.words[column * words_per_row + inner / pack] >>
             ((inner % pack) * bits)) &
            mask;
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

void expect_close(
    const std::vector<float>& device,
    const std::vector<float>& expected,
    double atol,
    double rtol) {
  REQUIRE_EQ(device.size(), expected.size());
  for (size_t index = 0; index < expected.size(); ++index) {
    INFO("index ", index, " got ", device[index], " expected ",
         expected[index]);
    CHECK(std::fabs(device[index] - expected[index]) <=
        atol + rtol * std::fabs(expected[index]));
  }
}

// True when the profile's coopmat claim is backed by the runtime
// hardware (real M1 fork); false means the dispatch must refuse.
bool coopmat_backed() {
  return hw().cooperative_matrix_f32_8;
}

void expect_simulated_refusal(
    const std::string& message,
    const char* kernel) {
  INFO("refusal message: ", message);
  REQUIRE(message.find("capability simulation") != std::string::npos);
  REQUIRE(message.find(g_profile) != std::string::npos);
  REQUIRE(message.find(kernel) != std::string::npos);
  REQUIRE(message.find("cooperative_matrix_fp32_8x8x8") !=
      std::string::npos);
}

} // namespace

TEST_CASE("capability report is stamped with the active profile") {
  if (!compute_available()) {
    return;
  }
  const auto& c = caps();
  REQUIRE(c.simulated);
  REQUIRE(c.simulation_profile == g_profile);
  REQUIRE_FALSE(c.simulated_driver_variant.empty());

  // Hardware truth stays hardware: unstamped, same physical device.
  const auto& h = hw();
  REQUIRE_FALSE(h.simulated);
  CHECK(h.device_name == c.device_name);
  CHECK(h.driver_name == c.driver_name);

  // Axis expectations per profile (docs/compatibility-matrix.md rows).
  if (profile_is(kM1Fork)) {
    CHECK_EQ(c.subgroup_size, 32u);
    CHECK(c.cooperative_matrix_f32_8);
    CHECK((c.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0);
    CHECK((c.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0);
    CHECK((c.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) !=
        0);
  } else if (profile_is(kM1Stock)) {
    CHECK_EQ(c.subgroup_size, 32u);
    CHECK_FALSE(c.cooperative_matrix_f32_8);
    CHECK((c.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0);
  } else if (profile_is(kSubgroup64)) {
    CHECK_EQ(c.subgroup_size, 64u);
    CHECK_FALSE(c.cooperative_matrix_f32_8);
    CHECK((c.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0);
  } else if (profile_is(kSmallSmem)) {
    CHECK_EQ(c.subgroup_size, 32u);
    CHECK(c.cooperative_matrix_f32_8);
    CHECK_EQ(c.max_compute_shared_memory_size, size_t{16384});
  } else if (profile_is(kNoCoopmat)) {
    CHECK_EQ(c.subgroup_size, 64u);
    CHECK_FALSE(c.cooperative_matrix_f32_8);
    CHECK((c.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0);
    CHECK((c.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) == 0);
    CHECK((c.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) ==
        0);
  } else {
    FAIL("unhandled profile ", g_profile);
  }

  // Provenance output carries the stamp too.
  auto info = gpu::device_info(0);
  auto it = info.find("simulated");
  REQUIRE(it != info.end());
  CHECK(std::get<size_t>(it->second) == size_t{1});
}

TEST_CASE("q4 g64 decode gemv matches the f64 host reference") {
  if (!compute_available() || !float16_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto e = expectations();

  constexpr int k = 256; // group 64 x 4; also k % 128 == 0
  constexpr int n = 8;
  std::mt19937 gen(20260911);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> matrix(static_cast<size_t>(n) * k);
  std::vector<float> x_values(k);
  for (auto& v : matrix) {
    v = dist(gen);
  }
  for (auto& v : x_values) {
    v = dist(gen);
  }
  auto host = host_affine_quantize(matrix, n, k, 64, 4);
  auto scales_rt = round_trip(stream, host.scales, float16);
  auto biases_rt = round_trip(stream, host.biases, float16);
  HostQuantizedWeights rounded = host;
  rounded.scales = scales_rt;
  rounded.biases = biases_rt;
  auto expected =
      host_quantized_matmul(rounded, x_values, 1, n, k, 64, 4);

  int pack = 8;
  int words_per_row = k / pack;
  int groups_per_row = k / 64;
  array x(x_values.begin(), Shape{1, k}, float16);
  array w_words(host.words.begin(), Shape{n, words_per_row}, uint32);
  array scales(scales_rt.begin(), Shape{n, groups_per_row}, float16);
  array biases(biases_rt.begin(), Shape{n, groups_per_row}, float16);
  array out = quantized_matmul(
      x, w_words, scales, biases, true, 64, 4, "affine", stream);
  // The subgroupAdd variant assumes each 32-lane logical slot IS one
  // physical subgroup. When the profile claims 32+ARITHMETIC and the
  // runtime hardware cannot back it, the dispatch must refuse by name
  // - executing it over smaller physical subgroups would silently sum
  // only slices of each slot. On real 32-lane hardware the check
  // below is the production proof; otherwise the fallback must hold
  // the documented GEMV envelope (test_matmul_family derives it).
  std::string gemv_error = evaluation_error(out);
  if (e.subgroup32_arithmetic &&
      !(hw().subgroup_size == 32u &&
          (hw().subgroup_operations &
              VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0)) {
    std::cout << "[capsim q4-gemv] profile=" << g_profile
              << " subgroup_gate=1 -> named refusal (physical subgroup"
              << " size " << hw().subgroup_size << ")\n";
    REQUIRE(gemv_error.find("capability simulation") !=
        std::string::npos);
    REQUIRE(gemv_error.find("QmmVec") != std::string::npos);
    return;
  }
  REQUIRE_MESSAGE(gemv_error.empty(), gemv_error);
  auto got = readback_f32(stream, out);
  // Documented GEMV envelope: (3*K/32 + 32) f32 multiply-adds against
  // the f64 reference, plus f16 storage of the output (2^-11
  // relative), with 1.5x slack for reduction order across the
  // gate-chosen kernel variants.
  float out_max = 0.0f;
  for (const auto v : expected) {
    out_max = std::max(out_max, std::fabs(v));
  }
  const float steps = 3.0f * k / 32.0f + 32.0f;
  const float bound =
      out_max * (steps * 4.5f * 0x1p-23f + 2.0f / 2048.0f) * 1.5f;
  std::cout << "[capsim q4-gemv] profile=" << g_profile
            << " subgroup_gate=" << e.subgroup32_arithmetic
            << " bound=" << bound << "\n";
  expect_close(got, expected, bound, 1e-5);
}

TEST_CASE("dense bf16 decode gemv matches the f64 bound") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto e = expectations();

  // Operands pre-quantized to the bf16 grid; the f64 reference sums
  // exact products (same construction as test_matmul_family).
  std::mt19937 rng(20260911);
  std::uniform_real_distribution<float> frac(-2.0f, 2.0f);
  auto grid = [&](size_t count) {
    std::vector<float> values(count);
    for (auto& v : values) {
      v = host_bf16_round(frac(rng));
    }
    return values;
  };

  // k % 128 == 0 and n % 4 == 0 so the dense BF16 decode GEMV gate is
  // the only deciding condition left.
  constexpr int k = 128;
  constexpr int n = 8;
  auto a_values = grid(k);
  auto w_nt = grid(static_cast<size_t>(k) * n);
  std::vector<float> sums(n, 0.0f);
  float partial_max = 0.0f;
  for (int c = 0; c < n; ++c) {
    double acc = 0.0;
    for (int r = 0; r < k; ++r) {
      acc += static_cast<double>(a_values[r]) *
          w_nt[static_cast<size_t>(c) * k + r];
    }
    sums[c] = static_cast<float>(acc);
    partial_max = std::max(partial_max, std::fabs(sums[c]));
  }
  const float steps = static_cast<float>(k / 8 + 5);
  const float bound =
      partial_max * (steps * 4.5f * 0x1p-23f + 2.0f / 256.0f) +
      2.0f / 256.0f;
  std::string bf16_error;
  array a = astype(
      array(a_values.begin(), Shape{1, k}, float32), bfloat16, stream);
  array b = transpose(
      astype(
          array(w_nt.begin(), Shape{n, k}, float32), bfloat16, stream),
      {1, 0},
      stream);
  auto run_bf16 = [&]() {
    return readback_f32(stream, matmul(a, b, stream));
  };
  if (!e.bf16_gemv_gate) {
    // Gate off: the tile fallback runs and the documented bound holds.
    auto got = run_bf16();
    float max_abs_err = 0.0f;
    for (int i = 0; i < n; ++i) {
      max_abs_err = std::max(max_abs_err, std::fabs(got[i] - sums[i]));
    }
    std::cout << "[capsim bf16-gemv] profile=" << g_profile
              << " gemv_gate=0 max_abs_err=" << max_abs_err
              << " bound=" << bound << "\n";
    CHECK(max_abs_err <= bound);
    return;
  }
  // Gate on: the dense BF16 decode GEMV's shuffle-down reduction is
  // defined only when the PHYSICAL subgroup is the same 32 lanes the
  // kernel's logical layout assumes. When the runtime hardware cannot
  // back that combination, the dispatch refuses by name instead of
  // silently producing wrong numbers (that silent trap is exactly what
  // the simulation exists to keep impossible). On real 32-lane
  // hardware the bound check below is the production proof.
  bool hw_backs = hw().subgroup_size == 32u &&
      (hw().subgroup_operations &
          VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0;
  if (!hw_backs) {
    bf16_error = evaluation_error(matmul(a, b, stream));
    std::cout << "[capsim bf16-gemv] profile=" << g_profile
              << " gemv_gate=1 -> named refusal (physical subgroup"
              << " size " << hw().subgroup_size << ")\n";
    REQUIRE(bf16_error.find("capability simulation") !=
        std::string::npos);
    REQUIRE(bf16_error.find("MatmulVecBF16") != std::string::npos);
    return;
  }
  auto got = run_bf16();
  float max_abs_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    max_abs_err = std::max(max_abs_err, std::fabs(got[i] - sums[i]));
  }
  std::cout << "[capsim bf16-gemv] profile=" << g_profile
            << " gemv_gate=1 hw32=1 max_abs_err=" << max_abs_err
            << " bound=" << bound << "\n";
  CHECK(max_abs_err <= bound);
}

TEST_CASE("coopmat f32 matmul routes per the profile") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto e = expectations();

  // m > 1, k % 8 == 0, alpha 1, no bias: the dense coopmat gate shape.
  constexpr int m = 16;
  constexpr int k = 16;
  constexpr int n = 16;
  std::vector<float> a_values(static_cast<size_t>(m) * k);
  std::vector<float> b_values(static_cast<size_t>(k) * n);
  std::mt19937 gen(7);
  std::uniform_int_distribution<int> small(-4, 4);
  for (auto& v : a_values) {
    v = static_cast<float>(small(gen));
  }
  for (auto& v : b_values) {
    v = static_cast<float>(small(gen));
  }
  std::vector<float> expected(static_cast<size_t>(m) * n, 0.0f);
  for (int r = 0; r < m; ++r) {
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      for (int inner = 0; inner < k; ++inner) {
        acc += static_cast<double>(a_values[r * k + inner]) *
            b_values[inner * n + c];
      }
      expected[r * n + c] = static_cast<float>(acc);
    }
  }

  array a(a_values.begin(), Shape{m, k}, float32);
  array b(b_values.begin(), Shape{k, n}, float32);
  std::string error = evaluation_error(matmul(a, b, stream));
  if (e.coopmat_claim && !coopmat_backed()) {
    // Simulated matrix unit over hardware without one: loud, named.
    std::cout << "[capsim f32-coopmat] profile=" << g_profile
              << " -> named refusal\n";
    expect_simulated_refusal(error, "MatmulF32Coopmat");
    return;
  }
  REQUIRE(error.empty());
  expect_close(readback_f32(stream, matmul(a, b, stream)),
      expected,
      1e-5,
      1e-6);
  std::cout << "[capsim f32-coopmat] profile=" << g_profile << " -> "
            << (e.coopmat_claim ? "MatmulF32Coopmat" : "16x16 tile")
            << " (backed=" << coopmat_backed() << ")\n";
}

TEST_CASE("f16 q4 g64 prefill routes per the profile") {
  if (!compute_available() || !float16_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto e = expectations();
  constexpr int m = 64;
  constexpr int k = 128;
  constexpr int n = 32;
  std::mt19937 gen(11);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<float> matrix(static_cast<size_t>(n) * k);
  std::vector<float> x_values(static_cast<size_t>(m) * k);
  for (auto& v : matrix) {
    v = dist(gen);
  }
  for (auto& v : x_values) {
    v = dist(gen);
  }
  auto host = host_affine_quantize(matrix, n, k, 64, 4);
  auto scales_rt = round_trip(stream, host.scales, float16);
  auto biases_rt = round_trip(stream, host.biases, float16);
  HostQuantizedWeights rounded = host;
  rounded.scales = scales_rt;
  rounded.biases = biases_rt;
  auto expected = host_quantized_matmul(rounded, x_values, m, n, k, 64, 4);

  array x(x_values.begin(), Shape{m, k}, float16);
  array w_words(host.words.begin(), Shape{n, k / 8}, uint32);
  array scales(scales_rt.begin(), Shape{n, k / 64}, float16);
  array biases(biases_rt.begin(), Shape{n, k / 64}, float16);
  array out = quantized_matmul(
      x, w_words, scales, biases, true, 64, 4, "affine", stream);
  std::string error = evaluation_error(out);
  if (e.coopmat_claim && !coopmat_backed()) {
    std::cout << "[capsim q4-prefill] profile=" << g_profile
              << " -> named refusal\n";
    expect_simulated_refusal(error, "QmmPrefillCoopmatF16");
    return;
  }
  REQUIRE(error.empty());
  // f32 accumulation of f16-storage operands against an f64 host
  // reference: the shared qmm envelope (small operands, k=128).
  expect_close(readback_f32(stream, out), expected, 5e-2, 1e-3);
  std::cout << "[capsim q4-prefill] profile=" << g_profile << " -> "
            << (e.coopmat_claim ? "QmmPrefillCoopmatF16" : "QmmTileRbF16")
            << " (backed=" << coopmat_backed() << ")\n";
}

TEST_CASE("sdpa decode routes per the profile") {
  if (!compute_available() || !float16_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto e = expectations();

  constexpr int heads = 2;
  constexpr int kv_heads = 2;
  constexpr int keys = 64;
  constexpr int width = 64;
  std::mt19937 gen(13);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> q_values(heads * width);
  std::vector<float> k_values(kv_heads * keys * width);
  std::vector<float> v_values(kv_heads * keys * width);
  for (auto& v : q_values) {
    v = dist(gen);
  }
  for (auto& v : k_values) {
    v = dist(gen);
  }
  for (auto& v : v_values) {
    v = dist(gen);
  }
  array q = astype(
      array(q_values.begin(), Shape{1, heads, 1, width}, float32),
      float16,
      stream);
  array k = astype(
      array(k_values.begin(), Shape{1, kv_heads, keys, width}, float32),
      float16,
      stream);
  array v = astype(
      array(v_values.begin(), Shape{1, kv_heads, keys, width}, float32),
      float16,
      stream);

  // f64 host reference over the f16-rounded inputs.
  auto q_rt = round_trip(stream, q_values, float16);
  auto k_rt = round_trip(stream, k_values, float16);
  auto v_rt = round_trip(stream, v_values, float16);
  const double scale = 1.0 / std::sqrt(static_cast<double>(width));
  std::vector<float> expected(static_cast<size_t>(heads) * width, 0.0f);
  for (int h = 0; h < heads; ++h) {
    int kvh = h / (heads / kv_heads);
    std::vector<double> scores(keys);
    double max_score = -1e30;
    for (int t = 0; t < keys; ++t) {
      double acc = 0.0;
      for (int d = 0; d < width; ++d) {
        acc += static_cast<double>(q_rt[h * width + d]) *
            static_cast<double>(
                k_rt[((size_t)kvh * keys + t) * width + d]);
      }
      scores[t] = acc * scale;
      max_score = std::max(max_score, scores[t]);
    }
    double denom = 0.0;
    for (int t = 0; t < keys; ++t) {
      scores[t] = std::exp(scores[t] - max_score);
      denom += scores[t];
    }
    for (int d = 0; d < width; ++d) {
      double acc = 0.0;
      for (int t = 0; t < keys; ++t) {
        acc += scores[t] *
            static_cast<double>(v_rt[((size_t)kvh * keys + t) * width + d]);
      }
      expected[h * width + d] = static_cast<float>(acc / denom);
    }
  }

  setenv("MLX_OMARCHY_SDPA_DECODE_NATIVE", "1", 1);
  array output = fast::scaled_dot_product_attention(
      q, k, v, 1.0f / std::sqrt(float(width)), "", {}, std::nullopt, false,
      stream);
  std::string error = evaluation_error(output);
  if (e.sdpa_native_gate && !coopmat_backed()) {
    unsetenv("MLX_OMARCHY_SDPA_DECODE_NATIVE");
    std::cout << "[capsim sdpa] profile=" << g_profile
              << " -> named refusal\n";
    expect_simulated_refusal(error, "SdpaDecodeNativeF16");
    return;
  }
  REQUIRE(error.empty());
  // f16 storage end to end against an f64 reference: generous
  // envelope, tuned by the score f16 rounding (2^-8 relative).
  expect_close(readback_f32(stream, output), expected, 0.08, 0.02);
  unsetenv("MLX_OMARCHY_SDPA_DECODE_NATIVE");
  std::cout << "[capsim sdpa] profile=" << g_profile << " -> "
            << (e.sdpa_native_gate ? "SdpaDecodeNativeF16"
                                   : "composed f16 SDPA")
            << " (backed=" << coopmat_backed() << ")\n";
}

// NormApple: the Apple row-reduction kernels may be selected only under
// subgroup_size == 32 + subgroup ARITHMETIC, and only at the measured
// (width, rows) shapes. Under every profile the result must be the
// composed chain bit-for-bit (shared reduction order), and a simulated
// gate claim the hardware cannot back refuses loudly at dispatch — that
// is the gate refusing on a non-Apple GPU by construction.
TEST_CASE("apple norm selection follows the subgroup gate") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  const auto& c = caps();
  const auto& h = hw();
  const bool claimed_ready = c.subgroup_size == 32u &&
      (c.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
  const bool hw_ready = h.subgroup_size == 32u &&
      (h.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;

  auto lattice = [](size_t count, uint32_t seed) {
    std::vector<float> v(count);
    uint32_t s = seed;
    for (size_t i = 0; i < count; ++i) {
      s = s * 1664525u + 1013904223u;
      v[i] = static_cast<float>(static_cast<double>(s % 20000u) / 10000.0) -
          1.0f;
    }
    return v;
  };

  if (claimed_ready && !hw_ready) {
    // Non-Apple device leg: the profile claims 32+ARITHMETIC the runtime
    // hardware cannot back; the dispatch site must refuse by name.
    auto x_data = lattice(9 * 2048, 71);
    array x = astype(
        array(x_data.data(), Shape{9, 2048}, float32), bfloat16, stream);
    array w = astype(
        array(lattice(2048, 72).data(), Shape{2048}, float32),
        bfloat16,
        stream);
    bool threw = false;
    try {
      array y = fast::rms_norm(x, w, 1e-6f, stream);
      eval(y);
    } catch (const std::exception& e) {
      threw = true;
      CHECK_MESSAGE(
          strstr(e.what(), "subgroup") != nullptr,
          "refusal names the axis: ",
          e.what());
    }
    REQUIRE(threw);
    return;
  }

  // Gate closed (subgroup-size-64, no-cooperative-matrix) or open
  // (m1 profiles on backed hardware): width 128 is never Apple-selected
  // (measured shape predicate), width 2048 rows 9 is selected only when
  // the gate is open. Either way the kill switch keeps the same bits.
  for (int width : {128, 2048}) {
    auto x_data = lattice(9 * width, 73);
    auto w_data = lattice(width, 74);
    array x = astype(
        array(x_data.data(), Shape{9, width}, float32), bfloat16, stream);
    array w = astype(
        array(w_data.data(), Shape{width}, float32), bfloat16, stream);
    array on = fast::rms_norm(x, w, 1e-6f, stream);
    eval(on);
    setenv("MLX_OMARCHY_NORM_APPLE", "0", 1);
    array off = fast::rms_norm(x, w, 1e-6f, stream);
    eval(off);
    unsetenv("MLX_OMARCHY_NORM_APPLE");
    array on32 = astype(on, float32, stream);
    array off32 = astype(off, float32, stream);
    eval(on32);
    eval(off32);
    REQUIRE(on32.shape() == off32.shape());
    const float* po = on32.data<float>();
    const float* pf = off32.data<float>();
    double worst_bits = 0.0;
    double worst_val = 0.0;
    for (int r = 0; r < 9; ++r) {
      double sum = 0.0;
      for (int col = 0; col < width; ++col) {
        double xv = x_data[r * width + col];
        sum += xv * xv;
      }
      double invrs = 1.0 / std::sqrt(sum / width + 1e-6);
      for (int col = 0; col < width; ++col) {
        size_t i = size_t(r) * width + col;
        double want = x_data[i] * invrs * w_data[col];
        worst_bits = std::max(
            worst_bits,
            std::abs(static_cast<double>(po[i]) - static_cast<double>(pf[i])));
        worst_val = std::max(
            worst_val,
            std::abs(static_cast<double>(po[i]) - want));
      }
    }
    INFO("width ", width, " worst default-vs-off ", worst_bits);
    CHECK_EQ(worst_bits, 0.0);
    CHECK_MESSAGE(
        worst_val <= 5e-2,
        "width ",
        width,
        " worst vs host math ",
        worst_val);
  }
}

int main(int argc, char** argv) {
  // --- Unit checks that need no device, run before any backend init.
  {
    // Unknown profile: loud, names the value and every known profile.
    setenv("MLX_OMARCHY_CAPS_SIM", "no-such-profile", 1);
    bool threw = false;
    std::string message;
    try {
      omarchy::capsim::requested_profile();
    } catch (const std::exception& ex) {
      threw = true;
      message = ex.what();
    }
    if (!threw || message.find("no-such-profile") == std::string::npos ||
        message.find(kM1Fork) == std::string::npos ||
        message.find(kNoCoopmat) == std::string::npos) {
      std::cerr << "unit: unknown-profile refusal failed: " << message
                << "\n";
      return 1;
    }
    unsetenv("MLX_OMARCHY_CAPS_SIM");

    // Registry sanity: the five documented profiles exist.
    for (const char* name :
         {kM1Fork, kM1Stock, kSubgroup64, kSmallSmem, kNoCoopmat}) {
      if (omarchy::capsim::find(name) == nullptr) {
        std::cerr << "unit: profile missing from registry: " << name
                  << "\n";
        return 1;
      }
    }

    // apply() maps the audit axes onto a report without a device.
    omarchy::CapabilityReport hw_report;
    hw_report.device_name = "probe";
    hw_report.subgroup_size = 64;
    hw_report.subgroup_operations = VK_SUBGROUP_FEATURE_BASIC_BIT;
    hw_report.cooperative_matrix_f32_8 = false;
    hw_report.max_compute_shared_memory_size = 65536;
    for (const char* name :
         {kM1Fork, kM1Stock, kSubgroup64, kSmallSmem, kNoCoopmat}) {
      auto out = omarchy::capsim::apply(
          hw_report, *omarchy::capsim::find(name));
      if (!out.simulated || out.simulation_profile != name ||
          out.device_name != "probe") {
        std::cerr << "unit: apply() did not stamp " << name << "\n";
        return 1;
      }
    }
    auto fork = omarchy::capsim::apply(
        hw_report, *omarchy::capsim::find(kM1Fork));
    if (fork.subgroup_size != 32 || !fork.cooperative_matrix_f32_8 ||
        fork.max_compute_shared_memory_size != 65536) {
      std::cerr << "unit: m1-fork delta wrong\n";
      return 1;
    }
    auto small = omarchy::capsim::apply(
        hw_report, *omarchy::capsim::find(kSmallSmem));
    if (small.max_compute_shared_memory_size != 16384 ||
        !small.cooperative_matrix_f32_8 || small.subgroup_size != 32) {
      std::cerr << "unit: small-shared-memory delta wrong\n";
      return 1;
    }
    auto none = omarchy::capsim::apply(
        hw_report, *omarchy::capsim::find(kNoCoopmat));
    if (none.cooperative_matrix_f32_8 || none.subgroup_size != 64 ||
        (none.subgroup_operations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0) {
      std::cerr << "unit: no-cooperative-matrix delta wrong\n";
      return 1;
    }
    std::cout << "unit: profile registry + apply() deltas OK\n";
  }

  // --- Profile selection for the GPU legs.
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <profile-name>\nprofiles:";
    for (const auto* p : omarchy::capsim::profiles()) {
      std::cerr << " " << p->name;
    }
    std::cerr << "\n";
    return 2;
  }
  g_profile = argv[1];
  if (omarchy::capsim::find(g_profile) == nullptr) {
    std::cerr << "unknown profile '" << g_profile << "'\n";
    return 2;
  }
  setenv("MLX_OMARCHY_CAPS_SIM", g_profile.c_str(), 1);

  doctest::Context ctx;
  ctx.applyCommandLine(argc, argv);
  int status = ctx.run();

  unsetenv("MLX_OMARCHY_CAPS_SIM");
  return status;
}
