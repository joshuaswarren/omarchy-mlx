// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <bit>
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

using namespace mlx::core;

namespace {

constexpr int kHk = 16;
constexpr int kD = 128;

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (gpu::is_available()) return true;
  std::cout << "Skipping: no qualifying Vulkan device\n";
  return false;
}

std::vector<float> values(size_t n, uint32_t seed, float scale) {
  std::vector<float> out(n);
  for (float& value : out) {
    seed = seed * 1664525u + 1013904223u;
    value = (static_cast<float>(seed % 20001u) / 10000.0f - 1.0f) * scale;
  }
  return out;
}

void round_bf16(std::vector<float>& data) {
  for (float& value : data) {
    uint32_t bits = std::bit_cast<uint32_t>(value);
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    value = std::bit_cast<float>(bits);
  }
}

std::vector<float> materialize_f32(const array& value, Stream stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize("gdn_maskless_readback");
  const float* data = copy.data<float>();
  return {data, data + copy.size()};
}

struct Reference {
  std::vector<double> y;
  std::vector<double> state;
};

Reference reference(
    const std::vector<float>& q,
    const std::vector<float>& k,
    const std::vector<float>& v,
    const std::vector<float>& g,
    const std::vector<float>& beta,
    int T,
    int Hv) {
  Reference result{
      std::vector<double>(static_cast<size_t>(T) * Hv * kD),
      std::vector<double>(static_cast<size_t>(Hv) * kD * kD, 0.0)};
  std::vector<double> next(result.state.size());
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < Hv; ++h) {
      const size_t row = (static_cast<size_t>(t) * Hv + h) * kD;
      const size_t state_row = static_cast<size_t>(h) * kD * kD;
      const double gate = g[static_cast<size_t>(t) * Hv + h];
      const double rate = beta[static_cast<size_t>(t) * Hv + h];
      for (int dv = 0; dv < kD; ++dv) {
        const size_t state_offset = state_row + static_cast<size_t>(dv) * kD;
        double kv = 0.0;
        for (int dk = 0; dk < kD; ++dk) {
          kv += result.state[state_offset + dk] * gate * k[row + dk];
        }
        const double delta = (v[row + dv] - kv) * rate;
        double output = 0.0;
        for (int dk = 0; dk < kD; ++dk) {
          const size_t index = state_offset + dk;
          const double updated = result.state[index] * gate + delta * k[row + dk];
          next[index] = updated;
          output += updated * q[row + dk];
        }
        result.y[row + dv] = output;
      }
    }
    result.state.swap(next);
  }
  return result;
}

void check_close(
    const std::vector<float>& got,
    const std::vector<double>& want,
    double tolerance,
    const std::string& label) {
  REQUIRE_EQ(got.size(), want.size());
  double max_error = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    max_error = std::max(max_error, std::abs(static_cast<double>(got[i]) - want[i]));
  }
  CHECK_MESSAGE(max_error <= tolerance, label, " max_abs=", max_error);
}

void check_case(int T, int rep, Stream stream) {
  const int Hv = kHk * rep;
  const size_t token_heads = static_cast<size_t>(T) * Hv;
  const size_t activation_size = token_heads * kD;
  auto q_data = values(activation_size, 0x10203040u + T + rep, 0.25f);
  auto k_data = values(activation_size, 0x50607080u + T + rep, 0.25f);
  auto v_data = values(activation_size, 0x90a0b0c0u + T + rep, 0.25f);
  auto g_data = values(token_heads, 0xd0e0f000u + T + rep, 0.07f);
  auto beta_data = values(token_heads, 0x12345678u + T + rep, 0.2f);
  for (float& gate : g_data) gate = 0.92f + std::abs(gate);
  for (float& rate : beta_data) rate = 0.3f + rate;
  round_bf16(q_data);
  round_bf16(k_data);
  round_bf16(v_data);
  round_bf16(beta_data);
  const Reference ref = reference(q_data, k_data, v_data, g_data, beta_data, T, Hv);

  array q = astype(array(q_data.begin(), Shape{1, T, Hv, kD}, float32), bfloat16, stream);
  array k = astype(array(k_data.begin(), Shape{1, T, Hv, kD}, float32), bfloat16, stream);
  array v = astype(array(v_data.begin(), Shape{1, T, Hv, kD}, float32), bfloat16, stream);
  array g = array(g_data.begin(), Shape{1, T, Hv}, float32);
  array beta = astype(array(beta_data.begin(), Shape{1, T, Hv}, float32), bfloat16, stream);
  array h0 = zeros({1, Hv, kD, kD}, float32, stream);
  array mask = ones({1, T}, bool_, stream);
  q.eval(); k.eval(); v.eval(); g.eval(); beta.eval(); h0.eval(); mask.eval();
  omarchy::get_command_encoder(stream).synchronize("gdn_maskless_inputs");

  auto maskless = fast::gated_delta_update(q, k, v, g, beta, h0, std::nullopt, stream);
  auto masked = fast::gated_delta_update(q, k, v, g, beta, h0, mask, stream);
  const std::string label = "GDN T=" + std::to_string(T) + " rep=" + std::to_string(rep);
  check_close(materialize_f32(maskless[0], stream), ref.y, 0.02, label + " maskless y vs fp64");
  check_close(materialize_f32(maskless[1], stream), ref.state, 2e-4, label + " maskless state vs fp64");
  check_close(materialize_f32(masked[0], stream), ref.y, 0.02, label + " masked y vs fp64");
  check_close(materialize_f32(masked[1], stream), ref.state, 2e-4, label + " masked state vs fp64");
}

} // namespace

TEST_CASE("GDN maskless prefill preserves fp64 final state across route boundary") {
  if (!compute_available()) return;
  Stream stream = gpu_stream();
  for (int rep : {1, 2, 3}) {
    for (int T : {63, 64, 65, 96, 352}) {
      CAPTURE(T);
      CAPTURE(rep);
      check_case(T, rep, stream);
    }
  }
}
