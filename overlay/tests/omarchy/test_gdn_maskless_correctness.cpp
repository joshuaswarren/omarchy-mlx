// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
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
    int Hv,
    const std::vector<float>& h0_data = {}) {
  std::vector<double> initial(
      static_cast<size_t>(Hv) * kD * kD, 0.0);
  if (!h0_data.empty()) {
    for (size_t i = 0; i < initial.size(); ++i) {
      initial[i] = h0_data[i];
    }
  }
  Reference result{
      std::vector<double>(static_cast<size_t>(T) * Hv * kD),
      std::move(initial)};
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


void check_case_nonzero_state(int T, int rep, Stream stream) {
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
  // Production starts from zeros, but the recurrence is state-dependent:
  // a non-zero initial state exercises the state path both routes carry.
  auto h0_data = values(static_cast<size_t>(Hv) * kD * kD, 0xbeef1234u + T + rep, 0.1f);
  const Reference ref = reference(q_data, k_data, v_data, g_data, beta_data, T, Hv, h0_data);

  array q = astype(array(q_data.begin(), Shape{1, T, Hv, kD}, float32), bfloat16, stream);
  array k = astype(array(k_data.begin(), Shape{1, T, Hv, kD}, float32), bfloat16, stream);
  array v = astype(array(v_data.begin(), Shape{1, T, Hv, kD}, float32), bfloat16, stream);
  array g = array(g_data.begin(), Shape{1, T, Hv}, float32);
  array beta = astype(array(beta_data.begin(), Shape{1, T, Hv}, float32), bfloat16, stream);
  array h0 = array(h0_data.begin(), Shape{1, Hv, kD, kD}, float32);
  q.eval(); k.eval(); v.eval(); g.eval(); beta.eval(); h0.eval();
  omarchy::get_command_encoder(stream).synchronize("gdn_maskless_inputs_nz");

  auto out = fast::gated_delta_update(q, k, v, g, beta, h0, std::nullopt, stream);
  const std::string label =
      "GDN nz-state T=" + std::to_string(T) + " rep=" + std::to_string(rep);
  check_close(materialize_f32(out[0], stream), ref.y, 0.02, label + " y vs fp64");
  check_close(materialize_f32(out[1], stream), ref.state, 2e-4, label + " state vs fp64");
}

} // namespace

TEST_CASE("GDN prefill carries a non-zero initial state across the hoist boundary") {
  if (!compute_available()) return;
  Stream stream = gpu_stream();
  for (int rep : {1, 2}) {
    for (int T : {512, 519}) {
      check_case_nonzero_state(T, rep, stream);
    }
  }
}


TEST_CASE("GDN maskless prefill preserves fp64 final state across route boundary") {
  if (!compute_available()) return;
  Stream stream = gpu_stream();
  for (int rep : {1, 2, 3}) {
    // 512/519 cross the kkt/qkt-hoist boundary (default ON for T >= 512;
    // MLX_OMARCHY_GDN_HOIST=0 restores the single-dispatch route).
    for (int T : {63, 64, 65, 96, 352, 512, 519}) {
      CAPTURE(T);
      CAPTURE(rep);
      check_case(T, rep, stream);
    }
  }
}

// Real captured Qwen3.5-27B prefill operands (two-head subset: the NaN head
// 29 plus a control head), checked in under fixtures/gdn_coopmat. These are
// the exact operands of the 2026-10-03 defect captures: layer0 exercises
// large-v / small-g ranges, layer12 carries g down to 5.9e-11 (below the
// old 1e-6 gate floor) in head 29. The fused coopmat route must match the
// fp64 per-token reference at the same tolerance the scan route achieves.
#ifndef GDN_FIXTURE_DIR
#define GDN_FIXTURE_DIR "fixtures/gdn_coopmat"
#endif

namespace {

struct FixtureTensors {
  std::vector<float> q, k, v, g, beta;  // token-major [T * Hv * kD] / [T * Hv]
};

std::vector<float> load_bf16_bits(
    const std::string& path, const std::vector<int>& shape) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  size_t count = 1;
  for (int d : shape) count *= d;
  std::vector<uint16_t> bits(count);
  in.read(reinterpret_cast<char*>(bits.data()), count * sizeof(uint16_t));
  REQUIRE_EQ(static_cast<size_t>(in.gcount()), count * sizeof(uint16_t));
  std::vector<float> out(count);
  for (size_t i = 0; i < count; ++i) {
    uint32_t wide = static_cast<uint32_t>(bits[i]) << 16;
    out[i] = std::bit_cast<float>(wide);
  }
  return out;
}

std::vector<float> load_f32(
    const std::string& path, const std::vector<int>& shape) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  size_t count = 1;
  for (int d : shape) count *= d;
  std::vector<float> out(count);
  in.read(reinterpret_cast<char*>(out.data()), count * sizeof(float));
  REQUIRE_EQ(static_cast<size_t>(in.gcount()), count * sizeof(float));
  return out;
}

// [H, T, D] fixture files -> token-major [T * H * D] vectors.
FixtureTensors load_fixture(const std::string& tag) {
  const std::string dir = GDN_FIXTURE_DIR;
  std::vector<int> qkv_shape = {2, 351, kD};
  std::vector<int> gate_shape = {2, 351};
  FixtureTensors fx;
  auto heads_first = [](std::vector<float> data, int heads, int tokens, int dim) {
    std::vector<float> out(data.size());
    for (int h = 0; h < heads; ++h)
      for (int t = 0; t < tokens; ++t)
        for (int d = 0; d < dim; ++d)
          out[(static_cast<size_t>(t) * heads + h) * dim + d] =
              data[(static_cast<size_t>(h) * tokens + t) * dim + d];
    return out;
  };
  fx.q = heads_first(load_bf16_bits(dir + "/" + tag + ".q.bf16", qkv_shape), 2, 351, kD);
  fx.k = heads_first(load_bf16_bits(dir + "/" + tag + ".k.bf16", qkv_shape), 2, 351, kD);
  fx.v = heads_first(load_bf16_bits(dir + "/" + tag + ".v.bf16", qkv_shape), 2, 351, kD);
  fx.beta = heads_first(load_bf16_bits(dir + "/" + tag + ".beta.bf16", gate_shape), 2, 351, 1);
  fx.g = heads_first(load_f32(dir + "/" + tag + ".g.f32", gate_shape), 2, 351, 1);
  return fx;
}

void check_fixture(const std::string& tag, double state_tol, Stream stream) {
  const FixtureTensors fx = load_fixture(tag);
  const int T = 351;
  const int Hv = 2;
  const Reference ref = reference(fx.q, fx.k, fx.v, fx.g, fx.beta, T, Hv);

  auto mk = [&](const std::vector<float>& data, Shape shape, bool bf) {
    array a = array(data.begin(), shape, float32);
    return bf ? astype(a, bfloat16, stream) : a;
  };
  array q = mk(fx.q, Shape{1, T, Hv, kD}, true);
  array k = mk(fx.k, Shape{1, T, Hv, kD}, true);
  array v = mk(fx.v, Shape{1, T, Hv, kD}, true);
  array g = mk(fx.g, Shape{1, T, Hv}, false);
  array beta = mk(fx.beta, Shape{1, T, Hv}, true);
  array h0 = zeros({1, Hv, kD, kD}, float32, stream);
  array mask = ones({1, T}, bool_, stream);
  q.eval(); k.eval(); v.eval(); g.eval(); beta.eval(); h0.eval(); mask.eval();
  omarchy::get_command_encoder(stream).synchronize("gdn_fixture_inputs");

  const std::string label = "GDN fixture " + tag;
  for (bool with_mask : {false, true}) {
    auto out = with_mask
        ? fast::gated_delta_update(q, k, v, g, beta, h0, mask, stream)
        : fast::gated_delta_update(q, k, v, g, beta, h0, std::nullopt, stream);
    const std::string arm = with_mask ? " masked" : " maskless";
    std::vector<float> y = materialize_f32(out[0], stream);
    std::vector<float> st = materialize_f32(out[1], stream);
    double nan_count = 0;
    double state_err = 0;
    for (size_t i = 0; i < st.size(); ++i) {
      if (!std::isfinite(st[i]) || !std::isfinite(y[i])) nan_count += 1;
    }
    CHECK_MESSAGE(nan_count == 0, label << arm << " NaN/inf count=" << nan_count);
    for (size_t i = 0; i < st.size(); ++i) {
      state_err = std::max(state_err, std::abs(static_cast<double>(st[i]) - ref.state[i]));
    }
    CHECK_MESSAGE(state_err <= state_tol, label << arm << " state max_abs=" << state_err << " tol=" << state_tol);
    // y is bf16 output on O(1..10^2) values: 8 bf16 quanta of the reference.
    double y_err = 0;
    double y_bad = 0;
    for (size_t i = 0; i < y.size(); ++i) {
      double quantum = std::max(std::abs(ref.y[i]) * 0x1p-8, 0x1p-10);
      double err = std::abs(static_cast<double>(y[i]) - ref.y[i]);
      y_err = std::max(y_err, err);
      if (err > 8 * quantum) y_bad += 1;
    }
    CHECK_MESSAGE(y_bad == 0, label << arm << " y over-quanta count=" << y_bad << " max_abs=" << y_err);
  }
}

} // namespace

TEST_CASE("GDN coopmat prefill matches fp64 on captured 27B operands") {
  if (!compute_available()) return;
  Stream stream = gpu_stream();
  check_fixture("layer0", 5e-5, stream);
  check_fixture("layer12", 1e-5, stream);
}

namespace {

// Route selection reads MLX_OMARCHY_GDN_RECUR32 per call (primitives.cpp),
// so the env set here redirects only the maskless T >= 64 arms of this
// case to the recur32 kernel - even when the whole battery runs as one
// process. The RAII guard restores the default routes for later cases.
struct EnvGuard {
  EnvGuard(const char* name, const char* value) : name_(name) {
    setenv(name, value, 1);
  }
  ~EnvGuard() {
    unsetenv(name_);
  }
  const char* name_;
};

} // namespace

TEST_CASE("GDN recur32 per-token route matches fp64 reference") {
  if (!compute_available()) return;
  EnvGuard guard("MLX_OMARCHY_GDN_RECUR32", "1");
  Stream stream = gpu_stream();
  // Same sweep as the maskless case above: 63/65 straddle the T >= 64
  // route boundary (63 keeps the exact scan), 512/519 cover the hoist
  // Ts the recur32 route now intercepts, and rep varies Hv.
  for (int rep : {1, 2, 3}) {
    for (int T : {63, 64, 65, 96, 352, 512, 519}) {
      CAPTURE(T);
      CAPTURE(rep);
      check_case(T, rep, stream);
    }
  }
  for (int rep : {1, 2}) {
    for (int T : {512, 519}) {
      check_case_nonzero_state(T, rep, stream);
    }
  }
}

TEST_CASE("GDN recur32 mode 2 covers short prefill TTFT shapes") {
  if (!compute_available()) return;
  EnvGuard guard("MLX_OMARCHY_GDN_RECUR32", "2");
  Stream stream = gpu_stream();
  // Route minimum is T >= 2 in this mode (T=1 is the decode kernel's);
  // every T here takes the recur32 route maskless, while the masked arm
  // of check_case keeps the exact scan route. Non-zero h0 covers
  // mid-conversation prefill, the state-carrying TTFT case.
  for (int rep : {1, 2, 3}) {
    for (int T : {2, 3, 5, 11, 16, 31, 32, 33, 63}) {
      CAPTURE(T);
      CAPTURE(rep);
      check_case(T, rep, stream);
    }
  }
  for (int rep : {1, 2}) {
    for (int T : {11, 63}) {
      check_case_nonzero_state(T, rep, stream);
    }
  }
}
