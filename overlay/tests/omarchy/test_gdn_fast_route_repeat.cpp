// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Qwen3.5-9B has 16 key heads and 32 value heads (Hk != Hv), and mlx-lm calls
// mx.fast.gated_delta_update with the un-repeated q and k. The omarchy
// backend expands q and k to Hv heads on the device and runs the fused
// kernels, for prefill and for decode. These cases pin that contract:
//   - the un-repeated call takes the fused path (a small, bounded dispatch
//     count, not the per-token composed fallback), and
//   - its output and state equal, bit for bit, the result of the call the
//     mlx-lm repeat patch makes (q and k expanded with mx.repeat first).
// Before the backend repeat existed, the un-repeated call took the composed
// fallback: hundreds of dispatches per layer and different numbers.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;
using mlx::core::omarchy::trace::counters;

namespace {

void skip(const char* reason) { std::cout << "Skipping: " << reason << "\n"; }

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device");
    return false;
  }
  return true;
}

// Qwen3.5-9B GDN shape (B=1, T=512 prefill):
//   Hk=16 (linear_num_key_heads), Dk=128, Hv=32 (linear_num_value_heads),
//   Dv=128, scalar g [B,T,Hv], scalar beta [B,T,Hv], f32 state [B,Hv,Dv,Dk].
// The model passes q and k with Hk heads.
struct Qwen95Shape {
  int B = 1;
  int T = 32;       // smaller T keeps the test fast; T>=64 still hits coopmat
  int Hk = 16;
  int Hv = 32;
  int Dk = 128;
  int Dv = 128;
};

std::vector<float> pattern(size_t count, uint32_t seed) {
  std::vector<float> v;
  v.reserve(count);
  uint32_t s = seed;
  for (size_t i = 0; i < count; ++i) {
    s = s * 1664525u + 1013904223u;
    v.push_back(static_cast<float>(static_cast<double>(s % 20000u) / 10000.0) - 1.0f);
  }
  return v;
}

// Conditioned inputs: unit-norm q and k rows, g in (0.3, 0.999), beta in
// (0, 1), zero state. q and k carry Hk heads. With pad > 0 they are slices
// of a longer array, so they are row-contiguous views with a buffer offset.
struct GdnInputs {
  array q, k, v, g, beta, h0;
};

GdnInputs make_conditioned(const Qwen95Shape& sh, Stream s, int pad = 0) {
  const int total_t = sh.T + pad;
  std::vector<float> q_data =
      pattern(static_cast<size_t>(sh.B) * total_t * sh.Hk * sh.Dk, 1);
  std::vector<float> k_data =
      pattern(static_cast<size_t>(sh.B) * total_t * sh.Hk * sh.Dk, 2);
  std::vector<float> v_data =
      pattern(static_cast<size_t>(sh.B) * sh.T * sh.Hv * sh.Dv, 3);
  auto l2norm_rows = [](std::vector<float>& x, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
      double sum = 0.0;
      for (int c = 0; c < cols; ++c) {
        sum += double(x[r * cols + c]) * double(x[r * cols + c]);
      }
      sum = std::sqrt(sum);
      if (sum < 1e-12) {
        continue;
      }
      for (int c = 0; c < cols; ++c) {
        x[r * cols + c] = float(double(x[r * cols + c]) / sum);
      }
    }
  };
  l2norm_rows(q_data, sh.B * total_t * sh.Hk, sh.Dk);
  l2norm_rows(k_data, sh.B * total_t * sh.Hk, sh.Dk);
  std::vector<float> g_data(static_cast<size_t>(sh.B) * sh.T * sh.Hv);
  std::vector<float> beta_data(g_data.size());
  for (size_t i = 0; i < g_data.size(); ++i) {
    g_data[i] =
        0.3f + 0.699f * (pattern(1, 1000u + uint32_t(i))[0] * 0.5f + 0.5f);
    beta_data[i] = 0.5f * (1.0f + pattern(1, 2000u + uint32_t(i))[0]);
  }
  auto bf16 = [&](const std::vector<float>& data, Shape shape) {
    return astype(array(data.data(), std::move(shape), float32), bfloat16, s);
  };
  array q = bf16(q_data, Shape{sh.B, total_t, sh.Hk, sh.Dk});
  array k = bf16(k_data, Shape{sh.B, total_t, sh.Hk, sh.Dk});
  if (pad > 0) {
    q = slice(q, {0, pad, 0, 0}, {sh.B, pad + sh.T, sh.Hk, sh.Dk}, s);
    k = slice(k, {0, pad, 0, 0}, {sh.B, pad + sh.T, sh.Hk, sh.Dk}, s);
  }
  array v = bf16(v_data, Shape{sh.B, sh.T, sh.Hv, sh.Dv});
  array g = bf16(g_data, Shape{sh.B, sh.T, sh.Hv});
  array beta = bf16(beta_data, Shape{sh.B, sh.T, sh.Hv});
  array h0 = zeros({sh.B, sh.Hv, sh.Dv, sh.Dk}, float32, s);
  return GdnInputs{q, k, v, g, beta, h0};
}

struct GdnRun {
  std::vector<array> out;
  uint64_t dispatches;
};

// One fast::gated_delta_update call, with the compute dispatches it caused.
GdnRun run_gdn(const GdnInputs& in, const array& q, const array& k, Stream s) {
  auto& enc = omarchy::get_command_encoder(s);
  enc.synchronize("gdn_repeat_before");
  uint64_t before = counters().vk_compute_dispatches.load();
  auto out = fast::gated_delta_update(q, k, in.v, in.g, in.beta, in.h0);
  out[0].eval();
  out[1].eval();
  enc.synchronize("gdn_repeat_after");
  return {out, counters().vk_compute_dispatches.load() - before};
}

} // namespace

TEST_CASE("gdn fused path: un-repeated q/k (Hk != Hv) equals the repeated route bit for bit") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  struct Case {
    int T;
    int pad;
    const char* name;
  };
  for (const Case& c : {
           Case{1, 0, "decode"},
           Case{32, 0, "prefill T=32"},
           Case{96, 0, "prefill T=96 (chunked route)"},
           Case{32, 3, "prefill T=32, q/k views with a buffer offset"},
           Case{1, 3, "decode, q/k views with a buffer offset"}}) {
    Qwen95Shape sh;
    sh.T = c.T;
    GdnInputs in = make_conditioned(sh, s, c.pad);
    // Route A: the caller expands q and k to Hv heads first (the mlx-lm
    // repeat patch). Route B: the caller passes them as they are.
    array q_exp = repeat(in.q, sh.Hv / sh.Hk, 2, s);
    array k_exp = repeat(in.k, sh.Hv / sh.Hk, 2, s);
    eval(std::vector<array>{
        in.q, in.k, in.v, in.g, in.beta, in.h0, q_exp, k_exp});
    omarchy::get_command_encoder(s).synchronize("gdn_repeat_inputs");

    GdnRun a = run_gdn(in, q_exp, k_exp, s);
    GdnRun b = run_gdn(in, in.q, in.k, s);
    std::cout << "[gdn_fast_route_repeat] " << c.name
              << ": repeated route " << a.dispatches
              << " dispatches, un-repeated route " << b.dispatches << "\n";

    // The fused path takes a small bounded number of dispatches. The
    // composed fallback is a per-token loop of small ops.
    CHECK_MESSAGE(
        a.dispatches <= 16,
        c.name, ": repeated route took ", a.dispatches, " dispatches");
    CHECK_MESSAGE(
        b.dispatches <= 16,
        c.name, ": un-repeated route took ", b.dispatches,
        " dispatches; the backend should expand q and k and run the fused"
        " kernels, not the composed fallback");
    CHECK_MESSAGE(
        array_equal(a.out[0], b.out[0]).item<bool>(),
        c.name, ": output differs between the repeated and un-repeated routes");
    CHECK_MESSAGE(
        array_equal(a.out[1], b.out[1]).item<bool>(),
        c.name, ": final state differs between the repeated and un-repeated routes");
  }
}

// Regression pin for the no-mistakes round-2 finding
// gdn-qk-c256-guard-mismatch: scripts/patch-mlx-lm-qknorm.py routes to
// the fused op when head_k_dim == 128, key_dim % 256 == 0 and
// B*C % 256 == 0, without constraining C % 256. C = 640 (key_dim 256 +
// 128 value channels) passes that route with an even batch and must
// RUN: the epilogue halves need C % 128 == 0 only. The host guard used
// to refuse this geometry with omarchy::unsupported.
TEST_CASE("gdn_conv_update qk epilogue runs the routed C % 256 == 128 geometry") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  const int B = 2;
  const int C = 640;
  const int K = 4;
  const int key_dim = 256;
  const float inv = 1.0f / std::sqrt(128.0f);

  auto state_data = pattern(static_cast<size_t>(B) * (K - 1) * C, 11);
  auto x_data = pattern(static_cast<size_t>(B) * C, 12);
  auto w_data = pattern(static_cast<size_t>(C) * K, 13);
  array state = astype(
      array(state_data.data(), Shape{B, K - 1, C}, float32), bfloat16, s);
  array x = astype(
      array(x_data.data(), Shape{B, 1, C}, float32), bfloat16, s);
  array weight = astype(
      array(w_data.data(), Shape{C, K, 1}, float32), bfloat16, s);

  std::vector<array> out =
      fast::gdn_conv_update(state, x, weight, true, key_dim, inv * inv, inv, 1e-6f, s);
  REQUIRE(out.size() == 2);
  eval(out[0]);
  eval(out[1]);

  // Composed reference: the exact ops the model fallback runs.
  array ci = concatenate({state, x}, 1, s);
  array conv = conv1d(ci, weight, 1, 0, 1, C, s);
  array act = conv * sigmoid(conv, s);
  auto parts = split(act, {key_dim, 2 * key_dim}, -1, s);
  auto norm_one = [&](const array& part, float scale) {
    auto r = reshape(part, Shape{B, 1, key_dim / 128, 128}, s);
    array n = fast::rms_norm_scaled(r, std::nullopt, scale, 1e-6f, s);
    return reshape(n, part.shape(), s);
  };
  array ref = concatenate(
      {norm_one(parts[0], inv * inv),
       norm_one(parts[1], inv),
       parts[2]},
      -1,
      s);
  array ref_state = slice(ci, Shape{0, 1, 0}, Shape{B, ci.shape(1), C}, s);
  eval(ref);
  eval(ref_state);

  // The carry-out state moves as raw bits.
  array got_state32 = astype(out[1], float32, s);
  array ref_state32 = astype(ref_state, float32, s);
  eval(got_state32);
  eval(ref_state32);
  REQUIRE(got_state32.shape() == ref_state32.shape());
  const float* gs = got_state32.data<float>();
  const float* rs = ref_state32.data<float>();
  for (size_t i = 0; i < got_state32.size(); ++i) {
    INFO("state bit mismatch at ", i, " got=", gs[i], " ref=", rs[i]);
    CHECK_EQ(gs[i], rs[i]);
  }

  // Output: the fused path reproduces the composed rounding; allow two
  // bf16 steps of relative slack for the standalone conv1d kernel's
  // tap accumulation order.
  array got32 = astype(out[0], float32, s);
  array ref32 = astype(ref, float32, s);
  eval(got32);
  eval(ref32);
  const float* g = got32.data<float>();
  const float* r = ref32.data<float>();
  double worst = 0.0;
  for (size_t i = 0; i < got32.size(); ++i) {
    double denom = std::max(
        1e-3,
        std::max(std::abs(static_cast<double>(g[i])),
                 std::abs(static_cast<double>(r[i]))));
    worst = std::max(
        worst,
        std::abs(static_cast<double>(g[i]) - static_cast<double>(r[i])) / denom);
  }
  INFO("worst relative deviation ", worst);
  CHECK_MESSAGE(
      worst <= 0.02,
      "fused vs composed worst relative deviation ",
      worst,
      " exceeds two bf16 steps");
}
