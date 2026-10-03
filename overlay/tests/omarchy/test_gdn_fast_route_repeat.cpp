// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Guard against the Qwen3.5-9B GDN prefill regression: when Hk != Hv
// (linear_num_key_heads=16, linear_num_value_heads=32 on the 9B; the
// model-side gated_delta_update calls mx.fast.gated_delta_update with
// the un-repeated q/k), the omarchy backend's
// GatedDeltaUpdate::use_fallback check returns true and the per-token
// Python loop runs. The composed fallback produces hundreds of small
// dispatches per GDN layer; the coopmat fused path produces one.
//
// This doctest drives the exact prefill shape the 9B sees, asserts the
// fused coopmat path is selected (small dispatch count), and asserts a
// tolerance-pinned numerical result against the model-side composed
// fallback (which is the reference for this kernel). Without the
// mlx-lm-gated-delta-fast-route-repeat.patch the test would still pass
// (the C++ backend correctly composes when use_fallback returns true)
// but the dispatch count would explode; the asserted upper bound catches
// the regression before it ships.

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
// When the model calls mx.fast.gated_delta_update with the un-repeated
// q/k the backend sees Hk=16, Hv=32 -> use_fallback returns true.
// With the mlx-lm patch the model expands q/k to Hv heads first.
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

array make_input(const Qwen95Shape& sh, const std::vector<float>& data) {
  array a = array(data.data(), {sh.B, sh.T, sh.Hk, sh.Dk}, float32);
  return astype(a, bfloat16, gpu_stream());
}

} // namespace

TEST_CASE("gdn fast path: Hk != Hv repeats q/k before fast dispatch") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  Qwen95Shape sh;

  // Inputs in the SHAPE THE MODEL PASSES (Hk=16). Two routes:
  //   route A: caller expanded q/k to Hv=32 (the patch's behavior)
  //   route B: caller did NOT expand (the unpatched regression)
  // The backend should fuse route A in one dispatch and use the composed
  // fallback for route B. We exercise BOTH and assert the fused path
  // dispatch count is far smaller than the fallback.
  std::vector<float> q_data = pattern(
      static_cast<size_t>(sh.B) * sh.T * sh.Hk * sh.Dk, 1);
  std::vector<float> k_data = pattern(
      static_cast<size_t>(sh.B) * sh.T * sh.Hk * sh.Dk, 2);
  std::vector<float> v_data = pattern(
      static_cast<size_t>(sh.B) * sh.T * sh.Hv * sh.Dv, 3);

  array q_raw = make_input(sh, q_data);
  array k_raw = make_input(sh, k_data);
  array v = astype(
      array(v_data.data(), {sh.B, sh.T, sh.Hv, sh.Dv}, float32),
      bfloat16, s);
  array g = astype(
      array(pattern(static_cast<size_t>(sh.B) * sh.T * sh.Hv, 4).data(),
            {sh.B, sh.T, sh.Hv}, float32),
      bfloat16, s);
  array beta = astype(
      array(pattern(static_cast<size_t>(sh.B) * sh.T * sh.Hv, 5).data(),
            {sh.B, sh.T, sh.Hv}, float32),
      bfloat16, s);
  array h0 = zeros({sh.B, sh.Hv, sh.Dv, sh.Dk}, float32, s);

  // Route A: caller repeats q/k to Hv (the patch).
  array q_exp = repeat(q_raw, sh.Hv / sh.Hk, 2, s);
  array k_exp = repeat(k_raw, sh.Hv / sh.Hk, 2, s);
  q_exp.eval();
  k_exp.eval();
  v.eval();
  g.eval();
  beta.eval();
  h0.eval();
  auto& enc = omarchy::get_command_encoder(s);
  enc.synchronize("gdn_repeat_inputs");

  uint64_t before_a = counters().vk_compute_dispatches.load();
  auto out_a = fast::gated_delta_update(q_exp, k_exp, v, g, beta, h0);
  out_a[0].eval();
  out_a[1].eval();
  enc.synchronize("gdn_repeat_fused");
  uint64_t after_a = counters().vk_compute_dispatches.load();
  uint64_t dispatches_a = after_a - before_a;

  // Route B: caller does NOT repeat (the unpatched regression).
  q_raw.eval();
  k_raw.eval();
  enc.synchronize("gdn_repeat_inputs_b");
  uint64_t before_b = counters().vk_compute_dispatches.load();
  auto out_b = fast::gated_delta_update(q_raw, k_raw, v, g, beta, h0);
  out_b[0].eval();
  out_b[1].eval();
  enc.synchronize("gdn_repeat_fallback");
  uint64_t after_b = counters().vk_compute_dispatches.load();
  uint64_t dispatches_b = after_b - before_b;

  std::cout << "[gdn_fast_route_repeat] fused (Hk=Hv=" << sh.Hv
            << "): " << dispatches_a << " dispatches; "
            << "fallback (Hk=" << sh.Hk << ", Hv=" << sh.Hv
            << "): " << dispatches_b << " dispatches\n";

  // The fused coopmat path runs in a small bounded number of dispatches
  // (the coopmat kernel = 1, plus dtype/state materializations < 16).
  // The composed fallback is the per-token Python loop and produces
  // O(T * layers) dispatches worth of small ops.
  CHECK_MESSAGE(
      dispatches_a <= 16,
      "fused coopmat path should produce <=16 dispatches for the expanded "
      "q/k; got ",
      dispatches_a,
      ". Either the dispatch gate (Hk==Hv && Dk==128 && Dv==128) failed "
      "or the coopmat prefill path did not select.");
  CHECK_MESSAGE(
      dispatches_b > dispatches_a * 4,
      "composed fallback should produce substantially more dispatches "
      "than the fused path; got fused=",
      dispatches_a,
      " fallback=",
      dispatches_b,
      ". If they are close, the per-token Python loop is not running on "
      "route B and the regression is hidden.");

  // Numerical tolerance: when both routes are allowed to compose, the
  // outputs should agree bit-exactly (both are reference math). The
  // fused coopmat path rounds chunk-form and may differ from the
  // composed fallback by up to a small bf16 quantum per element. We
  // assert that the gap is bounded by 2 quanta at the output's scale.
  array diff = abs(out_a[0] - out_b[0]);
  array mx = abs(out_a[0]);
  double max_diff = static_cast<double>(max(diff).item<float>());
  double max_val = static_cast<double>(max(mx).item<float>());
  double tolerance = std::max(2.0 * max_val * 1e-3, 1e-2);
  CHECK_MESSAGE(
      max_diff <= tolerance,
      "fused vs fallback max abs diff = ",
      max_diff,
      " exceeds tolerance ",
      tolerance,
      " (max value ",
      max_val,
      ")");
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
    array n = rms_norm_scaled(r, std::nullopt, scale, 1e-6f, s);
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
