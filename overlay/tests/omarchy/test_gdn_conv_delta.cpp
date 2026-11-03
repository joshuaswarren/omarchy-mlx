// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT
// Fused GDN decode conv + gated-delta step (fast::gdn_conv_delta_update,
// MLX_OMARCHY_GDN_CONV_DELTA, default off). One dispatch must replace the two
// production dispatches it fuses (fast::gdn_conv_update with SiLU and the q/k
// norm folded, then fast::gated_delta_update_raw) and must return the same
// bytes: the delta output, the recurrent state and the conv carry-out state.
// The reference side is the production pair itself, not a re-derivation, so
// the test fails if the fused kernel's arithmetic drifts from either kernel.
// Geometry is the Qwen3.8-2B GDN layer: 16 key heads, 16 value heads, head
// dim 128, conv width 6144, 4 taps.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;
using mlx::core::omarchy::trace::counters;

namespace {
constexpr int H = 16, D = 128, KEY_DIM = H * D, C = 3 * KEY_DIM, TAPS = 4;

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

array pattern(Shape shape, uint32_t seed, float scale, float shift) {
  size_t n = 1;
  for (auto d : shape) n *= static_cast<size_t>(d);
  std::vector<float> v;
  v.reserve(n);
  uint32_t x = seed;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    v.push_back(
        shift + scale * (static_cast<float>(static_cast<double>(x % 20000u) / 10000.0) - 1.0f));
  }
  return array(v.data(), shape, float32);
}

array bf16(Shape shape, uint32_t seed, float scale, Stream s, float shift = 0.0f) {
  return astype(pattern(shape, seed, scale, shift), bfloat16, s);
}

struct Inputs {
  array conv_state, qkv, weight, a, b, a_log, dt_bias, delta_state;
};

Inputs make_inputs(uint32_t seed, float state_scale, Stream s) {
  Inputs in{
      bf16({1, TAPS - 1, C}, seed + 1, 1.0f, s),
      bf16({1, 1, C}, seed + 2, 1.0f, s),
      bf16({C, TAPS, 1}, seed + 3, 0.5f, s),
      bf16({1, 1, H}, seed + 4, 1.0f, s),
      bf16({1, 1, H}, seed + 5, 1.0f, s),
      bf16({H}, seed + 6, 1.0f, s, 1.0f),
      bf16({H}, seed + 7, 1.0f, s),
      pattern({1, H, D, D}, seed + 8, state_scale, 0.0f)};
  eval({in.conv_state, in.qkv, in.weight, in.a, in.b, in.a_log, in.dt_bias, in.delta_state});
  return in;
}

const float INV = 1.0f / std::sqrt(128.0f);
const float EPS = 1e-6f;

struct Run {
  std::vector<array> out;
  uint64_t dispatches;
};

// The two production dispatches the fused op replaces, exactly as the model calls them.
Run production_pair(const Inputs& in, Stream s) {
  uint64_t before = counters().vk_compute_dispatches.load();
  auto conv = fast::gdn_conv_update(
      in.conv_state, in.qkv, in.weight, true, KEY_DIM, INV * INV, INV, EPS, s);
  auto parts = split(conv[0], {KEY_DIM, 2 * KEY_DIM}, -1, s);
  Shape heads{1, 1, H, D};
  auto delta = fast::gated_delta_update_raw(
      reshape(parts[0], heads, s), reshape(parts[1], heads, s), reshape(parts[2], heads, s),
      in.a, in.b, in.a_log, in.dt_bias, in.delta_state, std::nullopt, s);
  Run r{{delta[0], delta[1], conv[1]}, 0};
  eval(r.out);
  r.dispatches = counters().vk_compute_dispatches.load() - before;
  return r;
}

Run fused(const Inputs& in, Stream s) {
  uint64_t before = counters().vk_compute_dispatches.load();
  Run r{fast::gdn_conv_delta_update(
            in.conv_state, in.qkv, in.weight, in.a, in.b, in.a_log, in.dt_bias, in.delta_state,
            KEY_DIM, INV * INV, INV, EPS, s),
        0};
  eval(r.out);
  r.dispatches = counters().vk_compute_dispatches.load() - before;
  return r;
}

bool have_gpu() {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device\n";
    return false;
  }
  return true;
}
} // namespace

TEST_CASE("the fused conv + delta step returns the production pair's bytes") {
  if (!have_gpu()) return;
  setenv("MLX_OMARCHY_GDN_CONV_DELTA", "1", 1);
  Stream s = gpu_stream();
  const char* names[3] = {"delta output", "recurrent state", "conv carry-out state"};
  for (uint32_t seed : {10u, 20u, 30u}) {
    for (float state_scale : {0.05f, 2.0f}) {
      Inputs in = make_inputs(seed, state_scale, s);
      Run want = production_pair(in, s);
      Run got = fused(in, s);
      REQUIRE(got.out.size() == 3);
      for (int i = 0; i < 3; ++i) {
        CHECK_MESSAGE(got.out[i].shape() == want.out[i].shape(), std::string(names[i]), ": shape differs");
        CHECK_MESSAGE(
            array_equal(got.out[i], want.out[i], s).item<bool>(), std::string(names[i]),
            " differs from the production pair (seed ", seed, ", state scale ", state_scale, ")");
      }
    }
  }
}

TEST_CASE("the fused conv + delta step is one dispatch where the pair is two") {
  if (!have_gpu()) return;
  setenv("MLX_OMARCHY_GDN_CONV_DELTA", "1", 1);
  Stream s = gpu_stream();
  Inputs in = make_inputs(10, 0.05f, s);
  production_pair(in, s);  // warm both routes so one-time setup is not counted
  fused(in, s);
  Run pair = production_pair(in, s);
  Run one = fused(in, s);
  std::cout << "[gdn_conv_delta] production pair " << pair.dispatches << " dispatches, fused " << one.dispatches << "\n";
  CHECK_MESSAGE(pair.dispatches == 2, "the reference pair is expected to be two dispatches");
  // The fused kernel only exists for the untiled prefetch delta route; where the unfused route would use
  // another kernel (a tiled-default device) the op composes the pair. GDN_CONV_DELTA_REQUIRE_ENGAGED=1
  // (the hardware tickets) turns that fallback into a failure.
  if (one.dispatches == pair.dispatches && std::getenv("GDN_CONV_DELTA_REQUIRE_ENGAGED") == nullptr) {
    std::cout << "Skipping the engagement check: this device's unfused route does not use the prefetch delta kernel\n";
    return;
  }
  CHECK_MESSAGE(one.dispatches == 1, "the fused op did not collapse the pair into one dispatch");
}

TEST_CASE("the fused conv + delta step is off unless the flag is set") {
  if (!have_gpu()) return;
  unsetenv("MLX_OMARCHY_GDN_CONV_DELTA");
  Stream s = gpu_stream();
  Inputs in = make_inputs(10, 0.05f, s);
  Run want = production_pair(in, s);
  Run got = fused(in, s);
  CHECK_MESSAGE(got.dispatches == want.dispatches, "the unset flag must run the composed pair");
  for (int i = 0; i < 3; ++i) {
    CHECK(array_equal(got.out[i], want.out[i], s).item<bool>());
  }
}
