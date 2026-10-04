// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// GDN legacy-part policy guard. On a G13 legacy part (device name G13
// but not G13C) every GDN dispatch that is fused on other parts must stay
// fused: Hk == Hv decode through both fast.cpp entries (raw gates and
// precomputed gates) and Hk == Hv prefill. The only legacy-part policy is
// the Hk < Hv q/k expansion route, which the mlx-lm patcher owns
// (scripts/patch-mlx-lm-gdn-raw-repeat.py); GatedDeltaUpdate::use_fallback
// must carry no chip term. A chip clause there once sent every GDN
// dispatch on G13 parts to the composed per-token chain (9B 91.7 -> 24.5
// tok/s at pf512).
//
// main() pins MLX_OMARCHY_CAPS_SIM=m1-g13-legacy before the backend
// initializes, so the legacy kernel selections run on any host device.
// Each case compares the dispatch count of the Hk == Hv call against an
// Hk < Hv call of the same size, which always takes the composed chain.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;
using mlx::core::omarchy::trace::counters;

namespace {

constexpr const char* kProfile = "m1-g13-legacy";

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

std::vector<float> pattern(size_t count, uint32_t seed, float scale) {
  std::vector<float> v;
  v.reserve(count);
  uint32_t s = seed;
  for (size_t i = 0; i < count; ++i) {
    s = s * 1664525u + 1013904223u;
    v.push_back(
        scale *
        (static_cast<float>(static_cast<double>(s % 20000u) / 10000.0) -
         1.0f));
  }
  return v;
}

array bf16(Shape shape, uint32_t seed, float scale, Stream s) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
  }
  auto data = pattern(n, seed, scale);
  return astype(array(data.data(), shape, float32), bfloat16, s);
}

// Dispatches issued evaluating `run`'s outputs, inputs already resident.
uint64_t dispatches(Stream s, const std::vector<array>& inputs,
                    const std::function<std::vector<array>()>& run) {
  for (auto a : inputs) {
    a.eval();
  }
  auto& enc = omarchy::get_command_encoder(s);
  enc.synchronize("gdn_legacy_inputs");
  uint64_t before = counters().vk_compute_dispatches.load();
  auto outs = run();
  for (auto& o : outs) {
    o.eval();
  }
  enc.synchronize("gdn_legacy_outputs");
  return counters().vk_compute_dispatches.load() - before;
}

struct Pair {
  uint64_t fused;
  uint64_t composed;
};

void check_fused(const char* what, Pair p, uint64_t fused_cap) {
  std::cout << "[gdn_legacy_policy] " << what << ": Hk==Hv " << p.fused
            << " dispatches, Hk<Hv (composed) " << p.composed << "\n";
  CHECK_MESSAGE(p.fused <= fused_cap, what, ": Hk==Hv took ", p.fused,
                " dispatches on a simulated G13 legacy part; the fused "
                "kernel needs <= ", fused_cap,
                ". GatedDeltaUpdate::use_fallback sent it to the composed "
                "chain.");
  CHECK_MESSAGE(p.composed > 4 * p.fused, what,
                ": the composed reference (", p.composed,
                ") is not far above the Hk==Hv count (", p.fused,
                "), so this case cannot tell fused from composed.");
}

// Raw-gate decode (T = 1): the mx.fast.gated_delta_update_raw entry.
Pair raw_decode(Stream s, int hk_composed) {
  const int H = 16, D = 128;
  auto run_shape = [&](int hk) {
    array q = bf16({1, 1, hk, D}, 1, 0.1f, s);
    array k = bf16({1, 1, hk, D}, 2, 0.1f, s);
    array v = bf16({1, 1, H, D}, 3, 1.0f, s);
    array a = bf16({1, 1, H}, 4, 1.0f, s);
    array b = bf16({1, 1, H}, 5, 1.0f, s);
    array a_log = bf16({H}, 6, 1.0f, s);
    array dt = bf16({H}, 7, 1.0f, s);
    array h0 = zeros({1, H, D, D}, float32, s);
    return dispatches(s, {q, k, v, a, b, a_log, dt, h0}, [&] {
      return fast::gated_delta_update_raw(q, k, v, a, b, a_log, dt, h0);
    });
  };
  return {run_shape(H), run_shape(hk_composed)};
}

// Precomputed-gate GDN through mx.fast.gated_delta_update: decode (T = 1)
// and prefill (T > 1).
Pair gated(Stream s, int T, int hk_composed) {
  const int H = 16, D = 128;
  auto run_shape = [&](int hk) {
    array q = bf16({1, T, hk, D}, 11, 0.1f, s);
    array k = bf16({1, T, hk, D}, 12, 0.1f, s);
    array v = bf16({1, T, H, D}, 13, 1.0f, s);
    array g = astype(add(bf16({1, T, H}, 14, 0.05f, s), array(0.9f), s),
                     bfloat16, s);
    array beta = astype(add(bf16({1, T, H}, 15, 0.25f, s), array(0.5f), s),
                        bfloat16, s);
    array h0 = zeros({1, H, D, D}, float32, s);
    return dispatches(s, {q, k, v, g, beta, h0}, [&] {
      return fast::gated_delta_update(q, k, v, g, beta, h0);
    });
  };
  return {run_shape(H), run_shape(hk_composed)};
}

bool simulated_legacy() {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device\n";
    return false;
  }
  const auto& info = gpu::device_info(0);
  auto it = info.find("device_name");
  REQUIRE(it != info.end());
  const auto& name = std::get<std::string>(it->second);
  REQUIRE_MESSAGE(name.find("G13") != std::string::npos &&
                      name.find("G13C") == std::string::npos,
                  "capability simulation did not apply the legacy device "
                  "name; got '", name, "'");
  return true;
}

} // namespace

TEST_CASE("legacy part: Hk==Hv raw-gate decode stays fused") {
  if (!simulated_legacy()) return;
  check_fused("raw decode T=1", raw_decode(gpu_stream(), 8), 4);
}

TEST_CASE("legacy part: Hk==Hv gated decode stays fused") {
  if (!simulated_legacy()) return;
  check_fused("gated decode T=1", gated(gpu_stream(), 1, 8), 4);
}

TEST_CASE("legacy part: Hk==Hv prefill stays fused") {
  if (!simulated_legacy()) return;
  check_fused("prefill T=32", gated(gpu_stream(), 32, 8), 16);
}

int main(int argc, char** argv) {
  setenv("MLX_OMARCHY_CAPS_SIM", kProfile, 1);
  doctest::Context context(argc, argv);
  return context.run();
}
