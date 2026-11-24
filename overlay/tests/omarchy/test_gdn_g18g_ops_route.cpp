// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// G18G GDN ops-route guard. On an Apple M6 (G18G) part every GDN dispatch
// must compose: the fused update and raw-decode kernels miscompute there
// (PAR-35, 2026-11-24: per-layer teacher-forced dumps diverge from layer 1
// prefill on; the ops path is bit-identical to the reference). The part
// keeps the ops route until the fused kernels match.
//
// main() pins MLX_OMARCHY_CAPS_SIM=m6-g18g before the backend initializes,
// so the routing runs on any host device. Each case compares the dispatch
// count of the head-dim 128 call, fused on qualified parts, against the
// head-dim 64 call of the same shape, which always composes
// (GatedDeltaUpdate::use_fallback refuses Dk != 128 || Dv != 128). Composed
// counts track each other; a fused D=128 call lands far below.

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
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;
using mlx::core::omarchy::trace::counters;

namespace {

constexpr const char* kProfile = "m6-g18g";

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
  enc.synchronize("gdn_g18g_inputs");
  uint64_t before = counters().vk_compute_dispatches.load();
  auto outs = run();
  for (auto& o : outs) {
    o.eval();
  }
  enc.synchronize("gdn_g18g_outputs");
  return counters().vk_compute_dispatches.load() - before;
}

struct Pair {
  uint64_t d128;
  uint64_t d64;
};

// On G18G both head dims compose, so the two counts stay within a narrow
// band of each other and both sit far above the fused cap.
void check_composed(const char* what, Pair p, uint64_t fused_cap) {
  std::cout << "[gdn_g18g_ops_route] " << what << ": D=128 " << p.d128
            << " dispatches, D=64 " << p.d64 << "\n";
  CHECK_MESSAGE(p.d128 > fused_cap, what, ": D=128 took ", p.d128,
                " dispatches on a simulated G18G part, at most ", fused_cap,
                " for the fused kernel; the ops route it should take "
                "composes far more.");
  CHECK_MESSAGE(p.d128 * 4 >= p.d64, what,
                ": D=128 (", p.d128, ") ran far below D=64 (", p.d64,
                ") on a simulated G18G part; both should compose, so this "
                "case cannot tell the ops route from fused.");
  CHECK_MESSAGE(p.d64 * 4 >= p.d128, what,
                ": D=64 (", p.d64, ") ran far below D=128 (", p.d128,
                ") on a simulated G18G part; the composed reference should "
                "not undercut the routed call.");
}

// Raw-gate decode (T = 1): the mx.fast.gated_delta_update_raw entry.
Pair raw_decode(Stream s) {
  const int H = 16;
  auto run_shape = [&](int D) {
    array q = bf16({1, 1, H, D}, 1, 0.1f, s);
    array k = bf16({1, 1, H, D}, 2, 0.1f, s);
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
  return {run_shape(128), run_shape(64)};
}

// Precomputed-gate GDN through mx.fast.gated_delta_update: decode (T = 1)
// and prefill (T > 1).
Pair gated(Stream s, int T) {
  const int H = 16;
  auto run_shape = [&](int D) {
    array q = bf16({1, T, H, D}, 11, 0.1f, s);
    array k = bf16({1, T, H, D}, 12, 0.1f, s);
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
  return {run_shape(128), run_shape(64)};
}

bool simulated_g18g() {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device\n";
    return false;
  }
  const auto& info = gpu::device_info(0);
  auto it = info.find("device_name");
  REQUIRE(it != info.end());
  const auto& name = std::get<std::string>(it->second);
  const bool g18g = name.find("G18G") != std::string::npos;
  REQUIRE_MESSAGE(g18g,
                  "capability simulation did not apply the G18G device "
                  "name; got '", name, "'");
  return true;
}

} // namespace

TEST_CASE("G18G: Hk==Hv raw-gate decode composes") {
  if (!simulated_g18g()) return;
  check_composed("raw decode T=1", raw_decode(gpu_stream()), 4);
}

TEST_CASE("G18G: Hk==Hv gated decode composes") {
  if (!simulated_g18g()) return;
  check_composed("gated decode T=1", gated(gpu_stream(), 1), 4);
}

TEST_CASE("G18G: Hk==Hv prefill composes") {
  if (!simulated_g18g()) return;
  check_composed("prefill T=32", gated(gpu_stream(), 32), 16);
}

// The route's pure gate, over parts and env values: G18G composes unless
// MLX_OMARCHY_GDN_G18G_FUSED=1, and no env value moves a non-G18G part.
TEST_CASE("gdn_g18g_composed_default: part x env matrix") {
  using omarchy::gdn_g18g_composed_default;
  const char* kG18 = "Apple M6 (G18G B1)";
  const char* kG13 = "Apple M1 (G13G B1)";
  const char* kG16 = "Apple M4 Pro (G16S)";
  const char* kVar = "MLX_OMARCHY_GDN_G18G_FUSED";

  unsetenv(kVar);
  CHECK(gdn_g18g_composed_default(kG18));
  CHECK(!gdn_g18g_composed_default(kG13));
  CHECK(!gdn_g18g_composed_default(kG16));

  setenv(kVar, "0", 1);
  CHECK(gdn_g18g_composed_default(kG18));
  CHECK(!gdn_g18g_composed_default(kG13));
  CHECK(!gdn_g18g_composed_default(kG16));

  setenv(kVar, "", 1);
  CHECK(gdn_g18g_composed_default(kG18));
  CHECK(!gdn_g18g_composed_default(kG13));

  setenv(kVar, "1", 1);
  CHECK(!gdn_g18g_composed_default(kG18));
  CHECK(!gdn_g18g_composed_default(kG13));
  CHECK(!gdn_g18g_composed_default(kG16));

  unsetenv(kVar);
}

TEST_CASE("G18G: GatedDeltaUpdateVJP::use_fallback composes") {
  if (!simulated_g18g()) return;
  Stream s = gpu_stream();
  CHECK(fast::GatedDeltaUpdateVJP::use_fallback(16, 128, 16, 128, s));
}

int main(int argc, char** argv) {
  // An inherited measurement env must not flip the route under test.
  unsetenv("MLX_OMARCHY_GDN_G18G_FUSED");
  setenv("MLX_OMARCHY_CAPS_SIM", kProfile, 1);
  doctest::Context context(argc, argv);
  return context.run();
}
