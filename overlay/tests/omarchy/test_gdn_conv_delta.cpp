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
#include <stdexcept>
#include <string>
#include <vector>
#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/fused_chain.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

// The C ABI the mlx-lm patch reads for the per-chip default: 0 = the stock route, 1 = conv + delta, 2 = conv + delta + gated norm.
extern "C" int mlx_omarchy_gdn_conv_delta_default(void);

using namespace mlx::core;
using mlx::core::omarchy::trace::MlxOmarchyTraceSnapshot;

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
  array conv_state, qkv, weight, a, b, a_log, dt_bias, delta_state, gate, norm_w;
};

Inputs make_inputs(uint32_t seed, float state_scale, Stream s, int hk = H, int hv = H) {
  const int kd = hk * D;
  const int c = 2 * kd + hv * D;
  Inputs in{
      bf16({1, TAPS - 1, c}, seed + 1, 1.0f, s),
      bf16({1, 1, c}, seed + 2, 1.0f, s),
      bf16({c, TAPS, 1}, seed + 3, 0.5f, s),
      bf16({1, 1, hv}, seed + 4, 1.0f, s),
      bf16({1, 1, hv}, seed + 5, 1.0f, s),
      bf16({hv}, seed + 6, 1.0f, s, 1.0f),
      bf16({hv}, seed + 7, 1.0f, s),
      pattern({1, hv, D, D}, seed + 8, state_scale, 0.0f),
      bf16({1, 1, hv, D}, seed + 9, 2.0f, s),
      bf16({D}, seed + 10, 0.5f, s, 1.0f)};
  eval({in.conv_state, in.qkv, in.weight, in.a, in.b, in.a_log, in.dt_bias, in.delta_state, in.gate, in.norm_w});
  return in;
}

const float INV = 1.0f / std::sqrt(128.0f);
const float EPS = 1e-6f;

struct Run {
  std::vector<array> out;
  uint64_t dispatches;
};

uint64_t compute_dispatches() {
  // When linked to a wheel, the inline counter in this executable is not libmlx's counter.
  MlxOmarchyTraceSnapshot snapshot{};
  REQUIRE(mlx_omarchy_trace_snapshot_abi_size() == sizeof(snapshot));
  mlx_omarchy_trace_snapshot(&snapshot);
  return snapshot.vk_compute_dispatches;
}

// The production dispatches the fused op replaces, exactly as the model calls them: conv, delta and, with
// `norm`, the gated norm (mx.fast.rms_norm_gated(out, z, weight, eps)).
Run production_pair(const Inputs& in, Stream s, bool norm = false, int hk = H, int hv = H) {
  const int kd = hk * D;
  const uint64_t before = compute_dispatches();
  auto conv = fast::gdn_conv_update(
      in.conv_state, in.qkv, in.weight, true, kd, INV * INV, INV, EPS, s);
  auto parts = split(conv[0], {kd, 2 * kd}, -1, s);
  Shape kheads{1, 1, hk, D};
  Shape vheads{1, 1, hv, D};
  auto delta = fast::gated_delta_update_raw(
      reshape(parts[0], kheads, s), reshape(parts[1], kheads, s), reshape(parts[2], vheads, s),
      in.a, in.b, in.a_log, in.dt_bias, in.delta_state, std::nullopt, s);
  array y = norm ? fast::rms_norm_gated(delta[0], in.gate, in.norm_w, EPS, s) : delta[0];
  Run r{{y, delta[1], conv[1]}, 0};
  eval(r.out);
  r.dispatches = compute_dispatches() - before;
  return r;
}

Run fused(const Inputs& in, Stream s, bool norm = false, int hk = H) {
  const uint64_t before = compute_dispatches();
  Run r{fast::gdn_conv_delta_update(
            in.conv_state, in.qkv, in.weight, in.a, in.b, in.a_log, in.dt_bias, in.delta_state,
            hk * D, INV * INV, INV, EPS, norm ? std::optional<array>(in.gate) : std::nullopt,
            norm ? std::optional<array>(in.norm_w) : std::nullopt, norm ? EPS : 0.0f, s),
        0};
  eval(r.out);
  r.dispatches = compute_dispatches() - before;
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

TEST_CASE("the fused op does not read the flag: with the variable unset it fuses wherever its geometry allows") {
  if (!have_gpu()) return;
  // The caller decides (the mlx-lm patch, from the backend's single resolver); the op itself fuses on geometry alone. With the variable UNSET the
  // fused op must therefore still be one dispatch where the unfused route is two (and three with the gated norm), with the production bytes. The
  // dispatch assertions need the untiled prefetch kernel, so they are required only where the part runs it (GDN_CONV_DELTA_REQUIRE_ENGAGED=1).
  unsetenv("MLX_OMARCHY_GDN_CONV_DELTA");
  setenv("MLX_OMARCHY_GDN_COALESCED", "0", 1);
  Stream s = gpu_stream();
  Inputs in = make_inputs(10, 0.05f, s);
  Run pair = production_pair(in, s);
  Run one = fused(in, s);
  for (int i = 0; i < 3; ++i) {
    CHECK(array_equal(one.out[i], pair.out[i], s).item<bool>());
  }
  Run seq = production_pair(in, s, true);
  Run two = fused(in, s, true);
  for (int i = 0; i < 3; ++i) {
    CHECK(array_equal(two.out[i], seq.out[i], s).item<bool>());
  }
  unsetenv("MLX_OMARCHY_GDN_COALESCED");
  if (one.dispatches == pair.dispatches && std::getenv("GDN_CONV_DELTA_REQUIRE_ENGAGED") == nullptr) {
    std::cout << "Skipping dispatch check: this device's unfused route is not the untiled prefetch kernel\n";
    return;
  }
  CHECK_MESSAGE(one.dispatches == 1, "with the flag unset the fused conv + delta op must still be one dispatch");
  CHECK_MESSAGE(two.dispatches == 1, "with the flag unset the fused conv + delta + norm op must still be one dispatch");
}

TEST_CASE("the conv + delta + gated norm step returns the production sequence's bytes") {
  if (!have_gpu()) return;
  setenv("MLX_OMARCHY_GDN_CONV_DELTA", "2", 1);
  setenv("MLX_OMARCHY_GDN_COALESCED", "0", 1);
  Stream s = gpu_stream();
  const char* names[3] = {"normed output", "recurrent state", "conv carry-out state"};
  for (uint32_t seed : {10u, 20u, 30u}) {
    for (float state_scale : {0.05f, 2.0f}) {
      Inputs in = make_inputs(seed, state_scale, s);
      Run want = production_pair(in, s, true);
      Run got = fused(in, s, true);
      REQUIRE(got.out.size() == 3);
      for (int i = 0; i < 3; ++i) {
        CHECK_MESSAGE(got.out[i].shape() == want.out[i].shape(), std::string(names[i]), ": shape differs");
        CHECK_MESSAGE(
            array_equal(got.out[i], want.out[i], s).item<bool>(), std::string(names[i]),
            " differs from the production sequence (seed ", seed, ", state scale ", state_scale, ")");
      }
    }
  }
  unsetenv("MLX_OMARCHY_GDN_COALESCED");
}

TEST_CASE("the conv + delta + gated norm step is one dispatch where the sequence is three") {
  if (!have_gpu()) return;
  setenv("MLX_OMARCHY_GDN_CONV_DELTA", "2", 1);
  Stream s = gpu_stream();
  Inputs in = make_inputs(10, 0.05f, s);
  production_pair(in, s, true);  // warm both routes so one-time setup is not counted
  fused(in, s, true);
  Run seq = production_pair(in, s, true);
  Run one = fused(in, s, true);
  std::cout << "[gdn_conv_delta] production sequence " << seq.dispatches << " dispatches, fused with norm " << one.dispatches << "\n";
  CHECK_MESSAGE(seq.dispatches == 3, "the reference sequence is expected to be three dispatches");
  if (one.dispatches == seq.dispatches && std::getenv("GDN_CONV_DELTA_REQUIRE_ENGAGED") == nullptr) {
    std::cout << "Skipping the engagement check: this device's unfused route does not use the prefetch delta kernel\n";
    return;
  }
  CHECK_MESSAGE(one.dispatches == 1, "the fused op did not collapse the sequence into one dispatch");
}

TEST_CASE("the GDN conv + delta default is the 13-inch and the 14-inch, and the ABI resolves the mode") {
  // The per-chip default: the fused conv + delta + gated norm route runs by default on the 13-inch (Apple M1, G13G) and the 14-inch (Apple M2 Max, G14C).
  CHECK(omarchy::gdn_conv_delta_norm_default_part("Apple M1 (G13G B1)"));
  CHECK(omarchy::gdn_conv_delta_norm_default_part("Apple M2 Max (G14C B1)"));
  CHECK(omarchy::gdn_conv_delta_norm_default_part("Apple M1 Max (G13C C0)"));
  CHECK_FALSE(omarchy::gdn_conv_delta_norm_default_part("Apple M6 (G18G B1)"));
  CHECK(omarchy::gdn_conv_delta_norm_default_part(""));
  // The single resolver, host-only (it takes the device name, so every part is observable on any machine): an explicit flag value wins over the
  // per-chip default, "0" and anything else read as the stock route, and UNSET follows the default: 2 on G13G and G14C, 0 elsewhere.
  const std::string m1 = "Apple M1 (G13G B1)";
  const std::string m1max = "Apple M1 Max (G13C C0)";
  const std::string m2max = "Apple M2 Max (G14C B1)";
  unsetenv("MLX_OMARCHY_GDN_CONV_DELTA");
  CHECK(omarchy::gdn_conv_delta_resolved_mode(m1) == 2);
  CHECK(omarchy::gdn_conv_delta_resolved_mode(m2max) == 2);
  CHECK(omarchy::gdn_conv_delta_resolved_mode(m1max) == 2);
  CHECK(omarchy::gdn_conv_delta_resolved_mode("") == 2);
  setenv("MLX_OMARCHY_GDN_CONV_DELTA", "", 1);  // an empty value counts as unset
  CHECK(omarchy::gdn_conv_delta_resolved_mode(m1) == 2);
  for (const std::string& part : {m1, m1max, m2max}) {
    setenv("MLX_OMARCHY_GDN_CONV_DELTA", "2", 1);
    CHECK(omarchy::gdn_conv_delta_resolved_mode(part) == 2);
    setenv("MLX_OMARCHY_GDN_CONV_DELTA", "1", 1);
    CHECK(omarchy::gdn_conv_delta_resolved_mode(part) == 1);
    setenv("MLX_OMARCHY_GDN_CONV_DELTA", "0", 1);
    CHECK(omarchy::gdn_conv_delta_resolved_mode(part) == 0);
    setenv("MLX_OMARCHY_GDN_CONV_DELTA", "3", 1);
    CHECK(omarchy::gdn_conv_delta_resolved_mode(part) == 0);
  }
  unsetenv("MLX_OMARCHY_GDN_CONV_DELTA");
  // The C ABI the mlx-lm patch reads answers exactly what the resolver answers for this device.
  if (gpu::is_available()) {
    const auto& info = gpu::device_info(0);
    auto it = info.find("device_name");
    std::string name = it != info.end() ? std::get<std::string>(it->second) : std::string();
    CHECK(mlx_omarchy_gdn_conv_delta_default() == omarchy::gdn_conv_delta_resolved_mode(name));
    setenv("MLX_OMARCHY_GDN_CONV_DELTA", "1", 1);
    CHECK(mlx_omarchy_gdn_conv_delta_default() == 1);
    unsetenv("MLX_OMARCHY_GDN_CONV_DELTA");
  }
}

TEST_CASE("the op's form follows its inputs, not the flag: norm arguments select the norm form with the flag unset or any value") {
  if (!have_gpu()) return;
  setenv("MLX_OMARCHY_GDN_COALESCED", "0", 1);
  Stream s = gpu_stream();
  Inputs in = make_inputs(10, 0.05f, s);
  Run want = production_pair(in, s, true);
  for (const char* value : {static_cast<const char*>(nullptr), "0", "1", "2", "3"}) {
    if (value == nullptr) {
      unsetenv("MLX_OMARCHY_GDN_CONV_DELTA");
    } else {
      setenv("MLX_OMARCHY_GDN_CONV_DELTA", value, 1);
    }
    Run got = fused(in, s, true);
    INFO("flag value " << (value == nullptr ? "(unset)" : value));
    for (int i = 0; i < 3; ++i) {
      CHECK(array_equal(got.out[i], want.out[i], s).item<bool>());
    }
    // Every flag value sees the same dispatches: the op never reads it.
    CHECK(got.dispatches == fused(in, s, true).dispatches);
  }
  unsetenv("MLX_OMARCHY_GDN_CONV_DELTA");
  unsetenv("MLX_OMARCHY_GDN_COALESCED");
}

TEST_CASE("the fused conv + delta + gated norm covers unequal key and value head counts") {
  if (!have_gpu()) return;
  // The 9B shape: half as many key heads as value heads. The op composes the same
  // production sequence the model runs and returns its bytes.
  unsetenv("MLX_OMARCHY_GDN_CONV_DELTA");
  Stream s = gpu_stream();
  const char* names[3] = {"delta output", "recurrent state", "conv carry-out state"};
  for (uint32_t seed : {10u, 20u, 30u}) {
    for (float state_scale : {0.05f, 2.0f}) {
      Inputs in = make_inputs(seed, state_scale, s, 8, 16);
      Run want = production_pair(in, s, true, 8, 16);
      Run got = fused(in, s, true, 8);
      REQUIRE(got.out.size() == 3);
      for (int i = 0; i < 3; ++i) {
        CHECK_MESSAGE(got.out[i].shape() == want.out[i].shape(), std::string(names[i]), ": shape differs");
        CHECK_MESSAGE(
            array_equal(got.out[i], want.out[i], s).item<bool>(), std::string(names[i]),
            " differs from the production sequence (seed ", seed, ", state scale ", state_scale, ")");
      }
    }
  }
}

TEST_CASE("the fused conv + delta + gated norm is one dispatch at unequal head counts") {
  if (!have_gpu()) return;
  unsetenv("MLX_OMARCHY_GDN_CONV_DELTA");
  Stream s = gpu_stream();
  Inputs in = make_inputs(10, 0.05f, s, 8, 16);
  production_pair(in, s, true, 8, 16);  // warm both routes so one-time setup is not counted
  fused(in, s, true, 8);
  Run seq = production_pair(in, s, true, 8, 16);
  Run one = fused(in, s, true, 8);
  std::cout << "[gdn_conv_delta_gqa] production sequence " << seq.dispatches << " dispatches, fused " << one.dispatches << "\n";
  if (one.dispatches == seq.dispatches && std::getenv("GDN_CONV_DELTA_REQUIRE_ENGAGED") == nullptr) {
    std::cout << "Skipping the engagement check: this device did not admit the GQA fused kernel\n";
    return;
  }
  CHECK_MESSAGE(one.dispatches == 1, "the fused op did not collapse the sequence into one dispatch");
}

TEST_CASE("the fused GDN op rejects invalid head ratios before dispatch") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  for (int hv : {8, 24}) {
    INFO("key heads 16, value heads " << hv);
    Inputs in = make_inputs(10, 0.05f, s, 16, hv);
    // Do not evaluate the returned graph: a regression must not run the shader.
    CHECK_THROWS_WITH_AS(
        fast::gdn_conv_delta_update(
            in.conv_state, in.qkv, in.weight, in.a, in.b, in.a_log, in.dt_bias,
            in.delta_state, KEY_DIM, INV * INV, INV, EPS, in.gate, in.norm_w, EPS, s),
        "[gdn_conv_delta_update] value heads must be a positive multiple of key heads",
        std::invalid_argument);
  }
}

TEST_CASE("the 16:32 GDN path preserves bytes with nonzero input offsets") {
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  Inputs base = make_inputs(42, 0.05f, s, 16, 32);
  const auto offset_view = [s](const array& x) {
    const int size = static_cast<int>(x.size());
    auto flat = reshape(x, {size}, s);
    auto padded = concatenate({full({8}, array(-123.0f), x.dtype(), s), flat}, 0, s);
    auto view = reshape(slice(padded, {8}, {size + 8}, s), x.shape(), s);
    view.eval();
    REQUIRE(view.offset() != 0);
    REQUIRE(view.flags().row_contiguous);
    return view;
  };
  Inputs offset = base;
  offset.conv_state = offset_view(base.conv_state);
  offset.qkv = offset_view(base.qkv);
  offset.weight = offset_view(base.weight);
  Run want = production_pair(base, s, true, 16, 32);
  Run stock = production_pair(offset, s, true, 16, 32);
  fused(offset, s, true, 16);
  Run got = fused(offset, s, true, 16);
  for (int i = 0; i < 3; ++i) {
    CHECK(array_equal(stock.out[i], want.out[i], s).item<bool>());
    CHECK(array_equal(got.out[i], want.out[i], s).item<bool>());
  }
  if (std::getenv("GDN_CONV_DELTA_REQUIRE_ENGAGED") != nullptr) {
    CHECK_MESSAGE(got.dispatches == 1, "16:32 offset case must engage the fused kernel");
  }
}

TEST_CASE("GdnConvUpdate reads offset-carrying inputs, not flat bindings") {
  // gdn_conv_decode.comp declares per-binding offsets; a nonzero-offset
  // view (split, slice) must be read through them, not from the buffer
  // head. eval_gpu keeps row-contiguous offset views unmaterialized, so
  // this case exercises the kernel's offset reads directly.
  if (!have_gpu()) return;
  Stream s = gpu_stream();
  const int PAD = 8;
  auto offset_view = [&](const array& a) {
    const int n = static_cast<int>(a.size());
    auto flat = reshape(a, {n}, s);
    auto padded =
        concatenate({full({PAD}, array(-123.0f), a.dtype(), s), flat}, 0, s);
    auto v = reshape(
        slice(padded, {PAD}, {n + PAD}, s), a.shape(), s);
    v.eval();
    REQUIRE(v.offset() != 0);
    REQUIRE(v.flags().row_contiguous);
    return v;
  };
  for (uint32_t seed : {10u, 20u}) {
    Inputs in = make_inputs(seed, 0.05f, s);
    auto want = fast::gdn_conv_update(
        in.conv_state, in.qkv, in.weight, true, KEY_DIM, INV * INV, INV, EPS, s);
    eval({want[0], want[1]});
    auto got = fast::gdn_conv_update(
        offset_view(in.conv_state), offset_view(in.qkv), offset_view(in.weight),
        true, KEY_DIM, INV * INV, INV, EPS, s);
    eval({got[0], got[1]});
    for (int i : {0, 1}) {
      CHECK_MESSAGE(
          array_equal(got[i], want[i], s).item<bool>(),
          "GdnConvUpdate output ", i, " read a flat binding instead of the offset view (seed ", seed, ")");
    }
  }
}

TEST_CASE("coalesced GDN Sg32 default policy, accuracy, dispatch count, and off-switch") {
  // Chip default: G14C and G13G equal-head only; other parts and unequal-head GQA stay off.
  CHECK(omarchy::gdn_conv_coalesced_default("Apple M2 Max (G14C B1)", 16, 16));
  CHECK_FALSE(omarchy::gdn_conv_coalesced_default("Apple M2 Max (G14C B1)", 16, 32));
  CHECK(omarchy::gdn_conv_coalesced_default("Apple M1 (G13G B1)", 16, 16));
  CHECK_FALSE(omarchy::gdn_conv_coalesced_default("Apple M1 (G13G B1)", 16, 32));
  CHECK_FALSE(omarchy::gdn_conv_coalesced_default("Apple M1 Max (G13C C0)", 16, 16));
  CHECK_FALSE(omarchy::gdn_conv_coalesced_default("Apple M6 (G18G B1)", 16, 16));
  CHECK_FALSE(omarchy::gdn_conv_coalesced_default("", 16, 16));

  if (!have_gpu()) return;
  Stream s = gpu_stream();
  for (uint32_t seed : {10u, 20u, 30u}) {
    for (float state_scale : {0.05f, 2.0f}) {
      Inputs in = make_inputs(seed, state_scale, s);
      setenv("MLX_OMARCHY_GDN_COALESCED", "0", 1);
      Run want = production_pair(in, s, true);
      Run pf = fused(in, s, true);
      for (int i = 0; i < 3; ++i) {
        CHECK(array_equal(pf.out[i], want.out[i], s).item<bool>());
      }
      setenv("MLX_OMARCHY_GDN_COALESCED", "1", 1);
      fused(in, s, true); // warm pipeline
      Run coal = fused(in, s, true);
      unsetenv("MLX_OMARCHY_GDN_COALESCED");
      REQUIRE(coal.out.size() == 3);
      // Conv carry-out is untouched by the state reduction order and must be bit-exact.
      CHECK(array_equal(coal.out[2], want.out[2], s).item<bool>());
      // Normed bf16 output and f32 recurrent state match within subgroupAdd rounding (<= 1e-5 f32, <= 1 bf16 ULP).
      auto max_diff = [&](const array& a, const array& b) {
        auto d = max(abs(astype(a, float32, s) - astype(b, float32, s)), s);
        eval({d});
        return d.item<float>();
      };
      const float hf_err = max_diff(coal.out[1], want.out[1]);
      const float y_err = max_diff(coal.out[0], want.out[0]);
      CHECK_MESSAGE(hf_err <= 2e-5f, "hf max_diff=", hf_err, " seed=", seed, " scale=", state_scale);
      CHECK_MESSAGE(y_err <= 1e-2f, "y max_diff=", y_err, " seed=", seed, " scale=", state_scale);
      if (coal.dispatches != want.dispatches) {
        CHECK(coal.dispatches == 1);
      }
    }
  }
}

TEST_CASE("coalesced GDN Sg32 f32 A_log GQA 3-in-1 fusion policy, accuracy, and lazy off-switch") {
  CHECK(omarchy::gdn_conv_coalesced_f32a_default("Apple M2 Max (G14C B1)"));
  CHECK_FALSE(omarchy::gdn_conv_coalesced_f32a_default("Apple M1 (G13G B1)"));
  CHECK_FALSE(omarchy::gdn_conv_coalesced_f32a_default("Apple M1 Max (G13C C0)"));
  CHECK_FALSE(omarchy::gdn_conv_coalesced_f32a_default("Apple M6 (G18G B1)"));
  CHECK_FALSE(omarchy::gdn_conv_coalesced_f32a_default(""));

  if (!have_gpu()) return;
  Stream s = gpu_stream();
  constexpr int HK = 16;
  constexpr int HV = 32;
  Inputs in = make_inputs(101u, 0.2f, s, HK, HV);
  in.a_log = pattern({HV}, 107u, 0.5f, 1.0f);
  eval({in.a_log});

  auto run_9b = [&](const char* coal_flag) {
    setenv("MLX_OMARCHY_GDN_COALESCED", coal_flag, 1);
    setenv("MLX_OMARCHY_GDN_DECODE_COALESCED", "1", 1);
    Run r = fused(in, s, true, HK);
    unsetenv("MLX_OMARCHY_GDN_COALESCED");
    unsetenv("MLX_OMARCHY_GDN_DECODE_COALESCED");
    return r;
  };

  Run off = run_9b("0");
  // Warm pipeline, then measure fused dispatch count and bit-exactness vs composed Sg32.
  run_9b("1");
  Run on = run_9b("1");
  CHECK(off.dispatches >= 3u);
  if (on.dispatches != off.dispatches) {
    CHECK_EQ(on.dispatches, 1u);
  } else if (const char* req = std::getenv("MLX_OMARCHY_GDN_REQUIRE_ENGAGED");
             req != nullptr && std::string(req) == "1") {
    FAIL("coalesced f32 A_log 3-in-1 kernel did not engage (dispatches=", on.dispatches, ")");
  }
  CHECK(array_equal(on.out[0], off.out[0], s).item<bool>());
  CHECK(array_equal(on.out[1], off.out[1], s).item<bool>());
  CHECK(array_equal(on.out[2], off.out[2], s).item<bool>());
}

TEST_CASE("coalesced GDN Sg32Fast default policy, accuracy, and off-switch") {
  CHECK(omarchy::gdn_conv_sg32_fast_default("Apple M2 Max (G14C B1)"));
  CHECK_FALSE(omarchy::gdn_conv_sg32_fast_default("Apple M1 (G13G B1)"));
  CHECK_FALSE(omarchy::gdn_conv_sg32_fast_default("Apple M1 Max (G13C C0)"));
  CHECK_FALSE(omarchy::gdn_conv_sg32_fast_default("Apple M4 Pro (G16S B1)"));
  CHECK_FALSE(omarchy::gdn_conv_sg32_fast_default("Apple M6 (G18G B1)"));
  CHECK_FALSE(omarchy::gdn_conv_sg32_fast_default(""));

  if (!have_gpu()) return;
  Stream s = gpu_stream();
  auto max_diff = [&](const array& a, const array& b) {
    auto d = max(abs(astype(a, float32, s) - astype(b, float32, s)), s);
    eval({d});
    return d.item<float>();
  };

  // Equal-head bf16 A_log (Qwen3.8-2B shape)
  for (uint32_t seed : {10u, 20u}) {
    Inputs in = make_inputs(seed, 0.25f, s, 16, 16);
    setenv("MLX_OMARCHY_GDN_COALESCED", "1", 1);
    setenv("MLX_OMARCHY_GDN_SG32_FAST", "0", 1);
    Run base = fused(in, s, true, 16);
    setenv("MLX_OMARCHY_GDN_SG32_FAST", "1", 1);
    fused(in, s, true, 16);  // warm pipeline
    Run fast = fused(in, s, true, 16);
    unsetenv("MLX_OMARCHY_GDN_SG32_FAST");
    unsetenv("MLX_OMARCHY_GDN_COALESCED");
    REQUIRE(fast.out.size() == 3);
    CHECK(array_equal(fast.out[2], base.out[2], s).item<bool>());
    CHECK_MESSAGE(max_diff(fast.out[1], base.out[1]) <= 2e-5f, "equal-head hf diff seed=", seed);
    CHECK_MESSAGE(max_diff(fast.out[0], base.out[0]) <= 1e-2f, "equal-head y diff seed=", seed);
    if (fast.dispatches != base.dispatches) {
      CHECK(fast.dispatches == 1u);
    }
  }

  // Unequal-head f32 A_log (Qwen3.5-9B shape)
  {
    Inputs in = make_inputs(101u, 0.2f, s, 16, 32);
    in.a_log = pattern({32}, 107u, 0.5f, 1.0f);
    eval({in.a_log});
    setenv("MLX_OMARCHY_GDN_COALESCED", "1", 1);
    setenv("MLX_OMARCHY_GDN_SG32_FAST", "0", 1);
    Run base = fused(in, s, true, 16);
    setenv("MLX_OMARCHY_GDN_SG32_FAST", "1", 1);
    fused(in, s, true, 16);  // warm pipeline
    Run fast = fused(in, s, true, 16);
    unsetenv("MLX_OMARCHY_GDN_SG32_FAST");
    unsetenv("MLX_OMARCHY_GDN_COALESCED");
    REQUIRE(fast.out.size() == 3);
    CHECK(array_equal(fast.out[2], base.out[2], s).item<bool>());
    CHECK(max_diff(fast.out[1], base.out[1]) <= 2e-5f);
    CHECK(max_diff(fast.out[0], base.out[0]) <= 1e-2f);
  }
}

TEST_CASE("GDN Sg32 in_proj_a/b fold policy, bit-exactness, dispatch drop, and replay fallback") {
  // Policy: opt-in only, word-parsed via env_flag, and gated by MLX_OMARCHY_FUSED_CHAIN.
  // env_note_eval() refreshes the per-eval env cache between direct setenv() calls.
  unsetenv("MLX_OMARCHY_FUSED_CHAIN");
  unsetenv("MLX_OMARCHY_GDN_INPROJ_BA_FOLD");
  omarchy::env_note_eval();
  CHECK_FALSE(omarchy::gdn_inproj_ba_fold_enabled());
  setenv("MLX_OMARCHY_GDN_INPROJ_BA_FOLD", "1", 1);
  omarchy::env_note_eval();
  CHECK(omarchy::gdn_inproj_ba_fold_enabled());
  setenv("MLX_OMARCHY_GDN_INPROJ_BA_FOLD", "0", 1);
  omarchy::env_note_eval();
  CHECK_FALSE(omarchy::gdn_inproj_ba_fold_enabled());
  setenv("MLX_OMARCHY_GDN_INPROJ_BA_FOLD", "off", 1);
  omarchy::env_note_eval();
  CHECK_FALSE(omarchy::gdn_inproj_ba_fold_enabled());
  setenv("MLX_OMARCHY_GDN_INPROJ_BA_FOLD", "1", 1);
  setenv("MLX_OMARCHY_FUSED_CHAIN", "0", 1);
  omarchy::env_note_eval();
  CHECK_FALSE(omarchy::gdn_inproj_ba_fold_enabled());
  unsetenv("MLX_OMARCHY_FUSED_CHAIN");
  unsetenv("MLX_OMARCHY_GDN_INPROJ_BA_FOLD");
  omarchy::env_note_eval();

  if (!have_gpu()) return;
  Stream s = gpu_stream();

  auto check_case = [&](int hk, int hv, int k, bool a_log_f32, uint32_t seed) {
    Inputs base = make_inputs(seed, 0.05f, s, hk, hv);
    if (a_log_f32) {
      base.a_log = pattern({hv}, seed + 6u, 0.5f, 1.0f);
      eval({base.a_log});
    }
    array x = bf16({1, 1, k}, seed + 100u, 0.5f, s);
    array wa_f = bf16({hv, k}, seed + 101u, 0.25f, s);
    array wb_f = bf16({hv, k}, seed + 102u, 0.25f, s);
    auto qa = quantize(wa_f, 64, 4, "affine", std::nullopt, s);
    auto qb = quantize(wb_f, 64, 4, "affine", std::nullopt, s);
    eval(std::vector<array>{x, qa[0], qa[1], qa[2], qb[0], qb[1], qb[2]});

    auto run_chain = [&](const array& xin) -> Run {
      const uint64_t before = compute_dispatches();
      array a = quantized_matmul(xin, qa[0], qa[1], qa[2], true, 64, 4, "affine", s);
      array b = quantized_matmul(xin, qb[0], qb[1], qb[2], true, 64, 4, "affine", s);
      auto out = fast::gdn_conv_delta_update(
          base.conv_state, base.qkv, base.weight, a, b, base.a_log, base.dt_bias,
          base.delta_state, hk * D, INV * INV, INV, EPS, base.gate, base.norm_w,
          EPS, s);
      eval(out);
      return Run{out, compute_dispatches() - before};
    };

    setenv("MLX_OMARCHY_GDN_COALESCED", "1", 1);
    setenv("MLX_OMARCHY_GDN_SG32_FAST", "0", 1);
    unsetenv("MLX_OMARCHY_GDN_INPROJ_BA_FOLD");
    run_chain(x);  // warm
    Run ref = run_chain(x);

    setenv("MLX_OMARCHY_GDN_INPROJ_BA_FOLD", "1", 1);
    run_chain(x);  // warm
    Run got = run_chain(x);
    for (int i = 0; i < 3; ++i) {
      CHECK_MESSAGE(
          array_equal(got.out[i], ref.out[i], s).item<bool>(),
          "BA fold mismatch on output ", i, " (hk=", hk, ", hv=", hv, ", k=", k,
          ", f32_alog=", a_log_f32, ")");
    }
    if (got.dispatches != ref.dispatches) {
      CHECK_EQ(got.dispatches, 1u);
      CHECK(ref.dispatches >= 2u);
    } else if (std::getenv("GDN_CONV_DELTA_REQUIRE_ENGAGED") != nullptr) {
      FAIL("Sg32 BA fold did not engage (got=", got.dispatches, ", ref=", ref.dispatches, ")");
    }

    // Aligned nonzero x offset (8 bf16 = 16 bytes): folds and matches bit-exact.
    auto make_offset_x = [&](int pad) {
      auto flat = reshape(x, {k}, s);
      auto padded = concatenate({full({pad}, array(-7.0f), bfloat16, s), flat}, 0, s);
      auto view = reshape(slice(padded, {pad}, {k + pad}, s), {1, 1, k}, s);
      view.eval();
      REQUIRE(view.offset() == static_cast<size_t>(pad) * view.itemsize());
      REQUIRE(view.flags().row_contiguous);
      return view;
    };
    array x_aligned = make_offset_x(8);
    Run got_aligned = run_chain(x_aligned);
    for (int i = 0; i < 3; ++i) {
      CHECK(array_equal(got_aligned.out[i], ref.out[i], s).item<bool>());
    }
    if (got.dispatches == 1u) {
      CHECK_EQ(got_aligned.dispatches, 1u);
    }

    // Unaligned x offset (2 bf16 = 4 bytes): triggers replay_gdn_inproj_ba_fold
    // transparently and still matches bit-exact (FUSED_GEMV=0 keeps the
    // baseline on solo QuantizedMatmul::eval_gpu, which densifies unaligned x).
    array x_unaligned = make_offset_x(2);
    setenv("MLX_OMARCHY_FUSED_GEMV", "0", 1);
    Run got_replay = run_chain(x_unaligned);
    unsetenv("MLX_OMARCHY_FUSED_GEMV");
    for (int i = 0; i < 3; ++i) {
      CHECK(array_equal(got_replay.out[i], ref.out[i], s).item<bool>());
    }
    CHECK(got_replay.dispatches >= 2u);

    unsetenv("MLX_OMARCHY_GDN_INPROJ_BA_FOLD");
    unsetenv("MLX_OMARCHY_GDN_SG32_FAST");
    unsetenv("MLX_OMARCHY_GDN_COALESCED");
  };

  // Qwen3.8-2B equal-head bf16 A_log (Hk=16, Hv=16, K=2048)
  check_case(16, 16, 2048, false, 303u);
  // Qwen3.5-9B GQA f32 A_log (Hk=16, Hv=32, K=4096)
  check_case(16, 32, 4096, true, 404u);
}
