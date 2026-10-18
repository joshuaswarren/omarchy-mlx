// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Raw-gates decode on the Qwen3.8-27B contract: bf16 A_log, 16 key heads and 48
// value heads (Hk != Hv), Dk = Dv = 128. By default the backend expands q and k to
// Hv heads with two device copies per call and runs the default decode kernel.
// MLX_OMARCHY_GDN_GQA_BF16A=1 skips the two copies and runs
// GatedDeltaDecodeBF16Gqa, the default-order kernel with the key-head index
// of the composed kernel (value head j reads key head j / (Hv / Hk)).
// The gate is exact equality, bit for bit, of the output and the final state:
//   - flag on, un-repeated q and k == flag off, un-repeated q and k (the backend expansion);
//   - flag on, un-repeated q and k == the caller expanded q and k with mx.repeat first;
// for B = 1, q/k views with a buffer offset, and batched decode (B = 3), and the call
// takes exactly two compute dispatches fewer. Calls outside the route (flag unset or
// "0", an f32 A_log, equal head counts) do not change.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/version.h"

using namespace mlx::core;
using mlx::core::omarchy::trace::counters;

namespace {

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device (set "
                 "MLX_OMARCHY_ALLOW_NON_APPLE=1 on a development machine).\n";
    return false;
  }
  return true;
}

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

// Provenance beside every measurement (AGENTS.md test rules): the loaded libmlx version stamp (the
// wheel's source commit) and the device that ran the test.
void provenance(const char* tag) {
  const auto& caps = omarchy::device(0).capabilities();
  std::cout << "[provenance] " << tag << " mlx=" << mlx::core::version() << " device=\"" << caps.device_name
            << "\" driver=\"" << caps.driver_info << "\" subgroup_size=" << caps.subgroup_size << "\n";
}

struct Shape27 {
  int B = 1;
  int Hk = 16;
  int Hv = 48;
  int Dk = 128;
  int Dv = 128;
};

struct Inputs {
  array q, k, v, a, b, a_log, dt_bias, h0;
};

// q and k carry Hk heads. With pad > 0 they are row-contiguous slices of a longer
// array (a buffer offset), as the conv output slices are in the model.
Inputs make_inputs(const Shape27& sh, bool a_log_f32, Stream s, int pad = 0) {
  auto bf16 = [&](std::vector<float> data, Shape shape, float scale) {
    for (float& x : data) x *= scale;
    return astype(array(data.data(), std::move(shape), float32), bfloat16, s);
  };
  const int total = 1 + pad;
  array q = bf16(pattern(size_t(sh.B) * total * sh.Hk * sh.Dk, 1), Shape{sh.B, total, sh.Hk, sh.Dk}, 0.1f);
  array k = bf16(pattern(size_t(sh.B) * total * sh.Hk * sh.Dk, 2), Shape{sh.B, total, sh.Hk, sh.Dk}, 0.1f);
  if (pad > 0) {
    q = slice(q, {0, pad, 0, 0}, {sh.B, pad + 1, sh.Hk, sh.Dk}, s);
    k = slice(k, {0, pad, 0, 0}, {sh.B, pad + 1, sh.Hk, sh.Dk}, s);
  }
  array v = bf16(pattern(size_t(sh.B) * sh.Hv * sh.Dv, 3), Shape{sh.B, 1, sh.Hv, sh.Dv}, 1.0f);
  array a = bf16(pattern(size_t(sh.B) * sh.Hv, 21), Shape{sh.B, 1, sh.Hv}, 1.0f);
  array b = bf16(pattern(size_t(sh.B) * sh.Hv, 22), Shape{sh.B, 1, sh.Hv}, 1.0f);
  std::vector<float> alog = pattern(size_t(sh.Hv), 23);
  for (float& x : alog) x *= 0.5f;
  array a_log = astype(array(alog.data(), Shape{sh.Hv}, float32), a_log_f32 ? float32 : bfloat16, s);
  array dt_bias = bf16(pattern(size_t(sh.Hv), 24), Shape{sh.Hv}, 0.1f);
  std::vector<float> h0 = pattern(size_t(sh.B) * sh.Hv * sh.Dv * sh.Dk, 25);
  array h0a = array(h0.data(), Shape{sh.B, sh.Hv, sh.Dv, sh.Dk}, float32);
  return Inputs{q, k, v, a, b, a_log, dt_bias, h0a};
}

struct Run {
  std::vector<array> out;
  uint64_t dispatches;
};

Run run_route(const Inputs& in, const array& q, const array& k, const char* flag, Stream s) {
  if (flag == nullptr) {
    unsetenv("MLX_OMARCHY_GDN_GQA_BF16A");
  } else {
    setenv("MLX_OMARCHY_GDN_GQA_BF16A", flag, 1);
  }
  // The bf16 A_log native route is part of the tiled decode, so the tile is forced on: a legacy G13 part (the 13-inch M1)
  // defaults to the untiled kernel, where the flag does nothing and the dispatch counts below would not differ.
  setenv("MLX_OMARCHY_GDN_DECODE_TILE", "1", 1);
  auto& enc = omarchy::get_command_encoder(s);
  enc.synchronize("gdn_gqa_bf16a_before");
  uint64_t before = counters().vk_compute_dispatches.load();
  auto out = fast::gated_delta_update_raw(q, k, in.v, in.a, in.b, in.a_log, in.dt_bias, in.h0, std::nullopt, s);
  out[0].eval();
  out[1].eval();
  enc.synchronize("gdn_gqa_bf16a_after");
  uint64_t n = counters().vk_compute_dispatches.load() - before;
  unsetenv("MLX_OMARCHY_GDN_GQA_BF16A");
  unsetenv("MLX_OMARCHY_GDN_DECODE_TILE");
  return Run{out, n};
}

bool same(const Run& x, const Run& y, Stream s) {
  return array_equal(x.out[0], y.out[0], s).item<bool>() && array_equal(x.out[1], y.out[1], s).item<bool>();
}

// A run under a set of environment variables (nullptr = unset); every variable named is unset again afterwards.
struct EnvVar {
  const char* name;
  const char* value;
};

struct EnvRun {
  Run run;
  int64_t kernel;
};

EnvRun run_env(const Inputs& in, const array& q, const array& k, const std::vector<EnvVar>& env, Stream s) {
  for (const EnvVar& e : env) {
    if (e.value == nullptr) {
      unsetenv(e.name);
    } else {
      setenv(e.name, e.value, 1);
    }
  }
  auto& enc = omarchy::get_command_encoder(s);
  enc.synchronize("gdn_pf_gqa_before");
  uint64_t before = counters().vk_compute_dispatches.load();
  auto out = fast::gated_delta_update_raw(q, k, in.v, in.a, in.b, in.a_log, in.dt_bias, in.h0, std::nullopt, s);
  out[0].eval();
  out[1].eval();
  enc.synchronize("gdn_pf_gqa_after");
  uint64_t n = counters().vk_compute_dispatches.load() - before;
  const int64_t kernel = counters().last_dispatched_kernel.load();
  for (const EnvVar& e : env) {
    unsetenv(e.name);
  }
  return EnvRun{Run{out, n}, kernel};
}

constexpr int64_t kind(omarchy::ComputeKernel k) {
  return static_cast<int64_t>(k);
}

} // namespace

TEST_CASE("gdn raw decode bf16 A_log: native GQA equals the expanded and the repeated routes bit for bit") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_gqa_bf16a");
  struct Case {
    int B;
    int pad;
    const char* name;
  };
  for (const Case& c : {Case{1, 0, "B=1"}, Case{1, 3, "B=1, q/k views with a buffer offset"}, Case{3, 0, "B=3 batched decode"},
                        Case{3, 2, "B=3, q/k views with a buffer offset"}}) {
    Shape27 sh;
    sh.B = c.B;
    Inputs in = make_inputs(sh, /*a_log_f32=*/false, s, c.pad);
    array q_rep = repeat(in.q, sh.Hv / sh.Hk, 2, s);
    array k_rep = repeat(in.k, sh.Hv / sh.Hk, 2, s);
    eval(std::vector<array>{in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, in.h0, q_rep, k_rep});
    omarchy::get_command_encoder(s).synchronize("gdn_gqa_bf16a_inputs");
    Run expanded = run_route(in, in.q, in.k, nullptr, s);  // flag off: the backend expands q and k
    Run repeated = run_route(in, q_rep, k_rep, nullptr, s);  // the caller expanded q and k
    Run native = run_route(in, in.q, in.k, "1", s);          // flag on: no expansion copies
    std::cout << "[gdn_gqa_bf16a] " << c.name << ": expanded route " << expanded.dispatches << " dispatches, repeated "
              << repeated.dispatches << ", native " << native.dispatches << "\n";
    CHECK_MESSAGE(same(native, expanded, s), c.name, ": native output or state differs from the expanded route");
    CHECK_MESSAGE(same(native, repeated, s), c.name, ": native output or state differs from the caller-repeated route");
    CHECK_MESSAGE(same(expanded, repeated, s), c.name, ": the two reference routes differ from each other");
    CHECK_MESSAGE(native.dispatches + 2 == expanded.dispatches, c.name, ": the native route took ", native.dispatches,
                  " dispatches against ", expanded.dispatches, " expanded (expected two fewer)");
    // Batched q/k views with a buffer offset are not contiguous, so every route first makes dense copies of them (an existing step, the
    // same in all three routes); the count against the caller-repeated route is checked where no such copy is needed.
    if (!(c.B > 1 && c.pad > 0)) {
      CHECK_MESSAGE(native.dispatches == repeated.dispatches, c.name, ": native ", native.dispatches, " against the repeated route ",
                    repeated.dispatches);
    }
  }
}

TEST_CASE("gdn raw decode: calls outside the bf16 A_log native route do not change") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_gqa_bf16a_outside");
  Shape27 sh;
  // Flag "0" and an unset flag are the same route.
  {
    Inputs in = make_inputs(sh, false, s);
    eval(std::vector<array>{in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, in.h0});
    omarchy::get_command_encoder(s).synchronize("gdn_gqa_bf16a_inputs2");
    Run off = run_route(in, in.q, in.k, nullptr, s);
    Run zero = run_route(in, in.q, in.k, "0", s);
    CHECK(same(off, zero, s));
    CHECK(off.dispatches == zero.dispatches);
  }
  // An f32 A_log takes the existing composed native route whether or not the flag is set.
  {
    Inputs in = make_inputs(sh, true, s);
    eval(std::vector<array>{in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, in.h0});
    omarchy::get_command_encoder(s).synchronize("gdn_gqa_bf16a_inputs3");
    Run off = run_route(in, in.q, in.k, nullptr, s);
    Run on = run_route(in, in.q, in.k, "1", s);
    CHECK(same(off, on, s));
    CHECK(off.dispatches == on.dispatches);
  }
  // Equal head counts (no GQA) with the flag on: unchanged.
  {
    Shape27 eq;
    eq.Hk = eq.Hv;
    Inputs in = make_inputs(eq, false, s);
    eval(std::vector<array>{in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, in.h0});
    omarchy::get_command_encoder(s).synchronize("gdn_gqa_bf16a_inputs4");
    Run off = run_route(in, in.q, in.k, nullptr, s);
    Run on = run_route(in, in.q, in.k, "1", s);
    CHECK(same(off, on, s));
    CHECK(off.dispatches == on.dispatches);
  }
}

// The untiled prefetch decode with native GQA (MLX_OMARCHY_GDN_PF_GQA=1, default off), on the Qwen3.5-9B contract: f32 A_log,
// 16 key heads and 32 value heads. It applies only where the tile is off (MLX_OMARCHY_GDN_DECODE_TILE=0 here, a legacy G13
// part by default). Gate: exact equality, bit for bit, of the output and the final state with the backend expansion and with
// the caller-repeated q and k, exactly two dispatches fewer, and the kernel named by its enum.
TEST_CASE("gdn raw decode f32 A_log, tile off: native GQA through the prefetch kernel equals the expanded and the repeated routes") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_pf_gqa");
  struct Case {
    int B;
    int pad;
    const char* name;
  };
  for (const Case& c : {Case{1, 0, "B=1"}, Case{1, 3, "B=1, q/k views with a buffer offset"}, Case{3, 0, "B=3 batched decode"},
                        Case{3, 2, "B=3, q/k views with a buffer offset"}}) {
    Shape27 sh;
    sh.B = c.B;
    sh.Hv = 32;
    Inputs in = make_inputs(sh, /*a_log_f32=*/true, s, c.pad);
    array q_rep = repeat(in.q, sh.Hv / sh.Hk, 2, s);
    array k_rep = repeat(in.k, sh.Hv / sh.Hk, 2, s);
    eval(std::vector<array>{in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, in.h0, q_rep, k_rep});
    omarchy::get_command_encoder(s).synchronize("gdn_pf_gqa_inputs");
    const std::vector<EnvVar> off{{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF_GQA", nullptr}};
    const std::vector<EnvVar> on{{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF_GQA", "1"}};
    EnvRun expanded = run_env(in, in.q, in.k, off, s);
    EnvRun repeated = run_env(in, q_rep, k_rep, off, s);
    EnvRun native = run_env(in, in.q, in.k, on, s);
    std::cout << "[gdn_pf_gqa] " << c.name << ": expanded " << expanded.run.dispatches << " dispatches, repeated "
              << repeated.run.dispatches << ", native " << native.run.dispatches << "\n";
    CHECK_MESSAGE(same(native.run, expanded.run, s), c.name, ": native output or state differs from the expanded route");
    CHECK_MESSAGE(same(native.run, repeated.run, s), c.name, ": native output or state differs from the caller-repeated route");
    CHECK_MESSAGE(same(expanded.run, repeated.run, s), c.name, ": the two reference routes differ from each other");
    CHECK_MESSAGE(native.run.dispatches + 2 == expanded.run.dispatches, c.name, ": the native route took ", native.run.dispatches,
                  " dispatches against ", expanded.run.dispatches, " expanded (expected two fewer)");
    if (!(c.B > 1 && c.pad > 0)) {
      CHECK_MESSAGE(native.run.dispatches == repeated.run.dispatches, c.name, ": native ", native.run.dispatches,
                    " against the repeated route ", repeated.run.dispatches);
    }
    CHECK_EQ(expanded.kernel, kind(omarchy::ComputeKernel::GatedDeltaDecodeBF16PfComposed));
    CHECK_EQ(repeated.kernel, kind(omarchy::ComputeKernel::GatedDeltaDecodeBF16PfComposed));
    CHECK_EQ(native.kernel, kind(omarchy::ComputeKernel::GatedDeltaDecodeBF16PfComposedGqa));
  }
}

TEST_CASE("gdn raw decode: calls outside the prefetch native GQA route do not change") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_pf_gqa_outside");
  Shape27 sh;
  sh.Hv = 32;
  Inputs in32 = make_inputs(sh, true, s);
  eval(std::vector<array>{in32.q, in32.k, in32.v, in32.a, in32.b, in32.a_log, in32.dt_bias, in32.h0});
  omarchy::get_command_encoder(s).synchronize("gdn_pf_gqa_inputs2");
  const EnvVar unset_flag{"MLX_OMARCHY_GDN_PF_GQA", nullptr};
  // A flag of "0" and an unset flag are the same route.
  {
    EnvRun off = run_env(in32, in32.q, in32.k, {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, unset_flag}, s);
    EnvRun zero = run_env(in32, in32.q, in32.k, {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF_GQA", "0"}}, s);
    CHECK(same(off.run, zero.run, s));
    CHECK(off.run.dispatches == zero.run.dispatches);
    CHECK_EQ(off.kernel, zero.kernel);
  }
  // With the tile on the tiled route keeps its own native GQA whether or not the flag is set.
  {
    EnvRun off = run_env(in32, in32.q, in32.k, {{"MLX_OMARCHY_GDN_DECODE_TILE", "1"}, unset_flag}, s);
    EnvRun on = run_env(in32, in32.q, in32.k, {{"MLX_OMARCHY_GDN_DECODE_TILE", "1"}, {"MLX_OMARCHY_GDN_PF_GQA", "1"}}, s);
    CHECK(same(off.run, on.run, s));
    CHECK(off.run.dispatches == on.run.dispatches);
    CHECK_EQ(off.kernel, kind(omarchy::ComputeKernel::GatedDeltaDecodeBF16Composed));
    CHECK_EQ(on.kernel, kind(omarchy::ComputeKernel::GatedDeltaDecodeBF16Composed));
  }
  // A bf16 A_log (the Qwen3.8 route) does not take it.
  {
    Inputs bf = make_inputs(sh, false, s);
    eval(std::vector<array>{bf.q, bf.k, bf.v, bf.a, bf.b, bf.a_log, bf.dt_bias, bf.h0});
    omarchy::get_command_encoder(s).synchronize("gdn_pf_gqa_inputs3");
    EnvRun off = run_env(bf, bf.q, bf.k, {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, unset_flag}, s);
    EnvRun on = run_env(bf, bf.q, bf.k, {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF_GQA", "1"}}, s);
    CHECK(same(off.run, on.run, s));
    CHECK(off.run.dispatches == on.run.dispatches);
    CHECK_EQ(on.kernel, kind(omarchy::ComputeKernel::GatedDeltaDecodeBF16Pf));
  }
  // Equal head counts (no GQA) with the flag on: unchanged.
  {
    Shape27 eq;
    eq.Hk = eq.Hv;
    Inputs in = make_inputs(eq, true, s);
    eval(std::vector<array>{in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, in.h0});
    omarchy::get_command_encoder(s).synchronize("gdn_pf_gqa_inputs4");
    EnvRun off = run_env(in, in.q, in.k, {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, unset_flag}, s);
    EnvRun on = run_env(in, in.q, in.k, {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF_GQA", "1"}}, s);
    CHECK(same(off.run, on.run, s));
    CHECK(off.run.dispatches == on.run.dispatches);
    CHECK_EQ(on.kernel, kind(omarchy::ComputeKernel::GatedDeltaDecodeBF16PfComposed));
  }
  // The default is unchanged: with no variable set the kernel is the tiled composed one, except on a legacy G13 part
  // (G13 and not G13C), whose default is the untiled prefetch kernel; the flag is off.
  {
    const std::string& device = omarchy::device(0).capabilities().device_name;
    const bool legacy = device.find("G13") != std::string::npos && device.find("G13C") == std::string::npos;
    EnvRun def = run_env(in32, in32.q, in32.k, {{"MLX_OMARCHY_GDN_DECODE_TILE", nullptr}, unset_flag}, s);
    CHECK_EQ(def.kernel, kind(legacy ? omarchy::ComputeKernel::GatedDeltaDecodeBF16PfComposed
                                     : omarchy::ComputeKernel::GatedDeltaDecodeBF16Composed));
  }
}
