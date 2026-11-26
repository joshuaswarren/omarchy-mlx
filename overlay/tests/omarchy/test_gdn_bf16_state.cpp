// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// End-to-end bf16 recurrent state for GDN decode (GatedDeltaUpdate, T=1).
// Halves state traffic (1 MB read + 1 MB write vs 2 MB each for fp32) while
// computing in float32 inside the kernel.
//
// Gates:
//   - Omitting initial_state (token 0) defaults to bf16 state for bf16 queries
//   - Providing an explicit bf16 initial_state returns a bf16 final state
//   - Every decode route keeps its own variant when the state is bf16: the
//     kernel id must be the -DSTATE_BF16 twin OF THE ROUTE'S OWN kernel, so a
//     selection that loses the composed-order native GQA key indexing (f32
//     A_log, Hk < Hv) or forces the tiled kernel over the untiled / prefetch
//     opt-outs fails these tests
//   - The bf16-state run tracks the f32-state run of the same route (out and
//     final state within 5e-2)
//   - MLX_OMARCHY_GDN_STATE_F32=1 restores the fp32 state default
//
// Route cases set MLX_OMARCHY_GDN_* env vars; MLX_OMARCHY_GDN_PF is read once
// per process, so each TEST_CASE must run in its own process (ctest, the
// doctest_discover_tests runner).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
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

constexpr int B = 1, D = 128;

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    std::cout << "Skipping: no qualifying Vulkan device.\n";
    return false;
  }
  return true;
}

void provenance(const char* tag) {
  const auto& caps = omarchy::device(0).capabilities();
  std::cout << "[provenance] " << tag << " mlx=" << mlx::core::version()
            << " device=\"" << caps.device_name << "\"\n";
}

array pattern(Shape shape, uint32_t seed, float scale, float shift, Stream s) {
  size_t n = 1;
  for (auto d : shape) {
    n *= static_cast<size_t>(d);
  }
  std::vector<float> v;
  v.reserve(n);
  uint32_t x = seed;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    v.push_back(
        shift +
        scale *
            (static_cast<float>(static_cast<double>(x % 20000u) / 10000.0) -
             1.0f));
  }
  return array(v.data(), shape, float32);
}

array bf16_arr(Shape shape, uint32_t seed, float scale, Stream s,
               float shift = 0.0f) {
  return astype(pattern(shape, seed, scale, shift, s), bfloat16, s);
}

struct Inputs {
  array q;
  array k;
  array v;
  array a;
  array b;
  array a_log;
  array dt_bias;
};

// q and k carry Hk heads; a/b/A_log/dt_bias are per value head (Hv).
Inputs make_inputs(Stream s, int hk, int hv, bool a_log_f32) {
  array a_log = pattern({hv}, 6, 0.5f, 1.0f, s);
  return Inputs{
      bf16_arr({B, 1, hk, D}, 1, 0.2f, s),
      bf16_arr({B, 1, hk, D}, 2, 0.2f, s),
      bf16_arr({B, 1, hv, D}, 3, 0.5f, s),
      bf16_arr({B, 1, hv}, 4, 0.1f, s, 0.2f),
      bf16_arr({B, 1, hv}, 5, 0.1f, s, 0.3f),
      a_log_f32 ? a_log : astype(a_log, bfloat16, s),
      bf16_arr({hv}, 7, 0.2f, s, 0.5f),
  };
}

struct EnvVar {
  const char* name;
  const char* value;
};

// The route check: run the same route twice, f32 state then bf16 state. The
// f32 pass pins the route's own kernel; the bf16 pass must be that kernel's
// -DSTATE_BF16 twin, and must track the f32 run within 5e-2.
void check_route(Stream s, int hk, int hv, bool a_log_f32,
                 const std::vector<EnvVar>& env,
                 omarchy::ComputeKernel want_base,
                 omarchy::ComputeKernel want_state) {
  auto in = make_inputs(s, hk, hv, a_log_f32);
  array h0_f32 = pattern({B, hv, D, D}, 25, 0.1f, 0.0f, s);
  array h0_bf16 = astype(h0_f32, bfloat16, s);

  std::optional<array> ref_out;
  std::optional<array> ref_state;
  for (int pass = 0; pass < 2; ++pass) {
    for (const auto& e : env) {
      setenv(e.name, e.value, 1);
    }
    auto& enc = omarchy::get_command_encoder(s);
    enc.synchronize("gdn_bf16_state_route_before");
    uint64_t before = counters().vk_compute_dispatches.load();
    auto out = fast::gated_delta_update_raw(
        in.q,
        in.k,
        in.v,
        in.a,
        in.b,
        in.a_log,
        in.dt_bias,
        pass == 0 ? h0_f32 : h0_bf16,
        std::nullopt,
        s);
    eval(out);
    enc.synchronize("gdn_bf16_state_route_after");
    uint64_t dispatches = counters().vk_compute_dispatches.load() - before;
    int64_t kernel = counters().last_dispatched_kernel.load();
    for (const auto& e : env) {
      unsetenv(e.name);
    }

    CHECK(dispatches > 0);
    CHECK_EQ(kernel,
             static_cast<int64_t>(pass == 0 ? want_base : want_state));
    if (pass == 0) {
      ref_out = out[0];
      ref_state = out[1];
      CHECK_EQ(out[1].dtype(), float32);
    } else {
      CHECK_EQ(out[1].dtype(), bfloat16);
      array d_out = abs(subtract(*ref_out, astype(out[0], float32, s), s), s);
      array d_state = abs(subtract(*ref_state, astype(out[1], float32, s), s), s);
      array m_out = mlx::core::max(d_out, std::vector<int>{0, 1, 2, 3}, false, s);
      array m_state = mlx::core::max(d_state, std::vector<int>{0, 1, 2, 3}, false, s);
      eval(m_out, m_state);
      CHECK_LT(m_out.item<float>(), 0.05f);
      CHECK_LT(m_state.item<float>(), 0.05f);
    }
  }
}

} // namespace

TEST_CASE("gdn raw decode: omitted initial_state defaults to bf16 state for bf16 queries") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_default");
  auto in = make_inputs(s, 16, 16, false);

  auto res = fast::gated_delta_update_raw(
      in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, std::nullopt, std::nullopt, s);
  eval(res);

  CHECK_EQ(res.size(), 2);
  CHECK_EQ(res[0].dtype(), bfloat16);
  CHECK_EQ(res[1].dtype(), bfloat16);
  CHECK_EQ(res[1].shape(), Shape{B, 16, D, D});
}

TEST_CASE("gdn raw decode: explicit bf16 state stays bf16 and tracks fp32 reference") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_tracking");
  auto in = make_inputs(s, 16, 16, false);

  array h0_f32 = pattern({B, 16, D, D}, 10, 0.1f, 0.0f, s);
  array h0_bf16 = astype(h0_f32, bfloat16, s);

  auto res_f32 = fast::gated_delta_update_raw(
      in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, h0_f32, std::nullopt, s);
  auto res_bf16 = fast::gated_delta_update_raw(
      in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, h0_bf16, std::nullopt, s);

  eval(res_f32);
  eval(res_bf16);

  CHECK_EQ(res_f32[1].dtype(), float32);
  CHECK_EQ(res_bf16[1].dtype(), bfloat16);

  array diff_out = abs(subtract(res_f32[0], astype(res_bf16[0], float32, s), s), s);
  array diff_st = abs(subtract(res_f32[1], astype(res_bf16[1], float32, s), s), s);
  array max_out = mlx::core::max(diff_out, std::vector<int>{0, 1, 2, 3}, false, s);
  array max_st = mlx::core::max(diff_st, std::vector<int>{0, 1, 2, 3}, false, s);
  eval(max_out, max_st);

  CHECK_LT(max_out.item<float>(), 0.05f);
  CHECK_LT(max_st.item<float>(), 0.05f);
}

TEST_CASE("gdn raw decode: MLX_OMARCHY_GDN_STATE_F32=1 restores fp32 state default") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_optout");
  auto in = make_inputs(s, 16, 16, false);

  setenv("MLX_OMARCHY_GDN_STATE_F32", "1", 1);
  auto res = fast::gated_delta_update_raw(
      in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, std::nullopt, std::nullopt, s);
  eval(res);
  unsetenv("MLX_OMARCHY_GDN_STATE_F32");

  CHECK_EQ(res[0].dtype(), bfloat16);
  CHECK_EQ(res[1].dtype(), float32);
}

// Route twins. Each case pins the route's f32 kernel and then requires the
// bf16-state run to take that same route's twin.

TEST_CASE("gdn bf16 state: tiled plain route keeps its twin") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_tiled");
  check_route(
      s,
      16,
      16,
      false,
      {{"MLX_OMARCHY_GDN_DECODE_TILE", "1"}, {"MLX_OMARCHY_GDN_PF", "0"}},
      omarchy::ComputeKernel::GatedDeltaDecodeBF16,
      omarchy::ComputeKernel::GatedDeltaDecodeStateBF16);
}

TEST_CASE("gdn bf16 state: tiled native GQA route (bf16 A_log, MLX_OMARCHY_GDN_GQA_BF16A=1) keeps its twin") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_tiled_gqa");
  check_route(
      s,
      8,
      16,
      false,
      {{"MLX_OMARCHY_GDN_DECODE_TILE", "1"}, {"MLX_OMARCHY_GDN_GQA_BF16A", "1"}},
      omarchy::ComputeKernel::GatedDeltaDecodeBF16Gqa,
      omarchy::ComputeKernel::GatedDeltaDecodeGqaStateBF16);
}

TEST_CASE("gdn bf16 state: composed route (f32 A_log, Hk < Hv) keeps the native GQA key indexing") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_tiled_composed");
  check_route(
      s,
      8,
      16,
      true,
      {{"MLX_OMARCHY_GDN_DECODE_TILE", "1"}},
      omarchy::ComputeKernel::GatedDeltaDecodeBF16Composed,
      omarchy::ComputeKernel::GatedDeltaDecodeComposedStateBF16);
}

TEST_CASE("gdn bf16 state: prefetch route keeps its twin") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_pf");
  check_route(
      s,
      16,
      16,
      false,
      {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF_WIDE", "0"}},
      omarchy::ComputeKernel::GatedDeltaDecodeBF16Pf,
      omarchy::ComputeKernel::GatedDeltaDecodePfStateBF16);
}

TEST_CASE("gdn bf16 state: wide prefetch route (MLX_OMARCHY_GDN_PF_WIDE=1) keeps its twin") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_pfw");
  check_route(
      s,
      16,
      16,
      false,
      {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF_WIDE", "1"}},
      omarchy::ComputeKernel::GatedDeltaDecodeBF16PfWide,
      omarchy::ComputeKernel::GatedDeltaDecodePfWideStateBF16);
}

TEST_CASE("gdn bf16 state: prefetch composed route (f32 A_log, equal heads) keeps its twin") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_pf_composed");
  check_route(
      s,
      16,
      16,
      true,
      {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF_WIDE", "0"}},
      omarchy::ComputeKernel::GatedDeltaDecodeBF16PfComposed,
      omarchy::ComputeKernel::GatedDeltaDecodePfComposedStateBF16);
}

TEST_CASE("gdn bf16 state: native GQA prefetch route (f32 A_log, Hk < Hv, MLX_OMARCHY_GDN_PF_GQA=1) keeps its twin") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_pf_gqa");
  check_route(
      s,
      8,
      16,
      true,
      {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF_GQA", "1"}},
      omarchy::ComputeKernel::GatedDeltaDecodeBF16PfComposedGqa,
      omarchy::ComputeKernel::GatedDeltaDecodePfComposedGqaStateBF16);
}

TEST_CASE("gdn bf16 state: untiled scalar route (MLX_OMARCHY_GDN_PF=0) keeps its twin") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_untiled");
  check_route(
      s,
      16,
      16,
      false,
      {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF", "0"}},
      omarchy::ComputeKernel::GatedDeltaDecodeBF16Untiled,
      omarchy::ComputeKernel::GatedDeltaDecodeUntiledStateBF16);
}

TEST_CASE("gdn bf16 state: untiled composed route (f32 A_log, expanded heads) keeps its twin") {
  if (!compute_available()) return;
  Stream s = gpu_stream();
  provenance("gdn_bf16_state_untiled_composed");
  check_route(
      s,
      8,
      16,
      true,
      {{"MLX_OMARCHY_GDN_DECODE_TILE", "0"}, {"MLX_OMARCHY_GDN_PF", "0"}},
      omarchy::ComputeKernel::GatedDeltaDecodeBF16UntiledComposed,
      omarchy::ComputeKernel::GatedDeltaDecodeUntiledComposedStateBF16);
}
