// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// End-to-end bf16 recurrent state for GDN decode (GatedDeltaUpdate, T=1).
// Halves state traffic (1 MB read + 1 MB write vs 2 MB each for fp32) while
// computing in float32 inside the workgroup tile.
// Gates:
//   - Omitting initial_state (token 0) defaults to bf16 state for bf16 queries
//   - Providing an explicit bf16 initial_state returns a bf16 final state
//   - Output state tracks the fp32-state reference within 5e-2
//   - MLX_OMARCHY_GDN_STATE_F32=1 restores the fp32 state default

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdlib>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

constexpr int B = 1, H = 16, D = 128;

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
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

Inputs make_inputs(Stream s) {
  return Inputs{
      bf16_arr({B, 1, H, D}, 1, 0.2f, s),
      bf16_arr({B, 1, H, D}, 2, 0.2f, s),
      bf16_arr({B, 1, H, D}, 3, 0.5f, s),
      bf16_arr({B, 1, H}, 4, 0.1f, s, 0.2f),
      bf16_arr({B, 1, H}, 5, 0.1f, s, 0.3f),
      bf16_arr({H}, 6, 0.5f, s, 1.0f),
      bf16_arr({H}, 7, 0.2f, s, 0.5f),
  };
}

} // namespace

TEST_CASE("gdn raw decode: omitted initial_state defaults to bf16 state for bf16 queries") {
  if (!gpu::is_available()) return;
  Stream s = gpu_stream();
  auto in = make_inputs(s);

  auto res = fast::gated_delta_update_raw(
      in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, std::nullopt, std::nullopt, s);
  eval(res);

  CHECK_EQ(res.size(), 2);
  CHECK_EQ(res[0].dtype(), bfloat16);
  CHECK_EQ(res[1].dtype(), bfloat16);
  CHECK_EQ(res[1].shape(), Shape{B, H, D, D});
}

TEST_CASE("gdn raw decode: explicit bf16 state stays bf16 and tracks fp32 reference") {
  if (!gpu::is_available()) return;
  Stream s = gpu_stream();
  auto in = make_inputs(s);

  array h0_f32 = pattern({B, H, D, D}, 10, 0.1f, 0.0f, s);
  array h0_bf16 = astype(h0_f32, bfloat16, s);

  auto res_f32 = fast::gated_delta_update_raw(
      in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, h0_f32, std::nullopt, s);
  auto res_bf16 = fast::gated_delta_update_raw(
      in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, h0_bf16, std::nullopt, s);

  eval(res_f32);
  eval(res_bf16);

  CHECK_EQ(res_f32[1].dtype(), float32);
  CHECK_EQ(res_bf16[1].dtype(), bfloat16);

  // Numerical tracking check against fp32 reference
  array diff_out = abs(subtract(res_f32[0], astype(res_bf16[0], float32, s), s));
  array diff_st = abs(subtract(res_f32[1], astype(res_bf16[1], float32, s), s));
  eval(diff_out, diff_st);

  float max_diff_out = max(diff_out, false, s).item<float>();
  float max_diff_st = max(diff_st, false, s).item<float>();

  CHECK_LT(max_diff_out, 0.05f);
  CHECK_LT(max_diff_st, 0.05f);
}

TEST_CASE("gdn raw decode: MLX_OMARCHY_GDN_STATE_F32=1 restores fp32 state default") {
  if (!gpu::is_available()) return;
  Stream s = gpu_stream();
  auto in = make_inputs(s);

  setenv("MLX_OMARCHY_GDN_STATE_F32", "1", 1);
  auto res = fast::gated_delta_update_raw(
      in.q, in.k, in.v, in.a, in.b, in.a_log, in.dt_bias, std::nullopt, std::nullopt, s);
  eval(res);
  unsetenv("MLX_OMARCHY_GDN_STATE_F32");

  CHECK_EQ(res[0].dtype(), bfloat16);
  CHECK_EQ(res[1].dtype(), float32);
}
