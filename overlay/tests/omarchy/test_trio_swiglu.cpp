// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Trio SwiGLU (TRIO_SWIGLU) for float16 AND bfloat16, eager decode
// shape: gate * sigmoid(gate) * up with the intermediates still
// reachable, so eager fusion materializes them and dispatches the trio
// swiglu kernel. The bf16 body rounds every intermediate with the
// swiglu.comp bf16 RNE round trip instead of packHalf2x16; the stored
// bits must match the standalone SwigluBF16 dispatch exactly. Firing
// is provable here: the trio kernel binds 7 buffers against the
// standalone kernel's 5, so the descriptor-write counter moves +2 per
// fused dispatch in the TRIO=1 arm.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/compile.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"

using namespace mlx::core;

namespace {

void skip(const char* reason) {
  std::cout << "Skipping: " << reason << "\n";
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

bool compute_available() {
  if (!gpu::is_available()) {
    skip("no qualifying Vulkan device (set MLX_OMARCHY_ALLOW_NON_APPLE=1).");
    return false;
  }
  return true;
}

void sync_stream(const Stream& stream) {
  mlx::core::omarchy::get_command_encoder(stream).synchronize();
}

void set_trio(const char* value) {
  setenv("MLX_OMARCHY_FUSED_TRIO", value, 1);
}

void set_fusion(const char* value) {
  setenv("MLX_OMARCHY_FUSED_CHAIN", value, 1);
}

void expect_bits_equal(const array& a, const array& b) {
  REQUIRE_EQ(a.size(), b.size());
  const uint16_t* pa = a.data<uint16_t>();
  const uint16_t* pb = b.data<uint16_t>();
  for (size_t i = 0; i < a.size(); ++i) {
    INFO("bit mismatch at ", i, " trio=0x", std::hex, pa[i], " solo=0x", pb[i]);
    CHECK_EQ(pa[i], pb[i]);
  }
}

// Runs one arm: compiles the swiglu-shaped function and evaluates it on
// the same inputs, returning the output with dispatch and
// descriptor-write deltas. The compiled tape dispatches ONE chain
// kernel; the trio kernel binds 7 buffers against the standalone
// kernel's 5 (+2 descriptor writes).
std::pair<array, std::pair<uint64_t, uint64_t>> run_arm(
    const array& gate,
    const array& up,
    const Stream& stream) {
  auto fn = [&](const std::vector<array>& in) {
    return std::vector<array>{in[0] * sigmoid(in[0]) * in[1]};
  };
  set_compile_mode(CompileMode::enabled);
  auto compiled = compile(fn);
  uint64_t d0 = mlx::core::omarchy::trace::counters().vk_compute_dispatches.load();
  uint64_t w0 =
      mlx::core::omarchy::trace::counters().vk_descriptor_update_writes.load();
  std::vector<array> outs = compiled(std::vector<array>{gate, up});
  for (auto& o : outs) {
    o.eval();
  }
  sync_stream(stream);
  set_compile_mode(CompileMode::disabled);
  uint64_t d = mlx::core::omarchy::trace::counters().vk_compute_dispatches.load() - d0;
  uint64_t w =
      mlx::core::omarchy::trace::counters().vk_descriptor_update_writes.load() - w0;
  return {outs[0], {d, w}};
}

void check_swiglu_case(Dtype dtype) {
  Stream stream = gpu_stream();
  set_compile_mode(CompileMode::disabled);
  set_fusion("1");

  auto input = [&]() {
    array a = astype(
        random::normal(Shape{4, 64}, float32, std::nullopt, stream),
        dtype,
        stream);
    a.eval();
    return a;
  };
  array gate = input();
  array up = input();
  sync_stream(stream);

  set_trio("0");
  auto [solo, solo_counts] = run_arm(gate, up, stream);
  set_trio("1");
  auto [trio, trio_counts] = run_arm(gate, up, stream);

  REQUIRE_EQ(trio.shape(), solo.shape());
  expect_bits_equal(trio, solo);
  // Same one chain dispatch in both arms; the trio kernel writes 2 more
  // descriptors (7 bindings vs 5). Anything else means the fused kernel
  // did not run and the bit comparison proved nothing.
  INFO("dispatches: solo=", solo_counts.first, " trio=", trio_counts.first,
       " writes: solo=", solo_counts.second, " trio=", trio_counts.second);
  CHECK_EQ(trio_counts.first, solo_counts.first);
  CHECK_EQ(trio_counts.second, solo_counts.second + 2u);
}

} // namespace

TEST_CASE("bf16 trio swiglu fires and is bit-exact") {
  if (!compute_available()) {
    return;
  }
  setenv("MLX_OMARCHY_ALLOW_NON_APPLE", "1", 0);
  check_swiglu_case(bfloat16);
}

TEST_CASE("f16 trio swiglu stays bit-exact") {
  if (!compute_available()) {
    return;
  }
  setenv("MLX_OMARCHY_ALLOW_NON_APPLE", "1", 0);
  check_swiglu_case(float16);
}
