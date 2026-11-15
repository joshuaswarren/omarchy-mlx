// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// RoPE pair fusion (MLX_OMARCHY_FUSED_TRIO, TRIO_ROPE_PAIR mode). The
// pair dispatch must produce the same stored bits as the two standalone
// RoPE dispatches it replaces, for float16 AND bfloat16, and the pair
// must actually fire: the dispatch-count delta between the two arms
// proves the fused kernel ran, because a refused pair silently falls
// back to the standalone path and would still be bit-exact.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cstdint>
#include <cstdlib>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"
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

constexpr int kBatch = 1;
constexpr int kTime = 3;
constexpr int kQHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 64;
constexpr float kRopeBase = 1000000.0f;
constexpr int kOffset = 17;

// One decode-shaped graph: q and k RoPE over a [B,T,H,D] producer, the
// transposed [B,H,T,D] view when transposed (the production GEMV-slice
// layout), one shared scalar offset, one stream. The inputs are built
// once per case so both arms rotate the exact same bytes.
std::pair<array, array> make_inputs(
    Dtype q_dtype,
    Dtype k_dtype,
    bool equal_last_pair_dim,
    const Stream& stream) {
  auto side = [&](int heads, Dtype d) {
    array in = astype(
        random::normal(
            Shape{kBatch, kTime, heads, kHeadDim},
            float32,
            std::nullopt,
            stream),
        d,
        stream);
    in.eval();
    return in;
  };
  // The planner pairs rope nodes with equal shape(-2): time for the
  // transposed production views, heads for row-contiguous producers.
  int kv_heads = equal_last_pair_dim ? kQHeads : kKvHeads;
  return {side(kQHeads, q_dtype), side(kv_heads, k_dtype)};
}

std::pair<array, array> rope_graph(
    const array& q_in,
    const array& k_in,
    bool transposed,
    const Stream& stream) {
  array q = q_in;
  array k = k_in;
  if (transposed) {
    q = transpose(q_in, std::vector<int>{0, 2, 1, 3}, stream);
    k = transpose(k_in, std::vector<int>{0, 2, 1, 3}, stream);
  }
  array q_out = fast::rope(q, kHeadDim, false, kRopeBase, 1.0f, kOffset, std::nullopt, stream);
  array k_out = fast::rope(k, kHeadDim, false, kRopeBase, 1.0f, kOffset, std::nullopt, stream);
  // One tape: the decode step evaluates the whole token graph together,
  // and the pair planner only sees rope nodes that share a plan.
  eval(std::vector<array>{q_out, k_out});
  sync_stream(stream);
  return {q_out, k_out};
}

void expect_bits_equal(const array& a, const array& b) {
  REQUIRE_EQ(a.size(), b.size());
  const uint16_t* pa = a.data<uint16_t>();
  const uint16_t* pb = b.data<uint16_t>();
  for (size_t i = 0; i < a.size(); ++i) {
    INFO("bit mismatch at ", i, " pair=0x", std::hex, pa[i], " solo=0x", pb[i]);
    CHECK_EQ(pa[i], pb[i]);
  }
}

// Runs both arms for one case and checks bits AND the fused-dispatch
// delta: the pair arm must use exactly one fewer compute dispatch than
// the standalone arm (one trio kernel replaces two rope kernels).
void check_pair_case(
    Dtype dtype,
    bool transposed,
    Dtype k_dtype = bfloat16) {
  Stream stream = gpu_stream();
  set_compile_mode(CompileMode::disabled);
  const auto& caps =
      mlx::core::omarchy::get_command_encoder(stream).device().capabilities();
  INFO("caps: shader_float16=", caps.shader_float16,
       " shader_int16=", caps.shader_int16,
       " storage16=", caps.storage_buffer_16bit_access);
  auto [q_in, k_in] = make_inputs(dtype, k_dtype, !transposed, stream);

  set_trio("0");
  uint64_t before_solo =
      mlx::core::omarchy::trace::counters().vk_compute_dispatches.load();
  auto [solo_q, solo_k] = rope_graph(q_in, k_in, transposed, stream);
  uint64_t solo =
      mlx::core::omarchy::trace::counters().vk_compute_dispatches.load() -
      before_solo;

  set_trio("1");
  uint64_t before_pair =
      mlx::core::omarchy::trace::counters().vk_compute_dispatches.load();
  auto [pair_q, pair_k] = rope_graph(q_in, k_in, transposed, stream);
  uint64_t pair =
      mlx::core::omarchy::trace::counters().vk_compute_dispatches.load() -
      before_pair;

  REQUIRE_EQ(pair_q.shape(), solo_q.shape());
  REQUIRE_EQ(pair_k.shape(), solo_k.shape());
  expect_bits_equal(pair_q, solo_q);
  expect_bits_equal(pair_k, solo_k);
  bool pair_fires = (dtype == k_dtype);
  std::cout << "case dtypes=" << (dtype == bfloat16 ? "bf16" : "f16")
            << "/" << (k_dtype == bfloat16 ? "bf16" : "f16")
            << " transposed=" << transposed << " dispatches: solo=" << solo
            << " pair=" << pair << " (expect pair=" << solo - (pair_fires ? 1 : 0)
            << ")\n";
  CHECK_EQ(pair, solo - (pair_fires ? 1 : 0));
}

} // namespace

TEST_CASE("bf16 rope pair fires and is bit-exact, row contiguous") {
  if (!compute_available()) {
    return;
  }
  setenv("MLX_OMARCHY_ALLOW_NON_APPLE", "1", 0);
  check_pair_case(bfloat16, false);
}

TEST_CASE("bf16 rope pair fires and is bit-exact, head/seq transposed") {
  if (!compute_available()) {
    return;
  }
  setenv("MLX_OMARCHY_ALLOW_NON_APPLE", "1", 0);
  check_pair_case(bfloat16, true);
}

TEST_CASE("f16 rope pair still fires and is bit-exact") {
  if (!compute_available()) {
    return;
  }
  setenv("MLX_OMARCHY_ALLOW_NON_APPLE", "1", 0);
  check_pair_case(float16, true, float16);
}

TEST_CASE("mixed f16/bf16 rope pair falls back bit-exact") {
  if (!compute_available()) {
    return;
  }
  setenv("MLX_OMARCHY_ALLOW_NON_APPLE", "1", 0);
  check_pair_case(bfloat16, true, float16);
}
