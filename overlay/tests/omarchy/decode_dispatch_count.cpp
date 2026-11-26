// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// FuseDecodeChains, 2026-09-02. Counts real Vulkan compute dispatches for
// one Qwen2.5-0.5B-Instruct decode token (batch 1, one new token, fp32
// structure faithful to mlx-lm: fast rms_norm, rope, SDPA; quantization
// lives in the qmm path, so this graph matches the bf16 leg exactly).
//
// The count is host-side bookkeeping in the Omarchy encoder, so it is
// valid on llvmpipe; only the tokens-per-second claim needs m1-test-host.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"

using namespace mlx::core;
using mlx::core::omarchy::trace::counters;

namespace {

// Qwen2.5-0.5B-Instruct shapes.
constexpr int kHidden = 896;
constexpr int kHeads = 14;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 64;
constexpr int kIntermediate = 4864;
constexpr int kVocab = 151936;
constexpr float kRmsEps = 1e-6f;
constexpr float kRopeBase = 1000000.0f;
constexpr float kScaleStd = 0.02f;

struct StepWeights {
  array ln1 = zeros({1}, float32);
  array ln2 = zeros({1}, float32);
  array wq = zeros({1}, float32);
  array wk = zeros({1}, float32);
  array wv = zeros({1}, float32);
  array wo = zeros({1}, float32);
  array gate = zeros({1}, float32);
  array up = zeros({1}, float32);
  array down = zeros({1}, float32);
};

std::vector<StepWeights> make_layers(int n_layers, const Stream& stream) {
  std::vector<StepWeights> layers;
  auto randn = [&](Shape shape) {
    array a = multiply(random::normal(shape, float32, std::nullopt, stream),
                       array(kScaleStd),
                       stream);
    a.eval();
    return a;
  };
  for (int i = 0; i < n_layers; ++i) {
    StepWeights lw;
    lw.ln1 = randn({kHidden});
    lw.ln2 = randn({kHidden});
    lw.wq = randn({kHidden, kHeads * kHeadDim});
    lw.wk = randn({kHidden, kKvHeads * kHeadDim});
    lw.wv = randn({kHidden, kKvHeads * kHeadDim});
    lw.wo = randn({kHeads * kHeadDim, kHidden});
    lw.gate = randn({kHidden, kIntermediate});
    lw.up = randn({kHidden, kIntermediate});
    lw.down = randn({kIntermediate, kHidden});
    layers.push_back(lw);
  }
  return layers;
}

// One decode token through the model: the exact op sequence mlx-lm's
// Qwen2 decoder runs for seq_len 1. mlx-lm's nn.silu decomposes to
// gate * sigmoid(gate) on this pinned mlx, which is what the compiled
// tape actually sees.
array decode_token(
    const std::vector<StepWeights>& layers,
    const array& lm_head,
    const array& x) {
  array h = x;
  for (const auto& lw : layers) {
    array a = fast::rms_norm(h, lw.ln1, kRmsEps);
    array q = matmul(a, lw.wq);
    array k = matmul(a, lw.wk);
    array v = matmul(a, lw.wv);
    q = reshape(q, {1, 1, kHeads, kHeadDim});
    k = reshape(k, {1, 1, kKvHeads, kHeadDim});
    v = reshape(v, {1, 1, kKvHeads, kHeadDim});
    q = transpose(q, {0, 2, 1, 3});
    k = transpose(k, {0, 2, 1, 3});
    v = transpose(v, {0, 2, 1, 3});
    q = fast::rope(q, kHeadDim, false, kRopeBase, 1.0f, 0);
    k = fast::rope(k, kHeadDim, false, kRopeBase, 1.0f, 0);
    array attn = fast::scaled_dot_product_attention(
        q, k, v, 1.0f / std::sqrt(static_cast<float>(kHeadDim)));
    attn = transpose(attn, {0, 2, 1, 3});
    attn = reshape(attn, {1, 1, kHeads * kHeadDim});
    h = h + matmul(attn, lw.wo);
    a = fast::rms_norm(h, lw.ln2, kRmsEps);
    array gate = matmul(a, lw.gate);
    array up = matmul(a, lw.up);
    h = h + matmul(gate * sigmoid(gate) * up, lw.down);
  }
  array out = fast::rms_norm(h, std::nullopt, kRmsEps);
  return matmul(out, lm_head);
}

uint64_t dispatches_for(const std::function<array()>& step, int warmups) {
  for (int i = 0; i < warmups; ++i) {
    array out = step();
    out.eval();
  }
  uint64_t before = counters().vk_compute_dispatches.load();
  array out = step();
  out.eval();
  uint64_t after = counters().vk_compute_dispatches.load();
  return after - before;
}

} // namespace

TEST_CASE("decode step dispatch count") {
  if (!gpu::is_available()) {
    printf("Skipping: no Vulkan device\n");
    return;
  }
  set_default_device(Device::gpu);
  Stream stream = new_stream(Device::gpu);
  int n_layers = 24;
  if (const char* env = std::getenv("LAYERS")) {
    n_layers = std::atoi(env);
  }

  auto layers = make_layers(n_layers, stream);
  array lm_head =
      multiply(random::normal(Shape{kHidden, kVocab}, float32, std::nullopt, stream),
               array(kScaleStd),
               stream);
  lm_head.eval();
  array x = multiply(
      random::normal(Shape{1, 1, kHidden}, float32, std::nullopt, stream),
      array(kScaleStd),
      stream);
  x.eval();
  setenv("MLX_OMARCHY_FUSED_CHAIN", "0", 1);

  uint64_t eager =
      dispatches_for([&] { return decode_token(layers, lm_head, x); }, 1);

  std::function<std::vector<array>(const std::vector<array>&)> fn =
      [&](const std::vector<array>& in) {
        return std::vector<array>{decode_token(layers, lm_head, in[0])};
      };
  // Explicitly disabled fusion preserves the per-node tape stream and must
  // equal eager exactly.
  set_compile_mode(CompileMode::enabled);
  auto compiled_fn = compile(fn);
  uint64_t compiled =
      dispatches_for([&] { return compiled_fn({x})[0]; }, 1);
  set_compile_mode(CompileMode::disabled);

  // The default-on path collapses each layer's SwiGLU chain from three
  // dispatches to one and touches no other decode operation.
  unsetenv("MLX_OMARCHY_FUSED_CHAIN");
  set_compile_mode(CompileMode::enabled);
  auto fused_fn = compile(fn);
  uint64_t compiled_fused =
      dispatches_for([&] { return fused_fn({x})[0]; }, 1);
  set_compile_mode(CompileMode::disabled);
  unsetenv("MLX_OMARCHY_FUSED_CHAIN");

  CHECK_EQ(compiled, eager);
  CHECK_EQ(compiled_fused, compiled - 2ull * n_layers);

  printf(
      "decode dispatches/token (layers=%d): eager=%llu"
      " compiled_pernode=%llu compiled_fused=%llu\n",
      n_layers,
      static_cast<unsigned long long>(eager),
      static_cast<unsigned long long>(compiled),
      static_cast<unsigned long long>(compiled_fused));
}
