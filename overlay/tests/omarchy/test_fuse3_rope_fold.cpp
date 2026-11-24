// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// FUSE-3 fold 1, the Q side: sdpa_decode_split_p1 built with
// -DSDPA_QROPE computes fast_rope_norm's RMS norm + rotation of the RAW
// query inside the attention dispatch instead of reading a rope_rms_norm
// output. The contract is bit-identity: for a fixed raw query the folded
// kernel's scratch (scores, softmax numerators, weighted value sums)
// must equal the unfused kernel's scratch on the rope_rms_norm output
// bit for bit, on whichever reduction path the rope host would pick
// (Apple subgroup reduction vs the shared tree; llvmpipe usually takes
// the non-Apple path, Apple hardware the Apple path — the test mirrors
// the rope host's selection so the same path runs on both arms).
//
// The test drives three kernels directly at the route's guarded shape
// (batch 1, one decode token, bf16, 8 query heads over 2 kv heads,
// 32-lane subgroup arithmetic):
//   1. FastRopeNormBF16 / FastRopeNormAppleBF16: raw q -> roped q
//   2. SdpaDecodeSplitP1R4BF16Hd256 (roped q)     -> unfused scratch
//   3. SdpaDecodeSplitP1R4BF16Hd256Qrope[Apple] (raw q) -> folded scratch
// Arms 2 and 3 must agree word for word. Without the extra bindings the
// folded kernel is never selected: the plain kernels compile unchanged.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/compute.h"
#include "mlx/backend/omarchy/allocator.h"
#include "mlx/backend/omarchy/device.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

constexpr uint32_t HD = 256;    // head width of the routed split kernels
constexpr uint32_t SPLIT_R = 4; // query heads sharing the kv head
constexpr uint32_t KV_HEADS = 2; // 8 query heads over 2 kv heads
constexpr uint32_t K_LEN = 35;  // the decode prompt step
constexpr float ROPE_SCALE = 1.0f;
constexpr float ROPE_EPS = 1e-6f;
// The half-split freq exponent: the rope host passes log(base) / half_dims
// (base 1000000, half dims 128 for the full-width rotation).
constexpr float ROPE_BETA = 13.815510558f / 128.0f;
constexpr float SDPA_SCALE = 0.0625f; // 1/sqrt(256)

bool compute_available() {
  if (!gpu::is_available()) {
    printf("Skipping: no qualifying Vulkan device\n");
    return false;
  }
  return true;
}

Stream gpu_stream() {
  set_default_device(Device::gpu);
  return new_stream(Device::gpu);
}

std::vector<float> pattern(size_t count, uint32_t seed, float amplitude) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        amplitude * (static_cast<float>(state % 20000u) / 10000.0f - 1.0f));
  }
  return values;
}

array bf16_array(const std::vector<float>& host, Shape shape, const Stream& s) {
  array f(host.data(), shape, float32);
  array b = astype(f, bfloat16, s);
  b.eval();
  return b;
}

omarchy::ComputeBinding bind(const array& a) {
  auto* buffer = static_cast<const omarchy::VulkanBuffer*>(a.buffer().ptr());
  return omarchy::ComputeBinding{buffer->buffer, 0, buffer->size};
}

// The rope host's norm-route selection (primitives.cpp apple_norm_enabled
// with the fused-norm shape): 32-lane subgroup arithmetic takes the Apple
// reduction; everything else takes the shared tree. The folded kernel
// must ride the SAME path as the rope kernel on both arms.
bool apple_norm_route(const Stream& s) {
  const auto& caps = omarchy::device(0).capabilities();
  return caps.subgroup_size == 32u &&
      (caps.subgroup_operations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
}

// Wait for everything recorded so far. Arrays written by a direct
// dispatch carry no primitive, so they are never eval()ed; the
// encoder's synchronize is the completion point for both arms.
void sync(const Stream& s) {
  omarchy::get_command_encoder(s).synchronize();
}

// One batched rope_rms_norm dispatch over the (1, heads, 1, HD) query,
// mirroring the rope host's fused-norm parameter packing: eps bits ride
// operation, the row length rides reduce_size, the weight element offset
// rides aux_offset, the half dims ride dims, forward rides flag 1, and
// the binding order is src, output, offsets, offsets, weight.
void rope_rms_norm_batch(
    const array& raw,
    array& out,
    const array& weight,
    const array& offsets,
    const Stream& s) {
  omarchy::ComputeParams p;
  p.count = static_cast<uint32_t>(raw.shape(1)); // one row per head
  p.operation = std::bit_cast<uint32_t>(ROPE_EPS);
  p.reduce_size = HD;
  p.aux_offset = 0;
  p.dims = HD / 2; // half dims (the rope host passes dims_/2)
  p.flags = 1u; // forward
  p.alpha = ROPE_SCALE;
  p.beta = ROPE_BETA;
  p.lhs_offset = 0;
  p.rhs_offset = 0;
  p.output_offset = 0;
  p.matrix_m = 0; // the offsets stride (batch 0 reads element 0)
  p.shape[0] = static_cast<uint32_t>(raw.shape(1)); // heads
  p.shape[1] = 1;                                   // time
  p.shape[2] = HD;
  p.in_strides[0] = static_cast<uint32_t>(raw.strides()[1]);
  p.in_strides[1] = static_cast<uint32_t>(raw.strides()[2]);
  p.in_strides[2] = 1;
  p.out_strides[0] = static_cast<uint32_t>(out.strides()[1]);
  p.out_strides[1] = HD;
  p.out_strides[2] = 1;
  std::array<omarchy::ComputeBinding, 5> bindings{
      bind(raw), bind(out), bind(offsets), bind(offsets), bind(weight)};
  auto& encoder = omarchy::get_command_encoder(s);
  const auto kernel = apple_norm_route(s)
      ? omarchy::ComputeKernel::FastRopeNormAppleBF16
      : omarchy::ComputeKernel::FastRopeNormBF16;
  encoder.dispatch_compute(
      kernel, bindings, p, static_cast<uint32_t>(raw.shape(1)), 1, 1);
  sync(s);
}

// One split pass 1 dispatch over both kv heads. When |qrope| the query
// binding carries the RAW array plus the rope bindings, and the rope
// params ride the free push fields (eps bits in operation, the rope scale
// bits in lhs_size, the freq exponent in beta, the half dims in dims, the
// query row's time index in matrix_m, the offsets buffer offset in
// output_offset, the weight element offset in aux_size).
void split_p1(
    const array& q,
    const array& k,
    const array& v,
    array& scratch,
    const Stream& s,
    bool qrope,
    const array* weight,
    const array* offsets,
    bool pv = false) {
  std::vector<omarchy::ComputeBinding> bindings{
      bind(q), bind(k), bind(v), bind(scratch)};
  omarchy::ComputeParams p;
  p.count = static_cast<uint32_t>(scratch.size());
  p.lhs_offset = 0;
  p.rhs_offset = 0;
  p.aux_offset = 0;
  p.output_offset = 0;
  p.matrix_k = K_LEN;
  p.alpha = SDPA_SCALE;
  p.shape[0] = static_cast<uint32_t>(q.strides()[1]);
  p.shape[1] = static_cast<uint32_t>(k.strides()[1]);
  p.shape[2] = static_cast<uint32_t>(k.strides()[2]);
  p.shape[3] = static_cast<uint32_t>(q.strides()[3]);
  p.in_strides[0] = 1;
  p.in_strides[1] = static_cast<uint32_t>(v.strides()[1]);
  p.in_strides[2] = static_cast<uint32_t>(v.strides()[2]);
  p.in_strides[3] = 1;
  const auto kernel = qrope
      ? omarchy::ComputeKernel::SdpaDecodeSplitP1R4BF16Hd256Qrope
      : omarchy::ComputeKernel::SdpaDecodeSplitP1R4BF16Hd256;
  const auto kernel_pv = omarchy::ComputeKernel::SdpaDecodeSplitP1VR4BF16Hd256;
  if (qrope) {
    p.operation = std::bit_cast<uint32_t>(ROPE_EPS);
    p.lhs_size = std::bit_cast<uint32_t>(ROPE_SCALE);
    p.beta = ROPE_BETA;
    p.dims = HD / 2; // half dims (the rope host passes dims_/2)
    p.matrix_m = 0; // the decode token's time index in this harness
    p.output_offset = 0;
    p.aux_size = 0;
    bindings.push_back(bind(*weight));
    bindings.push_back(bind(*offsets));
  }
  auto& encoder = omarchy::get_command_encoder(s);
  const bool apple = apple_norm_route(s);
  const auto routed = !qrope ? kernel
      : pv ? (apple ? omarchy::ComputeKernel::SdpaDecodeSplitP1VR4BF16Hd256QropeApple
                    : omarchy::ComputeKernel::SdpaDecodeSplitP1VR4BF16Hd256Qrope)
           : (apple ? omarchy::ComputeKernel::SdpaDecodeSplitP1R4BF16Hd256QropeApple
                    : omarchy::ComputeKernel::SdpaDecodeSplitP1R4BF16Hd256Qrope);
  const auto plain = pv ? kernel_pv : kernel;
  encoder.dispatch_compute(
      qrope ? routed : plain, bindings, p, static_cast<uint32_t>(KV_HEADS), 1u, 1u);
  sync(s);
}

} // namespace

TEST_CASE("folded split p1 scratch equals the unfused rope+split scratch") {
  if (!compute_available()) {
    return;
  }
  const auto s = gpu_stream();
  auto& encoder = omarchy::get_command_encoder(s);
  encoder.synchronize();

  const uint32_t heads = KV_HEADS * SPLIT_R;
  auto q_host = pattern(static_cast<size_t>(heads) * HD, 100003u, 1.0f);
  auto w_host = pattern(HD, 200003u, 0.5f);
  auto off_host = std::vector<int>{0};
  auto k_host = pattern(static_cast<size_t>(KV_HEADS) * K_LEN * HD, 300003u, 1.0f);
  auto v_host = pattern(static_cast<size_t>(KV_HEADS) * K_LEN * HD, 400003u, 1.0f);

  array raw = bf16_array(q_host, Shape{1, heads, 1, HD}, s);
  array weight = bf16_array(w_host, Shape{HD}, s);
  // A constant needs a primitive to eval on the device (the
  // test_primitives.cpp pattern): x + 0 keeps the bits.
  array offsets = add(array(off_host.data(), Shape{1}, int32), array(0, int32), s);
  offsets.eval();
  array k = bf16_array(k_host, Shape{1, KV_HEADS, K_LEN, HD}, s);
  array v = bf16_array(v_host, Shape{1, KV_HEADS, K_LEN, HD}, s);

  // The scratch row the split kernels write: SPLIT_MAX, SPLIT_SUM and the
  // D weighted-value floats per (kv head, query head); the exact layout
  // does not matter to this contract because both arms write the same
  // cells from the same q/k/v.
  array roped = array(raw.shape(), raw.dtype(), nullptr, {});
  roped.set_data(omarchy::allocator().malloc(roped.nbytes()));
  array scratch_unfused(
      std::vector<float>(heads * (HD + 2u), 0.0f).data(),
      Shape{KV_HEADS, SPLIT_R, HD + 2u}, float32);
  array scratch_folded(
      std::vector<float>(heads * (HD + 2u), 0.0f).data(),
      Shape{KV_HEADS, SPLIT_R, HD + 2u}, float32);
  scratch_unfused.set_data(omarchy::allocator().malloc(scratch_unfused.nbytes()));
  scratch_folded.set_data(omarchy::allocator().malloc(scratch_folded.nbytes()));
  sync(s);

  rope_rms_norm_batch(raw, roped, weight, offsets, s);
  const size_t words = heads * (HD + 2u);
  for (const bool pv : {false, true}) {
    split_p1(roped, k, v, scratch_unfused, s, false, nullptr, nullptr, pv);
    split_p1(raw, k, v, scratch_folded, s, true, &weight, &offsets, pv);

    const auto* unfused = scratch_unfused.data<float>();
    const auto* folded = scratch_folded.data<float>();
    size_t diffs = 0;
    for (size_t i = 0; i < words; ++i) {
      if (std::memcmp(unfused + i, folded + i, sizeof(float)) != 0) {
        if (diffs < 8u) {
          std::printf("pv=%d word %zu: unfused %.9g folded %.9g\n", int(pv), i, unfused[i], folded[i]);
        }
        ++diffs;
      }
    }
    CAPTURE(pv);
    CHECK_EQ(diffs, size_t{0});
  }
}

TEST_CASE("the folded route leaves the attention result unchanged") {
  if (!compute_available()) {
    return;
  }
  const auto s = gpu_stream();
  constexpr uint32_t heads = KV_HEADS * SPLIT_R;
  auto q_host = pattern(static_cast<size_t>(heads) * HD, 100003u, 1.0f);
  auto w_host = pattern(HD, 200003u, 0.5f);
  auto k_host = pattern(static_cast<size_t>(KV_HEADS) * K_LEN * HD, 300003u, 1.0f);
  auto v_host = pattern(static_cast<size_t>(KV_HEADS) * K_LEN * HD, 400003u, 1.0f);

  auto run = [&](bool fold) {
    setenv("MLX_OMARCHY_FUSE3", fold ? "1" : "0", 1);
    array x = bf16_array(q_host, Shape{1, heads, 1, HD}, s);
    array w = bf16_array(w_host, Shape{HD}, s);
    array k = bf16_array(k_host, Shape{1, KV_HEADS, K_LEN, HD}, s);
    array v = bf16_array(v_host, Shape{1, KV_HEADS, K_LEN, HD}, s);
    array off = zeros({1}, int32, s);
    array roped = fast::rope_rms_norm(
        x, HD, w, ROPE_EPS, false, 1000000.0f, 1.0f, off, s);
    array out = fast::scaled_dot_product_attention(
        roped, k, v, SDPA_SCALE, "", std::nullopt, std::nullopt, false, s);
    out.eval();
    omarchy::get_command_encoder(s).synchronize();
    std::vector<float> values(out.size());
    std::memcpy(
        values.data(), out.data<float>(), values.size() * sizeof(float));
    return values;
  };

  auto unfused = run(false);
  auto folded = run(true);
  unsetenv("MLX_OMARCHY_FUSE3");
  // The fold must not change a single bit of the attention result. On a
  // device without the split route the fold refuses and the skipped rope
  // node runs in place, which this also covers.
  CHECK_EQ(folded.size(), unfused.size());
  size_t diffs = 0;
  for (size_t i = 0; i < unfused.size() && i < folded.size(); ++i) {
    if (std::memcmp(&unfused[i], &folded[i], sizeof(float)) != 0) {
      if (diffs < 8u) {
        std::printf(
            "value %zu: unfused %.9g folded %.9g\n", i, unfused[i], folded[i]);
      }
      ++diffs;
    }
  }
  CHECK_EQ(diffs, size_t{0});
}
