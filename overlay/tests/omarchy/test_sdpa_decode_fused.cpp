// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// Bf16FusedDecode, 2026-09-11. The fused single-query decode SDPA
// (SdpaDecodeNativeF16 / SdpaDecodeNativeBF16) must keep one dispatch per
// call, agree with the f32-score composition it replaced on the covered
// decode shapes, and refuse uncovered shapes by falling through to that
// composition rather than changing arithmetic. Route engagement is judged
// by the host-side dispatch counter (valid on any Vulkan device); the
// fused-route assertions only run when the capability probe shows the f16
// sentinel engaging, so software drivers skip them by name instead of
// testing the composition twice.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

// Qwen2.5-0.5B decode attention shape.
constexpr int kHeads = 14;
constexpr int kKvHeads = 2;
constexpr int kKeys = 263;
constexpr int kCapacity = 320;
constexpr int kWidth = 64;
constexpr float kScale = 1.0f / std::sqrt(float(kWidth));

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

// Deterministic pseudo-random values in [-1, 1).
std::vector<float> pattern(size_t count, uint32_t seed) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (size_t index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        static_cast<float>(static_cast<double>(state % 20000u) / 10000.0) -
        1.0f);
  }
  return values;
}

std::vector<float> flat(const array& value, Stream stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = copy.data<float>();
  return std::vector<float>(data, data + copy.size());
}

void require_close(
    const std::vector<float>& got,
    const std::vector<float>& want,
    double tolerance,
    const std::string& what) {
  REQUIRE_EQ(got.size(), want.size());
  for (size_t index = 0; index < want.size(); ++index) {
    double diff = std::abs(static_cast<double>(got[index]) - want[index]);
    CHECK_MESSAGE(
        diff <= tolerance,
        what,
        " element ",
        index,
        ": got ",
        got[index],
        " want ",
        want[index]);
  }
}

uint64_t dispatches_for(const std::function<array()>& step, Stream stream) {
  array out = step();
  out.eval();
  omarchy::get_command_encoder(stream).synchronize();
  uint64_t before = omarchy::trace::counters().vk_compute_dispatches.load();
  out = step();
  out.eval();
  omarchy::get_command_encoder(stream).synchronize();
  uint64_t after = omarchy::trace::counters().vk_compute_dispatches.load();
  return after - before;
}

// Strided KV-cache slices plus a fresh q, the exact decode layout mlx-lm
// hands the attention: q is contiguous, k and v are capacity-strided views.
struct CacheInputs {
  array q;
  array k;
  array v;

  CacheInputs(array q_, array k_, array v_)
      : q(std::move(q_)), k(std::move(k_)), v(std::move(v_)) {}
};

CacheInputs make_cache(Dtype dtype, Stream stream) {
  auto q_values = pattern(kHeads * kWidth, 11);
  auto k_values = pattern(kKvHeads * kCapacity * kWidth, 22);
  auto v_values = pattern(kKvHeads * kCapacity * kWidth, 33);
  array q = astype(
      array(q_values.begin(), Shape{1, kHeads, 1, kWidth}, float32),
      dtype,
      stream);
  array k_cache = astype(
      array(k_values.begin(), Shape{1, kKvHeads, kCapacity, kWidth}, float32),
      dtype,
      stream);
  array v_cache = astype(
      array(v_values.begin(), Shape{1, kKvHeads, kCapacity, kWidth}, float32),
      dtype,
      stream);
  array k = slice(
      k_cache, {0, 0, 0, 0}, {1, kKvHeads, kKeys, kWidth}, stream);
  array v = slice(
      v_cache, {0, 0, 0, 0}, {1, kKvHeads, kKeys, kWidth}, stream);
  q.eval();
  k.eval();
  v.eval();
  omarchy::get_command_encoder(stream).synchronize();
  return CacheInputs(std::move(q), std::move(k), std::move(v));
}

array sdpa_call(const CacheInputs& in, Stream stream) {
  return fast::scaled_dot_product_attention(
      in.q, in.k, in.v, kScale, "", {}, std::nullopt, false, stream);
}

// The f32-score composition the fused route replaced, expressed with the
// same primitives: upcast, scale, scores matmul, softmax, probs matmul,
// downcast. GQA rides reshape shapes like the backend does.
array composition_reference(const CacheInputs& in, Stream stream) {
  int q_len = in.q.shape(2);
  array q32 = multiply(astype(in.q, float32, stream), array(kScale), stream);
  array k32 = astype(in.k, float32, stream);
  array v32 = astype(in.v, float32, stream);
  array qs = reshape(
      q32, Shape{1, kKvHeads, kHeads / kKvHeads, q_len, kWidth}, stream);
  array kt = swapaxes(
      reshape(k32, Shape{1, kKvHeads, 1, kKeys, kWidth}, stream),
      -1,
      -2,
      stream);
  array vs = reshape(v32, Shape{1, kKvHeads, 1, kKeys, kWidth}, stream);
  array scores = matmul(qs, kt, stream);
  array probs = softmax(scores, std::vector<int>{-1}, false, stream);
  array result = matmul(probs, vs, stream);
  return reshape(result, Shape{1, kHeads, q_len, kWidth}, stream);
}

bool bf16_route_ready(Stream stream) {
  // The composition-exact bf16 arm gates without subgroup requirements, so
  // it engages on any qualifying device - including llvmpipe, where the
  // bit-identity gate below is therefore exercised for real.
  CacheInputs bf16 = make_cache(bfloat16, stream);
  return dispatches_for([&] { return sdpa_call(bf16, stream); }, stream) == 1;
}

bool decode_route_ready(Stream stream) {
  // Sentinel: when the device qualifies, the f16 decode shape costs one
  // dispatch. Anything else means the fused route is refused here and the
  // fused-route assertions cannot be exercised on this device.
  CacheInputs f16 = make_cache(float16, stream);
  return dispatches_for([&] { return sdpa_call(f16, stream); }, stream) == 1;
}

} // namespace

TEST_CASE("fused bf16 decode SDPA is one dispatch and matches the composition") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  bool fused_available = decode_route_ready(stream);
  if (!fused_available) {
    printf("Skipping fused-route assertions: decode route refuses on this "
           "device (composition agreement still exercised)\n");
  }

  CacheInputs bf16 = make_cache(bfloat16, stream);
  if (fused_available) {
    uint64_t dispatches =
        dispatches_for([&] { return sdpa_call(bf16, stream); }, stream);
    CHECK_EQ(dispatches, 1);
  }
  require_close(
      flat(sdpa_call(bf16, stream), stream),
      flat(composition_reference(bf16, stream), stream),
      0.01,
      "fused bf16 decode vs f32-score composition");
}

TEST_CASE("decode shapes the fused route must refuse fall through to the composition") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();

  // q_len=2 is not single-query decode: the fused route must refuse it and
  // the composition must still answer it correctly.
  auto q2_values = pattern(kHeads * 2 * kWidth, 44);
  CacheInputs kv = make_cache(bfloat16, stream);
  array q2 = astype(
      array(q2_values.begin(), Shape{1, kHeads, 2, kWidth}, float32),
      bfloat16,
      stream);
  q2.eval();
  omarchy::get_command_encoder(stream).synchronize();
  CacheInputs q2_inputs(q2, kv.k, kv.v);
  uint64_t dispatches = dispatches_for(
      [&] {
        return fast::scaled_dot_product_attention(
            q2, kv.k, kv.v, kScale, "", {}, std::nullopt, false, stream);
      },
      stream);
  CHECK(dispatches > 1);
  require_close(
      flat(fast::scaled_dot_product_attention(
               q2, kv.k, kv.v, kScale, "", {}, std::nullopt, false, stream),
           stream),
      flat(composition_reference(q2_inputs, stream), stream),
      0.01,
      "refused q_len=2 falls through to the composition");

  // An additive array mask rides mask_arr: the fused route takes no mask,
  // so this decode shape must also compose.
  auto mask_values = pattern(1 * kHeads * 1 * kKeys, 55);
  array mask = astype(
      array(mask_values.begin(), Shape{1, kHeads, 1, kKeys}, float32),
      bfloat16,
      stream);
  mask.eval();
  omarchy::get_command_encoder(stream).synchronize();
  uint64_t mask_dispatches = dispatches_for(
      [&] {
        return fast::scaled_dot_product_attention(
            kv.q, kv.k, kv.v, kScale, "", mask, std::nullopt, false, stream);
      },
      stream);
  CHECK(mask_dispatches > 1);
  array masked_scores = multiply(
      astype(kv.q, float32, stream), array(kScale), stream);
  masked_scores = matmul(
      reshape(
          masked_scores,
          Shape{1, kKvHeads, kHeads / kKvHeads, 1, kWidth},
          stream),
      swapaxes(
          reshape(
              astype(kv.k, float32, stream),
              Shape{1, kKvHeads, 1, kKeys, kWidth},
              stream),
          -1,
          -2,
          stream),
      stream);
  array mask_ref = reshape(
      astype(mask, float32, stream),
      Shape{1, kKvHeads, kHeads / kKvHeads, 1, kKeys},
      stream);
  masked_scores = add(masked_scores, mask_ref, stream);
  array masked_out = matmul(
      softmax(masked_scores, std::vector<int>{-1}, false, stream),
      reshape(
          astype(kv.v, float32, stream),
          Shape{1, kKvHeads, 1, kKeys, kWidth},
          stream),
      stream);
  require_close(
      flat(fast::scaled_dot_product_attention(
               kv.q, kv.k, kv.v, kScale, "", mask, std::nullopt, false, stream),
           stream),
      flat(reshape(masked_out, Shape{1, kHeads, 1, kWidth}, stream), stream),
      0.01,
      "refused additive-mask decode falls through to the composition");
}

namespace {

// Variable-context cache builder for the bit-identity gate: same decode
// layout as make_cache, any key count up to the capacity.
// composition_reference with the key count explicit: the fixed-shape
// helper above bakes kKeys into its reshapes.
array composition_reference_len(
    const CacheInputs& in,
    int keys,
    Stream stream) {
  int q_len = in.q.shape(2);
  array q32 = multiply(astype(in.q, float32, stream), array(kScale), stream);
  array k32 = astype(in.k, float32, stream);
  array v32 = astype(in.v, float32, stream);
  array qs = reshape(
      q32, Shape{1, kKvHeads, kHeads / kKvHeads, q_len, kWidth}, stream);
  array kt = swapaxes(
      reshape(k32, Shape{1, kKvHeads, 1, keys, kWidth}, stream), -1, -2,
      stream);
  array vs = reshape(v32, Shape{1, kKvHeads, 1, keys, kWidth}, stream);
  array scores = matmul(qs, kt, stream);
  array probs = softmax(scores, std::vector<int>{-1}, false, stream);
  array result = matmul(probs, vs, stream);
  // The real composition narrows to the output dtype through the RNE cast;
  // without this the reference carries f32 tails no bf16 route could store.
  return astype(
      reshape(result, Shape{1, kHeads, q_len, kWidth}, stream),
      bfloat16,
      stream);
}

CacheInputs make_cache_len(
    Dtype dtype,
    int keys,
    int capacity,
    Stream stream) {
  auto q_values = pattern(kHeads * kWidth, 11);
  auto k_values = pattern(kKvHeads * capacity * kWidth, 22);
  auto v_values = pattern(kKvHeads * capacity * kWidth, 33);
  array q = astype(
      array(q_values.begin(), Shape{1, kHeads, 1, kWidth}, float32),
      dtype,
      stream);
  array k_cache = astype(
      array(k_values.begin(), Shape{1, kKvHeads, capacity, kWidth}, float32),
      dtype,
      stream);
  array v_cache = astype(
      array(v_values.begin(), Shape{1, kKvHeads, capacity, kWidth}, float32),
      dtype,
      stream);
  array k = slice(
      k_cache, {0, 0, 0, 0}, {1, kKvHeads, keys, kWidth}, stream);
  array v = slice(
      v_cache, {0, 0, 0, 0}, {1, kKvHeads, keys, kWidth}, stream);
  q.eval();
  k.eval();
  v.eval();
  omarchy::get_command_encoder(stream).synchronize();
  return CacheInputs(std::move(q), std::move(k), std::move(v));
}

// The composition-exact contract is bit equality: the fused bf16 route and
// the f32-score composition must store the same bf16 words. Both sides are
// exact bf16 values widened to f32, so comparing the uint32 patterns is
// exact and catches sign-of-zero differences a numeric compare would miss.
void require_bit_identical(
    const array& got,
    const array& want,
    Stream stream) {
  auto got_bits = flat(got, stream);
  auto want_bits = flat(want, stream);
  REQUIRE_EQ(got_bits.size(), want_bits.size());
  for (size_t index = 0; index < want_bits.size(); ++index) {
    uint32_t a;
    uint32_t b;
    std::memcpy(&a, &got_bits[index], 4);
    std::memcpy(&b, &want_bits[index], 4);
    REQUIRE_MESSAGE(
        a == b,
        "bit mismatch at ",
        index,
        ": got ",
        got_bits[index],
        " want ",
        want_bits[index]);
  }
}

} // namespace

// The composition-exact bf16 arm must store the same words as the f32-score
// composition on every covered key count - one dispatch per call, bit-equal
// output - and must refuse past the shared-memory key bound by falling
// through to the composition it reproduces.
TEST_CASE("fused bf16 decode is bit-identical to the f32 composition") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  for (int keys : {1, 5, 16, 17, 64, 263, 320}) {
    CAPTURE(keys);
    CacheInputs in = make_cache_len(bfloat16, keys, 320, stream);
    if (bf16_route_ready(stream)) {
      uint64_t dispatches = dispatches_for(
          [&] { return sdpa_call(in, stream); }, stream);
      MESSAGE("keys ", keys, " dispatches ", dispatches);
      if (keys >= 256) {
        // Inside the arm's winning regime: one dispatch per call.
        CHECK_EQ(dispatches, 1);
      } else {
        // Below the measured 256-key crossover the composition is faster;
        // the perf gate routes there and both sides are bit-identical.
        CHECK(dispatches > 1);
      }
    }
    require_bit_identical(
        sdpa_call(in, stream), composition_reference_len(in, keys, stream), stream);
  }

  // Past the bf16 arm's shared-memory key bound the route must refuse and
  // the composition answers - the words are identical either way.
  CAPTURE(2100);
  CacheInputs long_cache = make_cache_len(bfloat16, 2100, 2100, stream);
  if (bf16_route_ready(stream)) {
    uint64_t dispatches = dispatches_for(
        [&] { return sdpa_call(long_cache, stream); }, stream);
    CHECK(dispatches > 1);
  }
  require_bit_identical(
      sdpa_call(long_cache, stream),
      composition_reference_len(long_cache, 2100, stream),
      stream);
}

namespace {

// Qwen3-TTS talker decode shape: head_dim 128, GQA 16 query heads over 8
// kv heads, strided capacity-backed KV views. The hd128 gap this file's
// route closes (SpeechOutputFast profile: every TTS and Qwen3.x chat
// decode attention fell back to the ~10-dispatch composition).
constexpr int kHd = 128;
constexpr int kHdHeads = 16;
constexpr int kHdKvHeads = 8;
constexpr float kHdScale = 1.0f / std::sqrt(float(kHd));

struct HdCacheInputs {
  array q;
  array k;
  array v;
};

array hd_pattern_values(int count, uint32_t seed) {
  std::vector<float> values;
  values.reserve(count);
  uint32_t state = seed;
  for (int index = 0; index < count; ++index) {
    state = state * 1664525u + 1013904223u;
    values.push_back(
        static_cast<float>(static_cast<double>(state % 20000u) / 10000.0) -
        1.0f);
  }
  return array(values.begin(), Shape{count}, float32);
}

HdCacheInputs make_hd_cache(Dtype dtype, int keys, int capacity, Stream stream) {
  array q = astype(
      hd_pattern_values(kHdHeads * kHd, 11),
      dtype,
      stream);
  q = reshape(q, Shape{1, kHdHeads, 1, kHd}, stream);
  array k_cache = astype(
      hd_pattern_values(kHdKvHeads * capacity * kHd, 22), dtype, stream);
  k_cache = reshape(k_cache, Shape{1, kHdKvHeads, capacity, kHd}, stream);
  array v_cache = astype(
      hd_pattern_values(kHdKvHeads * capacity * kHd, 33), dtype, stream);
  v_cache = reshape(v_cache, Shape{1, kHdKvHeads, capacity, kHd}, stream);
  array k = slice(
      k_cache, {0, 0, 0, 0}, {1, kHdKvHeads, keys, kHd}, stream);
  array v = slice(
      v_cache, {0, 0, 0, 0}, {1, kHdKvHeads, keys, kHd}, stream);
  q.eval();
  k.eval();
  v.eval();
  omarchy::get_command_encoder(stream).synchronize();
  return HdCacheInputs{std::move(q), std::move(k), std::move(v)};
}

array hd_sdpa(const HdCacheInputs& in, Stream stream) {
  return fast::scaled_dot_product_attention(
      in.q, in.k, in.v, kHdScale, "", {}, std::nullopt, false, stream);
}

// The f32-score composition at head_dim 128, narrowed like the real route.
array hd_composition(const HdCacheInputs& in, int keys, Stream stream) {
  array q32 = multiply(astype(in.q, float32, stream), array(kHdScale), stream);
  array k32 = astype(in.k, float32, stream);
  array v32 = astype(in.v, float32, stream);
  array qs = reshape(
      q32, Shape{1, kHdKvHeads, kHdHeads / kHdKvHeads, 1, kHd}, stream);
  array kt = swapaxes(
      reshape(k32, Shape{1, kHdKvHeads, 1, keys, kHd}, stream), -1, -2,
      stream);
  array vs = reshape(v32, Shape{1, kHdKvHeads, 1, keys, kHd}, stream);
  array scores = matmul(qs, kt, stream);
  array probs = softmax(scores, std::vector<int>{-1}, false, stream);
  array result = matmul(probs, vs, stream);
  return astype(
      reshape(result, Shape{1, kHdHeads, 1, kHd}, stream),
      in.q.dtype(),
      stream);
}

void require_hd_bit_identical(
    const array& got,
    const array& want,
    Stream stream) {
  auto got_bits = flat(got, stream);
  auto want_bits = flat(want, stream);
  REQUIRE_EQ(got_bits.size(), want_bits.size());
  for (size_t index = 0; index < want_bits.size(); ++index) {
    uint32_t a;
    uint32_t b;
    std::memcpy(&a, &got_bits[index], 4);
    std::memcpy(&b, &want_bits[index], 4);
    REQUIRE_MESSAGE(
        a == b,
        "bit mismatch at ",
        index,
        ": got ",
        got_bits[index],
        " want ",
        want_bits[index]);
  }
}

bool hd_route_ready(Stream stream) {
  HdCacheInputs probe = make_hd_cache(bfloat16, 93, 128, stream);
  return dispatches_for([&] { return hd_sdpa(probe, stream); }, stream) == 1;
}

} // namespace

// Keep the engaged/refused key boundaries in sync with kDecodeBf16Windows
// in overlay/mlx/backend/omarchy/primitives.cpp: {64, 256, 2048},
// {128, 12, 7168}, {256, 12, 7168} (hd128 row measured on the M2 Max,
// artifacts/Attn128). Bitwise identity holds at every k either way.
TEST_CASE("fused hd128 bf16 decode is bit-identical to the composition") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  bool fused_available = hd_route_ready(stream);
  if (!fused_available) {
    printf("Skipping hd128 fused-route assertions: route refuses on this "
           "device (composition agreement still exercised)\n");
  }
  for (int keys : {1, 5, 12, 34, 93, 263, 512, 2100}) {
    CAPTURE(keys);
    HdCacheInputs in = make_hd_cache(bfloat16, keys, keys < 512 ? 512 : keys, stream);
    if (fused_available) {
      uint64_t dispatches =
          dispatches_for([&] { return hd_sdpa(in, stream); }, stream);
      if (keys >= 12) {
        CHECK_EQ(dispatches, 1);
      } else {
        CHECK(dispatches > 1);
      }
    }
    require_hd_bit_identical(hd_sdpa(in, stream), hd_composition(in, keys, stream), stream);
  }

  // Past the 7168-key shared-memory stream bound the hd128 route must
  // refuse and the composition answers - identical words either way.
  HdCacheInputs long_cache = make_hd_cache(bfloat16, 7200, 7200, stream);
  if (fused_available) {
    uint64_t dispatches =
        dispatches_for([&] { return hd_sdpa(long_cache, stream); }, stream);
    CHECK(dispatches > 1);
  }
  require_hd_bit_identical(
      hd_sdpa(long_cache, stream), hd_composition(long_cache, 7200, stream), stream);
}

TEST_CASE("fused hd128 f16 decode is one dispatch and matches the composition") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  HdCacheInputs in = make_hd_cache(float16, 263, 320, stream);
  uint64_t dispatches = dispatches_for([&] { return hd_sdpa(in, stream); }, stream);
  CHECK_EQ(dispatches, 1);
  require_close(
      flat(hd_sdpa(in, stream), stream),
      flat(hd_composition(in, 263, stream), stream),
      0.01,
      "fused hd128 f16 decode vs f32-score composition");

  // Past the one-pass crossover the native-shape two-pass pair answers:
  // two dispatches, same composition agreement.
  HdCacheInputs long_in = make_hd_cache(float16, 1100, 1100, stream);
  uint64_t long_dispatches =
      dispatches_for([&] { return hd_sdpa(long_in, stream); }, stream);
  CHECK_EQ(long_dispatches, 2);
  require_close(
      flat(hd_sdpa(long_in, stream), stream),
      flat(hd_composition(long_in, 1100, stream), stream),
      0.01,
      "fused hd128 f16 two-pass vs f32-score composition");
}

TEST_CASE("hd128 decode shapes without a compiled width keep the composition") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // head_dim 100 is no multiple of 32: no compiled blob, the gate refuses,
  // and the composition answers within tolerance.
  auto q_values = pattern(kHdHeads * 100, 11);
  array q = astype(
      array(q_values.begin(), Shape{1, kHdHeads, 1, 100}, float32),
      bfloat16,
      stream);
  array k_cache = astype(
      array(pattern(kHdKvHeads * 320 * 100, 22).begin(),
            Shape{1, kHdKvHeads, 320, 100},
            float32),
      bfloat16,
      stream);
  array v_cache = astype(
      array(pattern(kHdKvHeads * 320 * 100, 33).begin(),
            Shape{1, kHdKvHeads, 320, 100},
            float32),
      bfloat16,
      stream);
  array k = slice(k_cache, {0, 0, 0, 0}, {1, kHdKvHeads, 93, 100}, stream);
  array v = slice(v_cache, {0, 0, 0, 0}, {1, kHdKvHeads, 93, 100}, stream);
  q.eval();
  k.eval();
  v.eval();
  omarchy::get_command_encoder(stream).synchronize();
  HdCacheInputs in{std::move(q), std::move(k), std::move(v)};
  uint64_t dispatches = dispatches_for([&] { return hd_sdpa(in, stream); }, stream);
  CHECK(dispatches > 1);
  require_close(
      flat(hd_sdpa(in, stream), stream),
      flat(hd_composition(in, 93, stream), stream),
      0.01,
      "head_dim 100 falls through to the composition");
}
