// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// SdpaPrefillFlashBF16Hd128, 2026-10-05. The flash-style bf16 prefill
// route (online softmax, no materialized scores) must agree with the
// f32-score composition it replaces on every tile-boundary shape, fall
// through when anything outside its contract is asked (masks, causal,
// f16, non-contiguous), keep its per-q-tile dispatch split, and be
// run-to-run identical. The gate is per-op error vs an fp64 reference
// no worse than the composed path's (docs/numerics-gate.md); online
// softmax reassociates the sum, so agreement is at bf16 storage
// granularity, not bit identity. Shapes cover the tile tails
// (16/17/31/32/33/48/53/64, ragged q<k) where the famqwen flash probe
// lived. The causal arm (MLX_OMARCHY_SDPA_FLASH_CAUSAL=1, hd128/hd256)
// is gated the same way against a causal fp64 reference.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/omarchy/encoder.h"
#include "mlx/backend/omarchy/trace.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/stream.h"

using namespace mlx::core;

namespace {

constexpr int kHd = 128;

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

array make_bf16(Shape shape, uint32_t seed, Stream stream) {
  auto values = pattern(static_cast<size_t>(shape[0]) * shape[1] * shape[2] *
                            shape[3],
                        seed);
  array a = astype(
      array(values.begin(), shape, float32), bfloat16, stream);
  a.eval();
  omarchy::get_command_encoder(stream).synchronize();
  return a;
}

std::vector<float> flat(const array& value, Stream stream) {
  array copy = astype(value, float32, stream);
  copy.eval();
  omarchy::get_command_encoder(stream).synchronize();
  const float* data = copy.data<float>();
  return std::vector<float>(data, data + copy.size());
}

void set_flash_enabled(bool enabled) {
  if (enabled) {
    unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH");
  } else {
    setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH", "0", 1);
  }
}

array sdpa(array q, array k, array v, Stream stream) {
  return fast::scaled_dot_product_attention(
      std::move(q),
      std::move(k),
      std::move(v),
      1.0f / std::sqrt(static_cast<float>(kHd)),
      "",
      std::nullopt,
      {},
      false,
      stream);
}

struct Fp64Ref {
  std::vector<double> out;
  double rel_l2 = 0.0;
};

// Exact fp64 attention reference over the bf16-widened operands. GQA:
// query head h reads kv head h / (heads / kv_heads). causal: row i sees
// keys [0, lk - lq + i] (MLX's bottom-right alignment).
Fp64Ref fp64_reference(
    const array& q,
    const array& k,
    const array& v,
    Stream stream,
    int hd = kHd,
    bool causal = false) {
  auto qw = flat(q, stream);
  auto kw = flat(k, stream);
  auto vw = flat(v, stream);
  int b = q.shape(0), h = q.shape(1), lq = q.shape(2), lk = k.shape(2);
  int hkv = k.shape(1);
  float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  Fp64Ref ref;
  ref.out.resize(static_cast<size_t>(b) * h * lq * hd);
  double accum_sq = 0.0;
  for (int bi = 0; bi < b; ++bi) {
    for (int hi = 0; hi < h; ++hi) {
      const size_t kv = static_cast<size_t>(bi * hkv + hi / (h / hkv));
      for (int i = 0; i < lq; ++i) {
        const int keys = causal ? lk - lq + i + 1 : lk;
        std::vector<double> scores(keys);
        double m = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < keys; ++j) {
          double dot = 0.0;
          for (int d = 0; d < hd; ++d) {
            dot += static_cast<double>(
                       qw[((size_t)(bi * h + hi) * lq + i) * hd + d]) *
                static_cast<double>(kw[(kv * lk + j) * hd + d]);
          }
          scores[j] = dot * scale;
          m = std::max(m, scores[j]);
        }
        double sum = 0.0;
        for (int j = 0; j < keys; ++j) {
          scores[j] = std::exp(scores[j] - m);
          sum += scores[j];
        }
        for (int d = 0; d < hd; ++d) {
          double o = 0.0;
          for (int j = 0; j < keys; ++j) {
            o += scores[j] * static_cast<double>(vw[(kv * lk + j) * hd + d]);
          }
          o /= sum;
          size_t index = ((size_t)(bi * h + hi) * lq + i) * hd + d;
          ref.out[index] = o;
          accum_sq += o * o;
        }
      }
    }
  }
  double norm = std::sqrt(accum_sq);
  ref.rel_l2 = norm;
  return ref;
}

// Relative-L2 error of a bf16 output against the fp64 reference.
double rel_l2_error(
    const std::vector<float>& got,
    const Fp64Ref& ref) {
  double sq = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    double diff = static_cast<double>(got[i]) - ref.out[i];
    sq += diff * diff;
  }
  return std::sqrt(sq) / ref.rel_l2;
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
TEST_CASE("flash bf16 prefill defaults to score-memory safety, not sequence length") {
  if (!compute_available()) return;
  Stream stream = gpu_stream();
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
  set_flash_enabled(true);
  constexpr int length = 4096;
  array q = make_bf16({1, 4, length, kHd}, 701, stream);
  array k = make_bf16({1, 4, length, kHd}, 702, stream);
  array v = make_bf16({1, 4, length, kHd}, 703, stream);
  const uint64_t default_dispatches =
      dispatches_for([&] { return sdpa(q, k, v, stream); }, stream);
  setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L", "1", 1);
  const uint64_t forced_dispatches =
      dispatches_for([&] { return sdpa(q, k, v, stream); }, stream);
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
  const uint64_t flash_tiles = static_cast<uint64_t>((length + 31) / 32);
  const uint64_t score_elements = static_cast<uint64_t>(4) * length * length;
  const auto& caps = omarchy::get_command_encoder(stream).device().capabilities();
  const bool should_flash = score_elements > (1ull << 30) ||
      (caps.total_memory > 0 && score_elements > caps.total_memory / 16);
  CHECK_EQ(default_dispatches == flash_tiles, should_flash);
  CHECK_EQ(forced_dispatches, flash_tiles);
}


} // namespace

TEST_CASE("flash bf16 prefill matches the composition on tile-boundary shapes") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  // Engage the flash route on tiny shapes: the env floor is read per
  // call, so the whole tile-tail matrix rides the real kernel.
  setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L", "1", 1);

  const int heads = 4;
  const int lengths[] = {16, 17, 31, 32, 33, 48, 53, 64, 200, 384};
  for (int lq : lengths) {
    int lk = lq;
    array q = make_bf16({1, heads, lq, kHd}, 11 + lq, stream);
    array k = make_bf16({1, heads, lk, kHd}, 22 + lq, stream);
    array v = make_bf16({1, heads, lk, kHd}, 33 + lq, stream);

    auto ref = fp64_reference(q, k, v, stream);
    std::vector<float> flash = flat(sdpa(q, k, v, stream), stream);
    set_flash_enabled(false);
    std::vector<float> composed = flat(sdpa(q, k, v, stream), stream);
    set_flash_enabled(true);

    double flash_err = rel_l2_error(flash, ref);
    double composed_err = rel_l2_error(composed, ref);
    CHECK_MESSAGE(
        flash_err <= composed_err * 1.5 + 0.005,
        "flash rel-L2 ",
        flash_err,
        " vs composed ",
        composed_err,
        " at lq=",
        lq);
    CHECK_MESSAGE(
        flash_err <= 0.02,
        "flash rel-L2 ",
        flash_err,
        " exceeds the absolute cap at lq=",
        lq);
    for (size_t i = 0; i < flash.size(); ++i) {
      if (std::abs(flash[i] - composed[i]) > 0.02) {
        CHECK_MESSAGE(
            false,
            "flash vs composed element ",
            i,
            " at lq=",
            lq,
            ": ",
            flash[i],
            " vs ",
            composed[i]);
        break;
      }
    }
  }
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
}

TEST_CASE("flash prefill handles ragged and GQA shapes like the composition") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L", "1", 1);

  struct Case {
    int heads;
    int kv_heads;
    int lq;
    int lk;
  };
  const Case cases[] = {
      {4, 4, 33, 64}, // ragged q < k, tail q tile
      {4, 2, 32, 32}, // GQA repeats = 2
      {4, 1, 17, 48}, // GQA repeats = 4, ragged both ways
      {56, 56, 32, 32}, // H3-width head grid in one dispatch
  };
  for (const auto& c : cases) {
    array q = make_bf16({1, c.heads, c.lq, kHd}, 44 + c.lq, stream);
    array k = make_bf16({1, c.kv_heads, c.lk, kHd}, 55 + c.lk, stream);
    array v = make_bf16({1, c.kv_heads, c.lk, kHd}, 66 + c.lk, stream);

    std::vector<float> flash = flat(sdpa(q, k, v, stream), stream);
    set_flash_enabled(false);
    std::vector<float> composed = flat(sdpa(q, k, v, stream), stream);
    set_flash_enabled(true);
    for (size_t i = 0; i < flash.size(); ++i) {
      if (std::abs(flash[i] - composed[i]) > 0.02) {
        CHECK_MESSAGE(
            false,
            "flash vs composed element ",
            i,
            " (heads ",
            c.heads,
            "/",
            c.kv_heads,
            " lq ",
            c.lq,
            " lk ",
            c.lk,
            "): ",
            flash[i],
            " vs ",
            composed[i]);
        break;
      }
    }
  }

  // batch 2: the kernel's batch axis rides gl_WorkGroupID.z.
  {
    int l = 70;
    array qb = make_bf16({2, 4, l, kHd}, 88, stream);
    array kb = make_bf16({2, 4, l, kHd}, 89, stream);
    array vb = make_bf16({2, 4, l, kHd}, 90, stream);
    std::vector<float> flash_b = flat(sdpa(qb, kb, vb, stream), stream);
    set_flash_enabled(false);
    std::vector<float> composed_b = flat(sdpa(qb, kb, vb, stream), stream);
    set_flash_enabled(true);
    for (size_t i = 0; i < flash_b.size(); ++i) {
      if (std::abs(flash_b[i] - composed_b[i]) > 0.02) {
        CHECK_MESSAGE(
            false, "batch-2 flash vs composed element ", i);
        break;
      }
    }
  }
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
}

TEST_CASE("flash prefill splits dispatches per q tile") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L", "1", 1);
  const int heads = 4;
  array q = make_bf16({1, heads, 64, kHd}, 77, stream);
  array k = make_bf16({1, heads, 64, kHd}, 78, stream);
  array v = make_bf16({1, heads, 64, kHd}, 79, stream);
  uint64_t flash_dispatches =
      dispatches_for([&] { return sdpa(q, k, v, stream); }, stream);
  CHECK_EQ(flash_dispatches, 2u); // ceil(64/32) q tiles

  array q33 = make_bf16({1, heads, 33, kHd}, 87, stream);
  uint64_t ragged_dispatches = dispatches_for(
      [&] { return sdpa(q33, k, v, stream); }, stream);
  CHECK_EQ(ragged_dispatches, 2u);
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
}

TEST_CASE("flash prefill is run-to-run identical") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L", "1", 1);
  const int heads = 4;
  array q = make_bf16({1, heads, 53, kHd}, 91, stream);
  array k = make_bf16({1, heads, 53, kHd}, 92, stream);
  array v = make_bf16({1, heads, 53, kHd}, 93, stream);
  std::vector<float> first = flat(sdpa(q, k, v, stream), stream);
  for (int run = 0; run < 2; ++run) {
    std::vector<float> again = flat(sdpa(q, k, v, stream), stream);
    CHECK(std::memcmp(first.data(), again.data(), first.size() * sizeof(float)) ==
          0);
  }
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
}

TEST_CASE("masked and causal prefill keep the composition") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L", "1", 1);
  const int heads = 4;
  const int lq = 48;
  array q = make_bf16({1, heads, lq, kHd}, 101, stream);
  array k = make_bf16({1, heads, lq, kHd}, 102, stream);
  array v = make_bf16({1, heads, lq, kHd}, 103, stream);

  // Additive mask: flash must refuse (inputs 4) and compose.
  auto mask_values = pattern(heads * lq * lq, 104);
  array mask = astype(
      array(mask_values.begin(), Shape{1, heads, lq, lq}, float32),
      bfloat16,
      stream);
  mask.eval();
  omarchy::get_command_encoder(stream).synchronize();
  auto with_mask = [&](Stream s) {
    return fast::scaled_dot_product_attention(
        q, k, v, 1.0f / std::sqrt(static_cast<float>(kHd)), "", mask, {}, false, s);
  };
  uint64_t masked_dispatches =
      dispatches_for([&] { return with_mask(stream); }, stream);
  CHECK(masked_dispatches > 2u); // not the 2-dispatch flash split
  std::vector<float> flash_env = flat(with_mask(stream), stream);
  set_flash_enabled(false);
  std::vector<float> composed_env = flat(with_mask(stream), stream);
  set_flash_enabled(true);
  for (size_t i = 0; i < flash_env.size(); ++i) {
    if (std::abs(flash_env[i] - composed_env[i]) > 0.02) {
      CHECK_MESSAGE(false, "masked route changed output at ", i);
      break;
    }
  }

  // Causal prefill: same refusal.
  uint64_t causal_dispatches = dispatches_for(
      [&] {
        return fast::scaled_dot_product_attention(
            q,
            k,
            v,
            1.0f / std::sqrt(static_cast<float>(kHd)),
            "causal",
            std::nullopt,
            {},
            false,
            stream);
      },
      stream);
  CHECK(causal_dispatches > 2u);
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
}

namespace {

array sdpa_causal(const array& q, const array& k, const array& v, int hd, Stream stream) {
  return fast::scaled_dot_product_attention(
      q, k, v, 1.0f / std::sqrt(static_cast<float>(hd)), "causal",
      std::nullopt, {}, false, stream);
}

double max_abs_error(const std::vector<float>& got, const Fp64Ref& ref) {
  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    worst = std::max(worst, std::abs(static_cast<double>(got[i]) - ref.out[i]));
  }
  return worst;
}

} // namespace

// MLX_OMARCHY_SDPA_FLASH_CAUSAL=1 (default off): causal bf16 prompts at
// hd128 and hd256 take the flash kernel built with SDPA_CAUSAL. Online
// softmax reassociates the sums, so the gate is error vs fp64 no worse
// than the composed route's, under absolute caps, not bit identity.
TEST_CASE("causal flash prefill matches the fp64 reference at hd128 and hd256") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  unsetenv("MLX_OMARCHY_SDPA_FLASH_DISPATCH_FLOPS");
  struct Case {
    int hd;
    int heads;
    int kv_heads;
    int lq;
    int lk;
  };
  const Case cases[] = {
      {128, 4, 4, 64, 64},   // two q tiles, diagonal tiles
      {128, 4, 2, 33, 33},   // GQA, tail q tile
      {128, 4, 4, 17, 48},   // chunked prefill: q < k, offset 31
      {256, 4, 4, 64, 64},   // four 16-row q tiles
      {256, 8, 2, 53, 53},   // Qwen3.8 head grid (8 / 2), tail tile
      {256, 4, 1, 16, 80},   // GQA 4, offset 64
      {256, 2, 2, 200, 200}, // many key tiles skipped per q tile
  };
  for (const auto& c : cases) {
    array q = make_bf16({1, c.heads, c.lq, c.hd}, 501 + c.lq, stream);
    array k = make_bf16({1, c.kv_heads, c.lk, c.hd}, 502 + c.lk, stream);
    array v = make_bf16({1, c.kv_heads, c.lk, c.hd}, 503 + c.lk, stream);
    auto ref = fp64_reference(q, k, v, stream, c.hd, true);

    setenv("MLX_OMARCHY_SDPA_FLASH_CAUSAL", "1", 1);
    std::vector<float> flash = flat(sdpa_causal(q, k, v, c.hd, stream), stream);
    // One batched dispatch proves the flash kernel ran, not the composition.
    const uint64_t flash_dispatches = dispatches_for(
        [&] { return sdpa_causal(q, k, v, c.hd, stream); }, stream);
    unsetenv("MLX_OMARCHY_SDPA_FLASH_CAUSAL");
    std::vector<float> composed =
        flat(sdpa_causal(q, k, v, c.hd, stream), stream);

    CHECK_EQ(flash_dispatches, 1u);
    const double flash_l2 = rel_l2_error(flash, ref);
    const double composed_l2 = rel_l2_error(composed, ref);
    const double flash_max = max_abs_error(flash, ref);
    const double composed_max = max_abs_error(composed, ref);
    CHECK_MESSAGE(flash_l2 <= composed_l2 * 1.5 + 0.005, "rel-L2 ", flash_l2,
                  " vs composed ", composed_l2, " at hd ", c.hd, " lq ", c.lq);
    CHECK_MESSAGE(flash_l2 <= 0.02, "rel-L2 ", flash_l2, " at hd ", c.hd,
                  " lq ", c.lq);
    CHECK_MESSAGE(flash_max <= composed_max * 2.0 + 1.0 / 256.0, "max-abs ",
                  flash_max, " vs composed ", composed_max, " at hd ", c.hd,
                  " lq ", c.lq);
    CHECK_MESSAGE(flash_max <= 1.0 / 32.0, "max-abs ", flash_max, " at hd ",
                  c.hd, " lq ", c.lq);
  }
}

TEST_CASE("causal flash prefill output does not depend on the dispatch split") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  setenv("MLX_OMARCHY_SDPA_FLASH_CAUSAL", "1", 1);
  for (int hd : {128, 256}) {
    const int length = 96;
    const uint64_t tiles = static_cast<uint64_t>(length / (hd == 256 ? 16 : 32));
    array q = make_bf16({1, 4, length, hd}, 601 + hd, stream);
    array k = make_bf16({1, 4, length, hd}, 602 + hd, stream);
    array v = make_bf16({1, 4, length, hd}, 603 + hd, stream);
    unsetenv("MLX_OMARCHY_SDPA_FLASH_DISPATCH_FLOPS");
    std::vector<float> batched = flat(sdpa_causal(q, k, v, hd, stream), stream);
    setenv("MLX_OMARCHY_SDPA_FLASH_DISPATCH_FLOPS", "1", 1);
    const uint64_t split_dispatches = dispatches_for(
        [&] { return sdpa_causal(q, k, v, hd, stream); }, stream);
    std::vector<float> split = flat(sdpa_causal(q, k, v, hd, stream), stream);
    unsetenv("MLX_OMARCHY_SDPA_FLASH_DISPATCH_FLOPS");
    CHECK_EQ(split_dispatches, tiles);
    CHECK(std::memcmp(batched.data(), split.data(),
                      batched.size() * sizeof(float)) == 0);
  }
  unsetenv("MLX_OMARCHY_SDPA_FLASH_CAUSAL");
}
