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
// lived.

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

// Exact fp64 attention reference over the bf16-widened operands.
Fp64Ref fp64_reference(const array& q, const array& k, const array& v, Stream stream) {
  auto qw = flat(q, stream);
  auto kw = flat(k, stream);
  auto vw = flat(v, stream);
  int b = q.shape(0), h = q.shape(1), lq = q.shape(2), lk = k.shape(2);
  float scale = 1.0f / std::sqrt(static_cast<float>(kHd));
  Fp64Ref ref;
  ref.out.resize(static_cast<size_t>(b) * h * lq * kHd);
  double accum_sq = 0.0;
  for (int bi = 0; bi < b; ++bi) {
    for (int hi = 0; hi < h; ++hi) {
      for (int i = 0; i < lq; ++i) {
        std::vector<double> scores(lk);
        double m = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < lk; ++j) {
          double dot = 0.0;
          for (int d = 0; d < kHd; ++d) {
            dot += static_cast<double>(
                       qw[((size_t)(bi * h + hi) * lq + i) * kHd + d]) *
                static_cast<double>(
                    kw[((size_t)(bi * h + hi) * lk + j) * kHd + d]);
          }
          scores[j] = dot * scale;
          m = std::max(m, scores[j]);
        }
        double sum = 0.0;
        for (int j = 0; j < lk; ++j) {
          scores[j] = std::exp(scores[j] - m);
          sum += scores[j];
        }
        for (int d = 0; d < kHd; ++d) {
          double o = 0.0;
          for (int j = 0; j < lk; ++j) {
            o += scores[j] *
                static_cast<double>(
                    vw[((size_t)(bi * h + hi) * lk + j) * kHd + d]);
          }
          o /= sum;
          size_t index = ((size_t)(bi * h + hi) * lq + i) * kHd + d;
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
TEST_CASE("flash defers to the chunked composed route on coopmat devices") {
  if (!compute_available()) return;
  Stream stream = gpu_stream();
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH");
  const auto& caps = omarchy::get_command_encoder(stream).device().capabilities();
  const bool composed_serves = caps.cooperative_matrix_f32_8 &&
      caps.subgroup_size == 32u;
  // 8 x 16384^2 = 2.15e9 score elements: past the 2^30 storage-binding
  // bound that pushed prefill onto the flash kernel (2026-10-05). The
  // chunked composed route caps its f32 score chunks at 1 GiB, so the
  // binding never exists and the route measures 3.2x over flash at the
  // H3 demo shape (782 vs 245 GMAC/s, 2026-10-09 TFProf receipt).
  constexpr int length = 16384;
  constexpr int heads = 8;
  const uint64_t flash_tiles = static_cast<uint64_t>((length + 31) / 32);
  array q = make_bf16({1, heads, length, kHd}, 701, stream);
  array k = make_bf16({1, heads, length, kHd}, 702, stream);
  array v = make_bf16({1, heads, length, kHd}, 703, stream);
  const uint64_t default_dispatches =
      dispatches_for([&] { return sdpa(q, k, v, stream); }, stream);
  std::vector<float> composed = flat(sdpa(q, k, v, stream), stream);
  setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH", "1", 1);
  const uint64_t pinned_dispatches =
      dispatches_for([&] { return sdpa(q, k, v, stream); }, stream);
  std::vector<float> flashed = flat(sdpa(q, k, v, stream), stream);
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH");
  if (composed_serves) {
    CHECK_NE(default_dispatches, flash_tiles);
  } else {
    // No coopmat shape: flash stays the long-sequence route.
    CHECK_EQ(default_dispatches, flash_tiles);
  }
  CHECK_EQ(pinned_dispatches, flash_tiles);
  REQUIRE(composed.size() == flashed.size());
  for (size_t i = 0; i < composed.size(); ++i) {
    if (std::abs(composed[i] - flashed[i]) > 0.02) {
      CHECK_MESSAGE(
          false,
          "composed vs flash element ",
          i,
          " at lq=",
          length,
          ": ",
          composed[i],
          " vs ",
          flashed[i]);
      break;
    }
  }
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

TEST_CASE("chunked composed prefill keeps GQA rows in place") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
  set_flash_enabled(false);

  struct Case {
    int heads;
    int kv_heads;
    int length;
  };
  // 192 rows with the cap below gives 64-row chunks (three per call).
  const Case cases[] = {
      {8, 8, 192}, // MHA control
      {8, 2, 192}, // GQA repeats = 4
      {6, 1, 192}, // GQA repeats = 6, one KV head
  };
  for (const auto& c : cases) {
    array q = make_bf16({1, c.heads, c.length, kHd}, 301 + c.heads, stream);
    array k = make_bf16({1, c.kv_heads, c.length, kHd}, 302 + c.heads, stream);
    array v = make_bf16({1, c.kv_heads, c.length, kHd}, 303 + c.heads, stream);
    const uint64_t chunk_denom =
        static_cast<uint64_t>(c.heads) * c.length * sizeof(float);
    unsetenv("MLX_OMARCHY_SDPA_CHUNK_MAX_BYTES");
    std::vector<float> whole = flat(sdpa(q, k, v, stream), stream);
    setenv(
        "MLX_OMARCHY_SDPA_CHUNK_MAX_BYTES",
        std::to_string(chunk_denom * 64).c_str(),
        1);
    std::vector<float> chunked = flat(sdpa(q, k, v, stream), stream);
    unsetenv("MLX_OMARCHY_SDPA_CHUNK_MAX_BYTES");
    REQUIRE_EQ(chunked.size(), whole.size());
    double max_diff = 0.0;
    for (size_t i = 0; i < whole.size(); ++i) {
      max_diff = std::max(
          max_diff, static_cast<double>(std::abs(chunked[i] - whole[i])));
    }
    CHECK_MESSAGE(
        max_diff == 0.0,
        "chunked vs whole composed, heads ",
        c.heads,
        "/",
        c.kv_heads,
        ": max abs diff ",
        max_diff);
  }
  set_flash_enabled(true);
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
