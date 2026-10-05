// Copyright © 2026 Joshua Warren / mlx-omarchy contributors.
// SPDX-License-Identifier: MIT

// SdpaPrefillFlashBF16Hd128 and its explicitly gated cooperative-matrix twin.
// Both routes use online softmax without materialized scores and must agree
// with the f32-score composition they replace on tile-boundary shapes. They
// must fall through for unsupported masks, causal mode, f16, and layouts.
// Per-op error must stay within the fp64/composed gates in docs/numerics-gate.md;
// online softmax reassociates sums, so agreement is at bf16 storage granularity,
// not bit identity. Cases cover tile tails (16/17/31/32/33/48/53/64, ragged
// q<k) where the famqwen flash probe lived.

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

void set_coopmat_enabled(bool enabled) {
  if (enabled) {
    setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_COOPMAT", "1", 1);
  } else {
    setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_COOPMAT", "0", 1);
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
  const int b = q.shape(0), h = q.shape(1), kv_heads = k.shape(1);
  const int lq = q.shape(2), lk = k.shape(2);
  const int kv_repeats = h / kv_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHd));
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
                    kw[((size_t)(bi * kv_heads + hi / kv_repeats) * lk + j) *
                       kHd + d]);
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
                    vw[((size_t)(bi * kv_heads + hi / kv_repeats) * lk + j) *
                       kHd + d]);
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

// Sampled-row fp64 attention at the H3 shape: full-K softmax per
// sampled row only, so L=13365 stays tractable in a test. Rows sample
// the ragged q-tile tails and the deep interior.
double sampled_h3_rel_l2(
    const std::vector<float>& got,
    const array& q,
    const array& k,
    const array& v,
    Stream stream) {
  const auto qw = flat(q, stream);
  const auto kw = flat(k, stream);
  const auto vw = flat(v, stream);
  const int heads = q.shape(1);
  const int lq = q.shape(2);
  const int lk = k.shape(2);
  const int kv_heads = k.shape(1);
  const int repeats = heads / kv_heads;
  const int sample_rows[] = {0, 1, 128, 4096, 8192, 13364};
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHd));
  std::vector<double> scores(lk);
  double error_sq = 0.0;
  double norm_sq = 0.0;
  for (int hi = 0; hi < heads; ++hi) {
    for (int row : sample_rows) {
      if (row >= lq) {
        continue;
      }
      double max_score = -std::numeric_limits<double>::infinity();
      for (int j = 0; j < lk; ++j) {
        double dot = 0.0;
        for (int d = 0; d < kHd; ++d) {
          const size_t qi = (static_cast<size_t>(hi) * lq + row) * kHd + d;
          const size_t ki =
              (static_cast<size_t>(hi / repeats) * lk + j) * kHd + d;
          dot += static_cast<double>(qw[qi]) * static_cast<double>(kw[ki]);
        }
        scores[j] = dot * scale;
        max_score = std::max(max_score, scores[j]);
      }
      double sum = 0.0;
      for (double& score : scores) {
        score = std::exp(score - max_score);
        sum += score;
      }
      for (int d = 0; d < kHd; ++d) {
        double expected = 0.0;
        for (int j = 0; j < lk; ++j) {
          const size_t vi =
              (static_cast<size_t>(hi / repeats) * lk + j) * kHd + d;
          expected += scores[j] * static_cast<double>(vw[vi]);
        }
        expected /= sum;
        const size_t oi = (static_cast<size_t>(hi) * lq + row) * kHd + d;
        const double diff = static_cast<double>(got[oi]) - expected;
        error_sq += diff * diff;
        norm_sq += expected * expected;
      }
    }
  }
  return std::sqrt(error_sq / norm_sq);
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

bool flash_coopmat_device_ready(Stream stream) {
  const auto& caps = omarchy::get_command_encoder(stream).device().capabilities();
  if (!caps.cooperative_matrix_f32_8 || caps.subgroup_size != 32u ||
      caps.max_compute_work_group_size[0] < 256u ||
      caps.max_compute_work_group_invocations < 256u ||
      caps.max_compute_shared_memory_size < 28928u) {
    printf("Skipping: no coopmat device meeting flash route workgroup/shared limits\n");
    return false;
  }
  return true;
}

TEST_CASE("coopmat flash bf16 prefill matches fp64 on ragged tiles") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  if (!flash_coopmat_device_ready(stream)) {
    return;
  }
  setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L", "1", 1);
  set_flash_enabled(true);
  const int lengths[] = {16, 17, 31, 32, 33, 48, 53, 64, 200, 384};
  for (int length : lengths) {
    set_flash_enabled(true);
    array q = make_bf16({1, 2, length, kHd}, 101 + length, stream);
    array k = make_bf16({1, 2, length, kHd}, 201 + length, stream);
    array v = make_bf16({1, 2, length, kHd}, 301 + length, stream);
    const auto ref = fp64_reference(q, k, v, stream);

    set_coopmat_enabled(true);
    const auto coop = flat(sdpa(q, k, v, stream), stream);
    const auto coop_repeat_1 = flat(sdpa(q, k, v, stream), stream);
    const auto coop_repeat_2 = flat(sdpa(q, k, v, stream), stream);
    set_coopmat_enabled(false);

    const auto scalar = flat(sdpa(q, k, v, stream), stream);
    set_flash_enabled(false);
    const auto composed = flat(sdpa(q, k, v, stream), stream);

    const double coop_err = rel_l2_error(coop, ref);
    const double scalar_err = rel_l2_error(scalar, ref);
    const double composed_err = rel_l2_error(composed, ref);
    CHECK_MESSAGE(coop == coop_repeat_1,
        "coopmat output differs from run 2 at L=", length);
    CHECK_MESSAGE(coop == coop_repeat_2,
        "coopmat output differs from run 3 at L=", length);
    CHECK_MESSAGE(coop_err <= scalar_err + 1e-6,
        "coopmat rel-L2 ", coop_err, " exceeds scalar ", scalar_err,
        " at L=", length);
    CHECK_MESSAGE(coop_err <= composed_err + 1e-6,
        "coopmat rel-L2 ", coop_err, " exceeds composed ", composed_err,
        " at L=", length);
  }
  array q_gqa = make_bf16({2, 4, 17, kHd}, 417, stream);
  array k_gqa = make_bf16({2, 2, 33, kHd}, 418, stream);
  array v_gqa = make_bf16({2, 2, 33, kHd}, 419, stream);
  const auto gqa_ref = fp64_reference(q_gqa, k_gqa, v_gqa, stream);
  set_flash_enabled(true);
  set_coopmat_enabled(true);
  const auto gqa_coop = flat(sdpa(q_gqa, k_gqa, v_gqa, stream), stream);
  const auto gqa_repeat_1 = flat(sdpa(q_gqa, k_gqa, v_gqa, stream), stream);
  const auto gqa_repeat_2 = flat(sdpa(q_gqa, k_gqa, v_gqa, stream), stream);
  set_coopmat_enabled(false);
  const auto gqa_scalar = flat(sdpa(q_gqa, k_gqa, v_gqa, stream), stream);
  set_flash_enabled(false);
  const auto gqa_composed = flat(sdpa(q_gqa, k_gqa, v_gqa, stream), stream);
  const double gqa_coop_err = rel_l2_error(gqa_coop, gqa_ref);
  CHECK(gqa_coop == gqa_repeat_1);
  CHECK(gqa_coop == gqa_repeat_2);
  CHECK(gqa_coop_err <= rel_l2_error(gqa_scalar, gqa_ref) + 1e-6);
  CHECK(gqa_coop_err <= rel_l2_error(gqa_composed, gqa_ref) + 1e-6);
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_COOPMAT");
  set_flash_enabled(true);
}

TEST_CASE("coopmat H3-length bf16 prefill matches sampled fp64 rows") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  if (!flash_coopmat_device_ready(stream)) {
    return;
  }
  constexpr int length = 13365;
  array q = make_bf16({1, 56, length, kHd}, 5601, stream);
  array k = make_bf16({1, 56, length, kHd}, 5602, stream);
  array v = make_bf16({1, 56, length, kHd}, 5603, stream);
  setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L", "1", 1);
  set_flash_enabled(true);
  set_coopmat_enabled(true);
  const auto coop = flat(sdpa(q, k, v, stream), stream);
  const auto repeat_1 = flat(sdpa(q, k, v, stream), stream);
  const auto repeat_2 = flat(sdpa(q, k, v, stream), stream);
  set_coopmat_enabled(false);
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
  const double err = sampled_h3_rel_l2(coop, q, k, v, stream);
  CHECK(coop == repeat_1);
  CHECK(coop == repeat_2);
  CHECK_MESSAGE(err <= 0.00167, "sampled H3 rel-L2 exceeds scalar target: ", err);
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_COOPMAT");
  set_flash_enabled(true);
}

TEST_CASE("coopmat flash bf16 prefill stays within composed error at L<=4096") {
  if (!compute_available()) {
    return;
  }
  Stream stream = gpu_stream();
  if (!flash_coopmat_device_ready(stream)) {
    return;
  }
  setenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L", "1", 1);
  const int lengths[] = {2048, 4096};
  for (int length : lengths) {
    array q = make_bf16({1, 2, length, kHd}, 2201 + length, stream);
    array k = make_bf16({1, 2, length, kHd}, 3201 + length, stream);
    array v = make_bf16({1, 2, length, kHd}, 4201 + length, stream);
    set_flash_enabled(true);
    set_coopmat_enabled(true);
    const auto coop = flat(sdpa(q, k, v, stream), stream);
    set_coopmat_enabled(false);
    set_flash_enabled(false);
    const auto composed = flat(sdpa(q, k, v, stream), stream);
    const double coop_err = sampled_h3_rel_l2(coop, q, k, v, stream);
    const double composed_err = sampled_h3_rel_l2(composed, q, k, v, stream);
    CHECK_MESSAGE(coop_err <= composed_err + 1e-6,
        "coopmat sampled rel-L2 ", coop_err, " exceeds composed ",
        composed_err, " at L=", length);
  }
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_MIN_L");
  unsetenv("MLX_OMARCHY_SDPA_PREFILL_FLASH_COOPMAT");
  set_flash_enabled(true);
}

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
